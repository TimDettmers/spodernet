from torch.nn import LSTM
from torch.autograd import Variable

import torch
import torch.nn.functional as F

from spodernet.interfaces import AbstractModel
from spodernet.utils.global_config import Config

class Embedding(torch.nn.Module, AbstractModel):
    def __init__(self, embedding_size, num_embeddings):
        super(Embedding, self).__init__()

        self.emb= torch.nn.Embedding(num_embeddings,
                embedding_size, padding_idx=0)#, scale_grad_by_freq=True, padding_idx=0)

    def forward(self, feed_dict, *args):
        embedded_results = []
        if 'input' in feed_dict:
            embedded_results.append(self.emb(feed_dict['input']))

        if 'support' in feed_dict:
            embedded_results.append(self.emb(feed_dict['support']))

        return embedded_results


class PairedBiDirectionalLSTM(torch.nn.Module, AbstractModel):
    def __init__(self, batch_size, input_dim, hidden_dim,
            dropout=0.0, layers=1,
            bidirectional=True, to_cuda=False, conditional_encoding=True):
        super(PairedBiDirectionalLSTM, self).__init__()

        use_bias = True
        num_directions = (1 if not bidirectional else 2)



        self.conditional_encoding = conditional_encoding
        self.lstm1 = LSTM(input_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)
        self.lstm2 = LSTM(input_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)

        self.h01 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        self.c01 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))

        if Config.cuda:
            self.h01 = self.h01.cuda()
            self.c01 = self.c01.cuda()

        if not self.conditional_encoding:
            self.h02 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                    hidden_dim))
            self.c02 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                    hidden_dim))
            if to_cuda:
                self.h02 = self.h02.cuda()
                self.c02 = self.c02.cuda()




    def forward(self, feed_dict, *args):
        seq1, seq2 = args
        if self.conditional_encoding:
            self.h01.data.zero_()
            self.c01.data.zero_()
            out1, hid1 = self.lstm1(seq1, (self.h01, self.c01))
            out2, hid2 = self.lstm2(seq2, hid1)
        else:
            self.h01.data.zero_()
            self.c01.data.zero_()
            self.h02.data.zero_()
            self.c02.data.zero_()
            out1, hid1 = self.lstm1(seq1, (self.h01, self.c01))
            out2, hid2 = self.lstm2(seq2, (self.h02, self.c02))
        return [out1, out2]

class VariableLengthOutputSelection(torch.nn.Module, AbstractModel):
    def __init__(self):
        super(VariableLengthOutputSelection, self).__init__()
        self.b1 = None
        self.b2 = None

    def forward(self, feed_dict, *args):
        output_lstm1, output_lstm2 = args
        l1, l2 = feed_dict['input_length'], feed_dict['support_length']
        if self.b1 == None:
            b1 = torch.ByteTensor(output_lstm1.size())
            b2 = torch.ByteTensor(output_lstm2.size())
            if Config.cuda:
                b1 = b1.cuda()
                b2 = b2.cuda()

        b1.fill_(0)
        for i, num in enumerate(l1.data):
            b1[i,num-1,:] = 1
        out1 = output_lstm1[b1].view(Config.batch_size, -1)

        b2.fill_(0)
        for i, num in enumerate(l2.data):
            b2[i,num-1,:] = 1
        out2 = output_lstm2[b2].view(Config.batch_size, -1)

        out = torch.cat([out1,out2], 1)
        return [out]

class SoftmaxCrossEntropy(torch.nn.Module, AbstractModel):

    def __init__(self, input_dim, num_labels):
        super(SoftmaxCrossEntropy, self).__init__()
        self.num_labels = num_labels
        self.projection_to_labels = torch.nn.Linear(input_dim*4, num_labels)

    def forward(self, feed_dict, *args):
        outputs_prev_layer = args[0]
        t = feed_dict['target']

        logits = self.projection_to_labels(outputs_prev_layer)
        out = F.log_softmax(logits)
        loss = F.nll_loss(out, t)
        maximum, argmax = torch.topk(out.data, 1)

        return [logits, loss, argmax]

