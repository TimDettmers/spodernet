from torch.nn import LSTM
from torch.autograd import Variable

import torch

from spodernet.interfaces import AbstractModel

class Embedding(torch.nn.Module, AbstractModel):
    def __init__(self, embedding_size, num_embeddings):
        super(Embedding, self).__init__()

        self.emb= torch.nn.Embedding(num_embeddings,
                embedding_size, padding_idx=0)#, scale_grad_by_freq=True, padding_idx=0)

    def forward(self, *args):
        inputs, support = args
        input_seq = self.emb(inputs)
        support_seq = self.emb(support)

        return [input_seq, support_seq]


class PairedBiDirectionalLSTM(torch.nn.Module, torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim,
            dropout=0.0, output_projection_dim=None, layers=1,
            bidirectional=False, to_cuda=False, conditional_encoding=True):
        super(DualLSTM, self).__init__()

        use_bias = True
        num_directions = (1 if not bidirectional else 2)



        self.conditional_encoding = conditional_encoding
        self.lstm1 = LSTM(input_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)
        self.lstm2 = LSTM(input_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)

        #self.h0s = (Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True),
        #Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True))
        #self.c0s = (Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True),
        #            Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True))
        self.h01 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        self.c01 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))

        if to_cuda:
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




    def forward(self, *args):
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
        return [(out1, out2), (hid1, hid2)]

class SNLIClassification(torch.nn.Module):
    def __init__(self, hidden_size, scope=None, conditional_encoding=True):
        super(PairedBiDirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.scope = scope
        if not conditional_encoding:
            raise NotImplementedError("conditional_encoding=False is not implemented yet.")
    def __init__(self, batch_size, vocab, use_cuda=False):
        super(SNLIClassification, self).__init__()
        self.batch_size = batch_size
        input_dim = 256
        hidden_dim = 128
        num_directions = 2
        layers = 1
        self.projection_to_labels = torch.nn.Linear(2 * num_directions * hidden_dim, 3)
        self.dual_lstm = DualLSTM(self.batch_size,input_dim,
                hidden_dim,layers=layers,
                bidirectional=True,to_cuda=use_cuda )
        #self.init_weights()

        #print(i.size(1), input_dim)
        self.b1 = None
        self.b2 = None
        self.use_cuda = use_cuda

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.projection_to_labels.weight.data.uniform_(-initrange, initrange)

    def forward_to_output(self, input_seq, support_seq, inp_len, sup_len, targets):
        inputs = self.emb(input_seq)
        support = self.emb(support_seq)
        #inputs_packed = rnn_utils.pack_padded_sequence(inputs, inp_len.data.tolist(), True)
        #support_packed = rnn_utils.pack_padded_sequence(support, sup_len.data.tolist(), True)
        #(out_all1_packed, out_all2_packed), (h1, h2) = self.dual_lstm(inputs_packed, support_packed)
        #out_all1, lengths = rnn_utils.pad_packed_sequence(out_all1_packed, True)
        #out_all2, lengths = rnn_utils.pad_packed_sequence(out_all2_packed, True)
        (out_all1, out_all2), (h1, h2) = self.dual_lstm(inputs, support)

        if self.b1 == None:
            b1 = torch.ByteTensor(out_all1.size())
            b2 = torch.ByteTensor(out_all2.size())
            if self.use_cuda:
                b1 = b1.cuda()
                b2 = b2.cuda()
        #out1 = torch.index_select(out_all1,1,inp_len)
        #out2 = torch.index_select(out_all2,1,sup_len)
        b1.fill_(0)
        for i, num in enumerate(inp_len.data):
            b1[i,num-1,:] = 1
        out1 = out_all1[b1].view(self.batch_size,-1)

        b2.fill_(0)
        for i, num in enumerate(sup_len.data):
            b2[i,num-1,:] = 1
        out2 = out_all2[b2].view(self.batch_size,-1)

        out = torch.cat([out1,out2],1)

        #out1 = torch.transpose(out1,1,0).resize(self.b.batch_size,4*256)
        #out2 = torch.transpose(out2,1,0).resize(self.b.batch_size,4*256)
        #out1 = out1.view(self.b.batch_size,-1)
        #out2 = out2.view(self.b.batch_size,-1)
        #output = torch.cat([out1, out2],1)
        #output = torch.cat([out1[0], out2[0]],1)
        projected = self.projection_to_labels(out)
        #print(output)
        pred = F.log_softmax(projected)
        #print(pred[0:5])
        return pred

class DualLSTM(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim,
            dropout=0.0, output_projection_dim=None, layers=1,
            bidirectional=False, to_cuda=False, conditional_encoding=True):
        super(DualLSTM, self).__init__()

        use_bias = True
        num_directions = (1 if not bidirectional else 2)



        self.conditional_encoding = conditional_encoding
        self.lstm1 = LSTM(input_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)
        self.lstm2 = LSTM(input_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)

        #self.h0s = (Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True),
        #Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True))
        #self.c0s = (Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True),
        #            Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
        #        hidden_dim), requires_grad=True))
        self.h01 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        self.c01 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))

        if to_cuda:
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




    def forward(self, seq1, seq2):
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
        return [(out1, out2), (hid1, hid2)]
