from torch.nn import LSTM
from torch.autograd import Variable

import torch

class DualLSTM(torch.nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim,
            dropout=0.0, output_projection_dim=None, layers=1,
            bidirectional=False, to_cuda=True):
        super(DualLSTM, self).__init__()

        use_bias = True
        num_directions = (1 if not bidirectional else 2)



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

        self.h02 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        self.c02 = Variable(torch.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        if to_cuda:
            self.h01 = self.h01.cuda()
            self.c01 = self.c01.cuda()
            self.h02 = self.h02.cuda()
            self.c02 = self.c02.cuda()

        self.h01.data.zero_()
        self.c01.data.zero_()
        self.h02.data.zero_()
        self.c02.data.zero_()


    def forward(self, seq1, seq2):
        out1, (h1_n, c1_n) = self.lstm1(seq1, (self.h01, self.c01))
        out2, (h2_n, c2_n) = self.lstm2(seq2, (self.h02, self.c02))
        return h1_n, h2_n
