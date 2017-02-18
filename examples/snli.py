'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function

from spodernet.data.snli2spoder import snli2spoder
from spodernet.preprocessing import spoder2hdf5
from spodernet.util import hdf2numpy
from spodernet.preprocessing.batching import Batcher

import torch
from torch.nn import LSTM
from torch.autograd import Variable

class SNLIClassification(torch.nn.Module):
    def __init__(self, datasets, vocab):
        super(SNLIClassification, self).__init__()
        self.b = Batcher(datasets)

        batch_size = self.b.batch_size
        emb_dim = 256
        hidden_dim = 64
        layers = 2
        dropout = 0.2
        use_bias = True
        bidirectional = True
        num_directions = (1 if not bidirectional else 2)

        emb = torch.nn.Embedding(vocab.num_embeddings, emb_dim)
        self.emb = emb


        self.lstm = LSTM(emb_dim,hidden_dim,layers,
                         use_bias,True,0.2,bidirectional)
        self.h0 = Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        self.c0 = Variable(torch.cuda.FloatTensor(layers * num_directions, batch_size,
                hidden_dim))
        self.projection_to_labels = torch.nn.Linear(num_directions * hidden_dim, 3)
        self.cross_entropy_softmax = torch.nn.CrossEntropyLoss()
        self.argmax = torch.nn.Softmax

    def forward_to_output(self, input_seq, support_seq, targets):
        inputs = self.emb(input_seq)
        #support = self.emb(support_seq)
        self.h0.data.zero_()
        self.c0.data.zero_()
        outputs, (h_n, c_n) = self.lstm(inputs, (self.h0, self.c0))
        output = outputs[:,-1]
        projected = self.projection_to_labels(output)
        loss = self.cross_entropy_softmax(projected, targets)
        return loss


    def train(self):
        epochs = 5
        optimizer = torch.optim.Adagrad(self.parameters(), lr=0.001)
        for epoch in range(epochs):
            for inp, sup, t in self.b:
                optimizer.zero_grad()
                loss = self.forward_to_output(inp, sup, t)
                loss.backward()
                optimizer.step()
                print(loss)




def main():
    # load data
    names, file_paths = snli2spoder()

    # tokenize and convert to hdf5
    lower_list = [True for name in names]
    add_to_vocab_list = [name != 'test' for name in names]
    filetype = spoder2hdf5.SINGLE_INPUT_SINGLE_SUPPORT_CLASSIFICATION
    hdf5paths, vocab = spoder2hdf5.file2hdf(
        file_paths, names, lower_list, add_to_vocab_list, filetype)

    X, S, T = hdf5paths[0]
    datasets = [hdf2numpy(X), hdf2numpy(S), hdf2numpy(T)]
    snli = SNLIClassification(datasets, vocab)
    snli = snli.cuda()
    snli.train()

if __name__ == '__main__':
    main()
