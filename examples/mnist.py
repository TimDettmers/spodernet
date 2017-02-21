'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function

from spodernet.data.snli2spoder import snli2spoder
from spodernet.preprocessing import spoder2hdf5
from spodernet.util import hdf2numpy
from spodernet.preprocessing.batching import Batcher
from spodernet.hooks import AccuracyHook
from spodernet.models import DualLSTM

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import LSTM

class SNLIClassification(torch.nn.Module):
    def __init__(self, datasets, vocab):
        super(SNLIClassification, self).__init__()
        self.b = Batcher(datasets)
        input_dim = 28
        hidden_dim = 128
        num_directions = 1
        self.emb= torch.nn.Embedding(vocab.num_embeddings,
                input_dim, scale_grad_by_freq=True )
        self.projection_to_labels = torch.nn.Linear( num_directions * hidden_dim, 3)
        self.cross_entropy_softmax = torch.nn.CrossEntropyLoss()
        self.dual_lstm = DualLSTM(self.b.batch_size,input_dim, hidden_dim)
        self.init_weights()
        self.lstm = LSTM(28, 128, 1, True, True)

        #print(i.size(1), input_dim)
        self.proj1 = torch.nn.Linear(input_dim, hidden_dim)
        #self.proj2 = torch.nn.Linear(s.size(1)*input_dim, hidden_dim)

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.projection_to_labels.weight.data.uniform_(-initrange, initrange)

    def forward_to_output(self, input_seq, support_seq, target):
        #inputs = self.emb(input_seq)
        #support = self.emb(support_seq)
        #out1, out2 = self.dual_lstm(input_seq, support_seq)
        #output = torch.cat([out1[0], out2[0]],1)
        h0 = Variable(torch.cuda.FloatTensor(1, 128,
                128))
        c0 = Variable(torch.cuda.FloatTensor(1, 128,
                128))
        out, (hn, cn) = self.lstm(input_seq, (h0,c0))
        #inputs2 = inputs.view(-1, inputs.size(1) * inputs.size(2))
        #support2 = support.view(-1, support.size(1) * support.size(2))
        #output1 = self.proj1(inputs2)
        #output2 = self.proj2(support2)
        #output = torch.cat([output1, output2],1)
        hn = hn.view(128,128)
        projected = self.projection_to_labels(hn)
        #projected = self.proj1(input_seq)
        pred = F.log_softmax(projected)
        #print(pred[0:5])
        maxiumum, argmax = torch.topk(pred.data, 1)
        return pred, argmax

def train_model(self):
    epochs = 5
    hook = AccuracyHook('Train')
    self.b.add_hook(hook)
    crit = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(self.parameters(), lr=0.001)
    for epoch in range(epochs):
        #self.b.shuffle()
        #for inp, sup, t in self.b:
        for inp, sup, t in self.b:
            #optimizer.zero_grad()
            pred, argmax = self.forward_to_output(inp, sup, t)
            loss = F.nll_loss(pred, t)
            loss.backward()
            optimizer.step()


            self.b.add_to_hook_histories(t, argmax)


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
    
    datasets = [
        np.float32(hdf2numpy('/home/tim/data/mnist/X.hdf5',
            'Default')).reshape(-1, 28, 28),
        np.float32(hdf2numpy('/home/tim/data/mnist/X.hdf5',
            'Default')).reshape(-1, 28, 28),
                np.int64(hdf2numpy('/home/tim/data/mnist/y.hdf5',
                    'Default')).flatten()]
    
    #datasets = [hdf2numpy(X), hdf2numpy(S), hdf2numpy(T)]
    #inp, sup, t = datasets
    #for row in range(10):
    #    hypo = ''
    #    premise = ''
    #    for idx in inp[row]:
    #        premise += vocab.get_word(idx) + ' '
    #    for idx in sup[row]:
    #        hypo += vocab.get_word(idx) + ' '
    #    print([premise, hypo, t[row]])
    #rdm = np.random.RandomState(23435)
    #idx = np.arange(datasets[0].shape[0])
    #print(idx.shape)
    #rdm.shuffle(idx)
    #for ds in datasets:
    #    ds = ds[idx]

    snli = SNLIClassification(datasets, vocab)
    snli.cuda()
    snli.train()
    train_model(snli)

if __name__ == '__main__':
    main()


