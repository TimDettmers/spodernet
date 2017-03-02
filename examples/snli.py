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

class SNLIClassification(torch.nn.Module):
    def __init__(self, datasets, vocab, use_cuda=False):
        super(SNLIClassification, self).__init__()
        self.b = Batcher(datasets, batch_size=128, transfer_to_gpu=use_cuda, num_print_thresholds=100)
        i, s, _, _, t = self.b.datasets
        print(s.size())
        input_dim = 256
        hidden_dim = 128
        num_directions = 1
        layers = 1
        self.emb= torch.nn.Embedding(vocab.num_embeddings,
                input_dim, scale_grad_by_freq=True, padding_idx=0)
        self.projection_to_labels = torch.nn.Linear(layers * 2 * num_directions * hidden_dim, 3)
        self.dual_lstm = DualLSTM(self.b.batch_size,input_dim,
                hidden_dim,layers=layers,
                bidirectional=False,to_cuda=use_cuda )
        self.init_weights()

        #print(i.size(1), input_dim)
        self.proj1 = torch.nn.Linear(i.size(1)*input_dim, hidden_dim)
        self.proj2 = torch.nn.Linear(s.size(1)*input_dim, hidden_dim)
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
        out1 = out_all1[b1].view(self.b.batch_size,-1)

        b2.fill_(0)
        for i, num in enumerate(sup_len.data):
            b2[i,num-1,:] = 1
        out2 = out_all2[b2].view(self.b.batch_size,-1)

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

def train_model(self):
    epochs = 5
    hook = AccuracyHook('Train')
    self.b.add_hook(hook)
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    for epoch in range(epochs):
        for inp, sup, inp_len, sup_len, t in self.b:
            optimizer.zero_grad()
            pred = self.forward_to_output(inp, sup, inp_len, sup_len, t)
            #print(pred)
            loss = F.nll_loss(pred, t)
            #print(loss)
            loss.backward()
            optimizer.step()
            maxiumum, argmax = torch.topk(pred.data, 1)
            self.b.add_to_hook_histories(t, argmax)

def print_data(datasets, vocab, num=100):
    inp, sup, t = datasets
    for row in range(num):
        hypo = ''
        premise = ''
        for idx in inp[row]:
            if idx == 0: continue
            premise += vocab.get_word(idx) + ' '
        for idx in sup[row]:
            if idx == 0: continue
            hypo += vocab.get_word(idx) + ' '
        print([premise, hypo, vocab.idx2label[t[row]]])

def main():
    # load data
    names, file_paths = snli2spoder()

    # tokenize and convert to hdf5
    lower_list = [True for name in names]
    add_to_vocab_list = [name != 'test' for name in names]
    filetype = spoder2hdf5.SINGLE_INPUT_SINGLE_SUPPORT_CLASSIFICATION
    hdf5paths, vocab = spoder2hdf5.file2hdf(
        file_paths, names, lower_list, add_to_vocab_list, filetype)

    X, S, X_len, S_len, T = hdf5paths[0]
    datasets = [hdf2numpy(X), hdf2numpy(S), hdf2numpy(X_len), hdf2numpy(S_len), hdf2numpy(T)]
    rdm = np.random.RandomState(23435)
    idx = np.arange(datasets[0].shape[0])
    rdm.shuffle(idx)
    datasets2 = []
    k = 2000
    for ds in datasets:
        datasets2.append(ds[idx][:k])

    #print_data(datasets, vocab)
    #print_data(datasets2, vocab)
    #print(type(datasets[0]))
    snli = SNLIClassification(datasets2, vocab, use_cuda=True)
    snli.cuda()
    snli.train()
    train_model(snli)

if __name__ == '__main__':
    main()
