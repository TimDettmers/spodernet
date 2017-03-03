'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function

from spodernet.data.snli2spoder import snli2spoder
from spodernet.preprocessing import spoder2hdf5
from spodernet.util import hdf2numpy, load_hdf5_paths
from spodernet.preprocessing.batching import Batcher
from spodernet.hooks import AccuracyHook, LossHook
from spodernet.models import DualLSTM

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats.mstats import normaltest
#import torch.nn.utils.rnn as rnn_utils
np.set_printoptions(suppress=True)
import time

class SNLIClassification(torch.nn.Module):
    def __init__(self, batch_size, vocab, use_cuda=False):
        super(SNLIClassification, self).__init__()
        self.batch_size = batch_size 
        input_dim = 256
        hidden_dim = 512
        num_directions = 2
        layers = 2
        self.emb= torch.nn.Embedding(vocab.num_embeddings,
                input_dim, padding_idx=0)#, scale_grad_by_freq=True, padding_idx=0)
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

def train_model(model, train_batcher, dev_batcher):
    epochs = 5
    hook = AccuracyHook('Train')
    train_batcher.add_hook(hook)
    train_batcher.add_hook(LossHook('Train'))
    dev_batcher.add_hook(AccuracyHook('Dev'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for inp, sup, inp_len, sup_len, t in train_batcher:
            optimizer.zero_grad()
            pred = model.forward_to_output(inp, sup, inp_len, sup_len, t)
            #print(pred)
            loss = F.nll_loss(pred, t)
            #print(loss)
            loss.backward()
            optimizer.step()
            maxiumum, argmax = torch.topk(pred.data, 1)
            train_batcher.add_to_hook_histories(t, argmax, loss)

        model.eval()
        for inp, sup, inp_len, sup_len, t in dev_batcher:
            pred = model.forward_to_output(inp, sup, inp_len, sup_len, t)
            maxiumum, argmax = torch.topk(pred.data, 1)
            dev_batcher.add_to_hook_histories(t, argmax, loss)
    print(time.time() - t0)


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

    train_set = load_hdf5_paths(hdf5paths[0])
    dev_set = load_hdf5_paths(hdf5paths[1])

    batch_size = 128
    train_batcher = Batcher(train_set, batch_size=batch_size, len_indices=(2,3), data_indices=(0,1), num_print_thresholds=5, transfer_to_gpu=True)
    dev_batcher = Batcher(dev_set, batch_size=batch_size, num_print_thresholds=1, transfer_to_gpu=True)

    snli = SNLIClassification(batch_size, vocab, use_cuda=True)
    snli.cuda()
    snli.train()
    train_model(snli, train_batcher, dev_batcher)

if __name__ == '__main__':
    main()
