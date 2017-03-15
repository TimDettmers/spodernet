from __future__ import print_function
from torch.autograd import Variable
from itertools import chain

import torch
import numpy as np

from spodernet.interfaces import IAtBatchPreparedObservable
from spodernet.utils.util import Timer
from spodernet.utils.global_config import Config



class TorchConverter(IAtBatchPreparedObservable):
    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        inp = Variable(torch.from_numpy(np.int64(inp)))
        inp_len = Variable(torch.IntTensor(inp_len))
        sup = Variable(torch.from_numpy(np.int64(sup)))
        sup_len = Variable(torch.from_numpy(sup_len))
        t = Variable(torch.from_numpy(np.int64(t)))

        return [inp, inp_len, sup, sup_len, t, idx]

class TorchCUDAConverter(IAtBatchPreparedObservable):
    def __init__(self, device_id):
        self.device_id = device_id

    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        inp = inp.cuda(self.device_id, True)
        inp_len = inp_len.cuda(self.device_id, True)
        sup = sup.cuda(self.device_id, True)
        sup_len = sup_len.cuda(self.device_id)
        t = t.cuda(self.device_id)
        idx = idx

        return [inp, inp_len, sup, sup_len, t, idx]


class TorchDictConverter(IAtBatchPreparedObservable):
    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts

        feed_dict = {}
        feed_dict['input'] = inp
        feed_dict['input_length'] = inp_len
        feed_dict['support'] = sup
        feed_dict['support_length'] = sup_len
        feed_dict['target'] = t
        feed_dict['index'] = idx

        return feed_dict


######################################
#
#          Util functions
#
######################################

def convert_state(state):
    if isinstance(state.targets, Variable):
        state.targets = state.targets.data
    if isinstance(state.argmax, Variable):
        state.argmax = state.argmax.data
    if isinstance(state.pred, Variable):
        state.pred = state.pred.data

    return state



def train_torch(train_batcher, train_batcher_error, dev_batcher, test_batcher, model, epochs=5):
    generators = []
    for module in model.modules:
        if Config.cuda:
            module.cuda()
        generators.append(module.parameters())


    parameters = chain.from_iterable(generators)
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    print('starting training...')
    t0= Timer()
    for epoch in range(epochs):
        for module in model.modules:
            module.train()
        t0.tick()
        for i, feed_dict in enumerate(train_batcher):
            t0.tick()

            optimizer.zero_grad()
            logits, loss, argmax = model.forward(feed_dict)
            loss.backward()
            optimizer.step()
            train_batcher.state.argmax = argmax
            train_batcher.state.targets = feed_dict['target']
            if i == 500: break
            t0.tick()
        t0.tick()
        t0.tock()

        for module in model.modules:
            module.eval()

        for feed_dict in dev_batcher:
            logits, loss, argmax = model.forward(feed_dict)
            dev_batcher.state.argmax = argmax
            dev_batcher.state.targets = feed_dict['target']
