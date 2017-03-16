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

        str2var = {}
        str2var['input'] = inp
        str2var['input_length'] = inp_len
        str2var['support'] = sup
        str2var['support_length'] = sup_len
        str2var['target'] = t
        str2var['index'] = idx

        return str2var


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

def get_list_of_torch_modules(model):
    modules = []
    for module in model.modules:
        if hasattr(module, 'modules'):
            for module2 in module.modules:
                modules.append(module2)
        else:
            modules.append(module)
    return modules



def train_model(model, batcher, epochs=1, iterations=None):
    modules = get_list_of_torch_modules(model)
    generators = []
    for module in modules:
        if Config.cuda:
            module.cuda()
        generators.append(module.parameters())

    parameters = chain.from_iterable(generators)
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    for module in modules:
        module.train()

    for epoch in range(epochs):
        for i, str2var in enumerate(batcher):
            optimizer.zero_grad()
            logits, loss, argmax = model.forward(str2var)
            loss.backward()
            optimizer.step()
            batcher.state.argmax = argmax
            batcher.state.targets = str2var['target']

            if iterations > 0:
                if i == iterations: break


def eval_model(model, batcher, iterations=None):
    modules = get_list_of_torch_modules(model)
    for module in modules:
        module.eval()

    for i, str2var in enumerate(batcher):
        logits, loss, argmax = model.forward(str2var)
        batcher.state.argmax = argmax
        batcher.state.targets = str2var['target']

        if iterations > 0:
            if i == iterations: break
