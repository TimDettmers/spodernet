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
        for i in range(len(batch_parts)):
            batch_parts[i] = Variable(torch.from_numpy(np.int64(batch_parts[i])))
        return batch_parts

class TorchCUDAConverter(IAtBatchPreparedObservable):
    def __init__(self, device_id):
        self.device_id = device_id

    def at_batch_prepared(self, batch_parts):
        for i in range(len(batch_parts)):
            batch_parts[i] = batch_parts[i].cuda(self.device_id, True)
        return batch_parts


class TorchDictConverter(IAtBatchPreparedObservable):
    def at_batch_prepared(self, batch_parts):
        str2var = {}
        str2var['input'] = batch_parts[0]
        str2var['input_length'] = batch_parts[1]
        str2var['support'] = batch_parts[2]
        str2var['support_length'] = batch_parts[3]
        str2var['target'] = batch_parts[4]
        str2var['index'] = batch_parts[5]
        if len(batch_parts) > 6:
            for i in range(6,len(batch_parts)):
                str2var['var{0}'.format(i-6)] = batch_parts[i]

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
    if isinstance(state.loss, Variable):
        state.loss = state.loss.data
    if isinstance(state.multi_labels, Variable):
        state.multi_labels = state.multi_labels.data

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
