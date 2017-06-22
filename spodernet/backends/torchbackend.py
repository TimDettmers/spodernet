from __future__ import print_function
from torch.autograd import Variable
from itertools import chain

import torch
import numpy as np

from spodernet.interfaces import IAtBatchPreparedObservable
from spodernet.utils.util import Timer
from spodernet.utils.global_config import Config

class TorchConverter(IAtBatchPreparedObservable):
    def at_batch_prepared(self, str2var):
        for key in str2var.keys():
            str2var[key] = Variable(torch.from_numpy(np.int64(str2var[key])))
        return str2var

class TorchCUDAConverter(IAtBatchPreparedObservable):
    def __init__(self, device_id):
        self.device_id = device_id

    def at_batch_prepared(self, str2var):
        for key in str2var.keys():
            str2var[key] = str2var[key].cuda(self.device_id, True)
        return str2var


class TorchNegativeSampling(IAtBatchPreparedObservable):
    def __init__(self, max_index, keys_to_corrupt=['input', 'target']):
        self.max_index = max_index
        self.keys_to_corrupt = keys_to_corrupt
        self.rdm = np.random.RandomState(34534)

    def at_batch_prepared(self, str2var):
        samples_per_key = Config.batch_size/len(self.keys_to_corrupt)
        for i, key in enumerate(self.keys_to_corrupt):
            variable = str2var[key]
            new_idx = self.rdm.choice(self.max_index, samples_per_key)
            if Config.cuda:
                variable_corrupted = Variable(torch.cuda.LongTensor(variable.size()))
                variable_corrupted.data.copy_(variable.data)
                variable_corrupted.data[i*samples_per_key: (i+1)*samples_per_key] = torch.from_numpy(new_idx).cuda()
            else:
                variable_corrupted = Variable(torch.LongTensor(variable.size()))
                variable_corrupted.data.copy_(variable.data)
                variable_corrupted.data[i*samples_per_key: (i+1)*samples_per_key] = torch.from_numpy(new_idx)
            str2var[key + '_corrupt'] = variable_corrupted

        return str2var


class TorchTargetIdx2MultiTarget(IAtBatchPreparedObservable):
    def __init__(self, device_id, num_labels, variable_name, new_variable_name):
        self.num_labels = num_labels
        self.variable_name = variable_name
        self.new_variable_name = new_variable_name
        self.device_id = device_id


    def at_batch_prepared(self, str2var):
        t = str2var[self.variable_name]
        new_t = torch.zeros(Config.batch_size, self.num_labels)#np.zeros((Config.batch_size, self.num_labels), dtype=np.int64)
        for i, row in enumerate(t.data):
            for idx in t.data[i]:
                new_t[i, idx] = 1
            #    print(new_t[i, idx])
            ##if len(t.shape) == 1:
            #if len(t.size()) == 1:
            #    new_t[i, row] = 1
            #else:
            #    for col in row:
            #        new_t[i, col] = 1

        str2var[self.new_variable_name] = Variable(new_t)
        #if Config.cuda:
        #    str2var[self.new_variable_name] = str2var[self.new_variable_name].cuda(self.device_id, True)

        return str2var

######################################
#
#          Util functions
#
######################################


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
