import numpy as np
from torch.autograd import Variable
import torch

class AccuracyHook(object):
    def __init__(self, name=''):
        self.epoch2errors = {}
        self.current_scores = []
        self.name = name
        pass

    def add_to_history(self, targets, argmax, loss, data=None):
        n = targets.size()[0]

        if isinstance(targets, Variable):
            targets = targets.data
        if isinstance(argmax, Variable):
            argmax = argmax.data

        self.current_scores.append((torch.sum(targets==argmax))/np.float32(n))

    def print_statistic(self, epoch):
        m = np.mean(self.current_scores)
        se = np.std(self.current_scores)/np.sqrt(len(self.current_scores))
        lower = m-(1.96*se)
        upper = m+(1.96*se)
        print('{3} Accuracy: {2:.3}\t95% CI: ({0:.3}, {1:.3})'.format(lower, upper, m,
            self.name))
        if epoch not in self.epoch2errors:
            self.epoch2errors[epoch] = []
        self.epoch2errors[epoch].append([m, lower, upper])
        del self.current_scores[:]

class LossHook(object):
    def __init__(self, name=''):
        self.epoch2errors = {}
        self.current_scores = []
        self.name = name
        pass

    def add_to_history(self, targets, argmax, loss, data=None):
        if isinstance(targets, Variable):
            targets = targets.data
        if isinstance(loss, Variable):
            loss = loss.data

        self.current_scores.append(loss[0])

    def print_statistic(self, epoch):
        m = np.mean(self.current_scores)
        se = np.std(self.current_scores)/np.sqrt(len(self.current_scores))
        lower = m-(1.96*se)
        upper = m+(1.96*se)
        print('{3} Loss: {2:.5}\t95% CI: ({0:.5}, {1:.5})'.format(lower, upper, m,
            self.name))
        if epoch not in self.epoch2errors:
            self.epoch2errors[epoch] = []
        self.epoch2errors[epoch].append([m, lower, upper])
        del self.current_scores[:]
