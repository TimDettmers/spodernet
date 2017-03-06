import numpy as np
from torch.autograd import Variable
import torch

from spodernet.logger import Logger

log = Logger('hooks.py.txt')

class AbstractHook(object):
    def __init__(self, name=''):
        self.epoch2errors = {}
        self.current_scores = []
        self.name = name
        pass

    def calculate_metric(self, targets, argmax, loss):
        raise NotImplementedError('Classes that inherit from abstract hook need to implement the calcualte metric method.')

    def add_to_history(self, targets, argmax, loss):
        if isinstance(targets, Variable):
            targets = targets.data
        if isinstance(argmax, Variable):
            argmax = argmax.data

        log.statistical('Argmax value or None: {0}', argmax)
        log.statistical('Loss value: {0}', loss)
        metric = self.calculate_metric(targets, argmax, loss)
        log.statistical('A metric value like F1, classification error etc; a single number: {0}', metric)

        self.current_score.append(metric)

    def print_statistic(self, epoch):
        m = np.mean(self.current_scores)
        se = np.std(self.current_scores)/np.sqrt(len(self.current_scores))
        lower = m-(1.96*se)
        upper = m+(1.96*se)
        log.info('{3} Accuracy: {2:.3}\t95% CI: ({0:.3}, {1:.3})'.format(lower, upper, m, self.name))
        if epoch not in self.epoch2errors:
            self.epoch2errors[epoch] = []
        self.epoch2errors[epoch].append([m, lower, upper])
        del self.current_scores[:]

class AccuracyHook(AbstractHook):
    def __init__(self, name=''):
        super(AccuracyHook, self).__init__(name)

    def calculate_metric(self, targets, argmax, loss):
        n = targets.size()[0]
        return torch.sum(targets==argmax)/np.float32(n)

class LossHook(AbstractHook):
    def __init__(self, name=''):
        super(AccuracyHook, self).__init__(name)

    def calculate_metric(self, targets, argmax, loss):
        n = targets.size()[0]
        return loss[0]
