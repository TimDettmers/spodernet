from torch.autograd import Variable

import torch
import numpy as np
import scipy.stats
import datetime

from spodernet.preprocessing.batching import IAtIterEndObservable, IAtEpochEndObservable

from spodernet.logger import Logger
log = Logger('hooks.py.txt')

class AbstractHook(IAtIterEndObservable, IAtEpochEndObservable):
    def __init__(self, name, metric_name, print_every_x_batches):
        self.epoch_errors = []
        self.current_scores = []
        self.name = name
        self.iter_count = 0
        self.print_every = print_every_x_batches
        self.metric_name = metric_name

    def calculate_metric(self, state):
        raise NotImplementedError('Classes that inherit from abstract hook need to implement the calcualte metric method.')

    def convert_state(self, state):
        if isinstance(state.targets, Variable):
            state.targets = state.targets.data
        if isinstance(state.argmax, Variable):
            state.argmax = state.argmax.data
        if isinstance(state.pred, Variable):
            state.pred = state.pred.data

        return state

    def at_end_of_iter_event(self, state):
        state = self.convert_state(state)
        metric = self.calculate_metric(state)
        self.current_scores.append(metric)
        self.iter_count += 1
        if self.iter_count % self.print_every == 0:
            self.print_statistic()

    def at_end_of_epoch_event(self, state):
        self.epoch_errors.append(self.get_confidence_intervals())
        self.print_statistic()
        del self.current_scores[:]

    def get_confidence_intervals(self, percentile=0.95):
        z = scipy.stats.norm.ppf(percentile)
        m = np.mean(self.current_scores)
        se = np.std(self.current_scores)/np.sqrt(len(self.current_scores))
        lower = m-(z*se)
        upper = m+(z*se)
        return [lower, m, upper]

    def print_statistic(self):
        lower, m, upper = self.get_confidence_intervals()
        log.info('{3} {4}: {2:.3}\t95% CI: ({0:.3}, {1:.3})'.format(lower, upper, m, self.name, self.metric_name))

class AccuracyHook(AbstractHook):
    def __init__(self, name='', print_every_x_batches=1000):
        super(AccuracyHook, self).__init__(name, 'Accuracy', print_every_x_batches)

    def calculate_metric(self, state):
        n = state.targets.size()[0]
        return torch.sum(state.targets==state.argmax)/np.float32(n)

class LossHook(AbstractHook):
    def __init__(self, name='', print_every_x_batches=1000):
        super(AccuracyHook, self).__init__(name, 'Loss', print_every_x_batches)

    def calculate_metric(self, state):
        return state.loss[0]

class ETAHook(AbstractHook):
    def __init__(self, name='', print_every_x_batches=1000):
        super(ETAHook, self).__init__(name, 'ETA', print_every_x_batches)

    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def calculate_metric(self, state):
        n = state.num_batches
        i = state.current_idx
        cumulative_t = state.timer.tick('ETA')
        ETA_estimate = (cumulative_t/i)*n
        state.timer.tick('ETA')

        return ETA_estimate

    def print_statistic(self):
        lower, m, upper = self.get_confidence_intervals()
        lower, m, upper = self.get_time_string(lower), self.get_time_string(m), self.get_time_string(upper)
        log.info('{3} {4}: {2}\t95% CI: ({0}, {1})'.format(lower, upper, m, self.name, self.metric_name))

