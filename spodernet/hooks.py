from torch.autograd import Variable

import torch
import numpy as np
import scipy.stats
import datetime

from spodernet.observables import IAtIterEndObservable, IAtEpochEndObservable, IAtEpochStartObservable
from spodernet.util import Timer

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
        self.epoch = 1

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.n = 0
        self.mean = 0
        self.M2 = 0

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

        self.n += 1
        delta = metric - self.mean
        self.mean += delta/self.n
        delta2 = metric - self.mean
        self.M2 += delta*delta2

        self.current_scores.append(metric)
        self.iter_count += 1
        if self.iter_count % self.print_every == 0:
            self.print_statistic()
            self.n = 0
            self.mean = 0
            self.M2 = 0

    def at_end_of_epoch_event(self, state):
        self.epoch_errors.append(self.get_confidence_intervals())
        self.print_statistic(True)
        del self.current_scores[:]
        self.n = 0
        self.mean = 0
        self.M2 = 0
        self.epoch += 1

    def get_confidence_intervals(self, percentile=0.99, limit=1000):
        z = scipy.stats.norm.ppf(percentile)
        var = self.M2/ (self.n-1)
        SE = np.sqrt(var/self.n)
        lower = self.mean-(z*SE)
        upper = self.mean+(z*SE)
        return [self.n, lower, self.mean, upper]

    def print_statistic(self, at_epoch_end=False):
        n, lower, m, upper = self.get_confidence_intervals()
        str_message = '{3} {4}: {2:.3}\t99% CI: ({0:.3}, {1:.3}), n={5}'.format(lower, upper, m, self.name, self.metric_name, self.n)
        if at_epoch_end: log.info('\n')
        if at_epoch_end: log.info('#'*40)
        if at_epoch_end: log.info(' '*10 + 'COMPLETED EPOCH: {0}'.format(self.epoch) + ' '*30)
        log.info(str_message)
        if at_epoch_end: log.info('#'*40)
        if at_epoch_end: log.info('\n')

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

class ETAHook(AbstractHook, IAtEpochStartObservable):
    def __init__(self, name='', print_every_x_batches=1000):
        super(ETAHook, self).__init__(name, 'ETA', print_every_x_batches)
        self.t = Timer(silent=True)
        self.cumulative_t = 0.0
        self.skipped_first = False

    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h < 0: h = 0
        if m < 0: m = 0
        if s < 0: s = 0
        return "%d:%02d:%02d" % (h, m, s)

    def calculate_metric(self, state):
        n = state.num_batches
        i = state.current_idx
        cumulative_t = self.t.tick('ETA')
        total_time_estimate = (cumulative_t/i)*n
        self.t.tick('ETA')
        self.cumulative_t = cumulative_t

        return total_time_estimate

    def print_statistic(self):
        if not self.skipped_first:
            # the first estimation is very unreliable for time measures
            self.skipped_first = True
            return
        n, lower, m, upper = self.get_confidence_intervals()
        lower -= self.cumulative_t
        m -= self.cumulative_t
        upper -= self.cumulative_t
        lower, m, upper = self.get_time_string(lower), self.get_time_string(m), self.get_time_string(upper)
        log.info('{3} {4}: {2}\t99% CI: ({0}, {1}), n={5}'.format(lower, upper, m, self.name, self.metric_name, n))

    def at_start_of_epoch_event(self, batcher_state):
        self.t.tick('ETA')
        self.t.tick('Epoch')

    def at_end_of_epoch_event(self, state):
        self.t.tock('ETA')
        epoch_time = self.t.tock('Epoch')
        self.epoch_errors.append([epoch_time])
        log.info('Total epoch time: {0}'.format(self.get_time_string(epoch_time)))
        del self.current_scores[:]
        self.n = 0
        self.mean = 0
        self.M2 = 0
        self.skipped_first = False
        self.epoch += 1
