import numpy as np
import torch
from torch.autograd import Variable
import time
import datetime


class Batcher(object):
    '''Takes data and creates batches over which one can iterate.'''

    def __init__(self, datasets, batch_size=128, transfer_to_gpu=True,
            num_print_thresholds=1000, sort_idx_seq_pairs=None):
        self.datasets = []
        self.batch_size = batch_size
        self.n = datasets[0].shape[0]
        self.idx = 0

        #shuffle_idx = np.arange(self.n)
        #np.random.shuffle(shuffle_idx)
        #for ds in datasets:
        #    ds = ds[shuffle_idx]

        self.print_thresholds = np.int32((int(self.n/batch_size)) *
                np.arange(num_print_thresholds)/float(num_print_thresholds))
        self.threshold_time = np.zeros(num_print_thresholds)

        self.init_datasets(datasets, transfer_to_gpu)
        self.t0 = time.time()
        self.num_thresholds = num_print_thresholds
        self.epoch = 1
        self.hooks = []
        self.sort_idx_seq_pairs = sort_idx_seq_pairs

    def add_hook(self, hook):
        self.hooks.append(hook)

    def init_datasets(self, datasets, transfer_to_gpu):
        for ds in datasets:
            dtype = ds.dtype
            if dtype == np.int32 or dtype == np.int64:
                ds_torch = torch.LongTensor(np.int64(ds))
            elif dtype == np.float32 or dtype == np.float64:
                ds_torch = torch.FloatTensor(np.float32(ds))

            if transfer_to_gpu:
                ds_torch = ds_torch.cuda()

            self.datasets.append(ds_torch)


        # we ignore off-size batches
        self.batches = int(self.n / self.batch_size)

    def shuffle(self, seed=2343):
        rdm = np.random.RandomState(seed)
        idx = np.arange(self.n)
        rdm.shuffle(idx)
        for ds in self.datasets:
            ds = ds[idx]

    def __iter__(self):
        return self

    def print_ETA(self, time_idx):
        self.threshold_time[time_idx] -=\
                np.sum(self.threshold_time[:time_idx-1])
        m = np.mean(self.threshold_time[:time_idx])
        se = np.std(
                self.threshold_time[:time_idx])/np.sqrt(np.float64(time_idx+1))
        togo = self.num_thresholds-time_idx
        lower = m-(1.96*se)
        upper = m+(1.96*se)
        lower_time = datetime.timedelta(seconds=np.round(lower*togo))
        upper_time = datetime.timedelta(seconds=np.round(upper*togo))
        mean_time = datetime.timedelta(seconds=np.round(m*togo))
        print('Epoch ETA: {2}\t95% CI: ({0}, {1})'.format(
            lower_time, upper_time, mean_time))

    def add_to_hook_histories(self, targets, argmax, loss):
        for hook in self.hooks:
            hook.add_to_history(targets, argmax, loss)


    def next(self):
        if self.idx + 1 < self.batches:
            if np.sum(self.idx == self.print_thresholds) > 0:
                time_idx = np.where(self.idx == self.print_thresholds)[0][0]
                self.threshold_time[time_idx] = time.time()-self.t0
                if time_idx > 1:
                    self.print_ETA(time_idx)
                    for hook in self.hooks:
                        hook.print_statistic(self.epoch)

            batches = []
            for ds in self.datasets:
                start = self.idx * self.batch_size
                end = (self.idx + 1) * self.batch_size
                batches.append(ds[start:end])
            self.idx += 1

            if self.sort_idx_seq_pairs != None:
                for (idx_sort, idx_seq) in self.sort_idx_seq_pairs:
                    idx_sorted, idx_argsorted = batches[idx_sort].sort(0, descending=True)
                    batches[idx_sort] = idx_sorted
                    batches[idx_seq]= batches[idx_seq][idx_sorted]


            for i in range(len(batches)):
                batches[i] = Variable(batches[i])

            return batches
        else:
            self.idx = 0
            self.epoch += 1
            self.threshold_time *= 0
            self.t0 = time.time()
            print('EPOCH: {0}'.format(self.epoch))
            raise StopIteration()


