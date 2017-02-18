import numpy as np
import torch
from torch.autograd import Variable


class Batcher(object):
    '''Takes data and creates batches over which one can iterate.'''

    def __init__(self, datasets, batch_size=128, transfer_to_gpu=True):
        self.datasets = []
        self.batch_size = batch_size
        self.n = datasets[0].shape[0]
        self.idx = 0

        self.init_datasets(datasets, transfer_to_gpu)

    def init_datasets(self, datasets, transfer_to_gpu):
        for ds in datasets:
            dtype = ds.dtype
            if dtype == np.int32 or dtype == np.int64:
                ds_torch = torch.LongTensor(np.int64(ds))

                if transfer_to_gpu:
                    ds_torch = ds_torch.cuda()

            self.datasets.append(Variable(ds_torch))


        # we ignore off-size batches
        self.batches = int(self.n / self.batch_size)

    def shuffle(self, seed=2343):
        rdm = np.random.RandomState(seed)
        idx = np.arange(self.n)
        rdm.shuffle(idx)
        for ds in dataset:
            ds = ds[idx]

    def __iter__(self):
        return self

    def next(self):
        if self.idx + 1 < self.batches:
            batches = []
            for ds in self.datasets:
                start = self.idx * self.batch_size
                end = (self.idx + 1) * self.batch_size
                batches.append(ds[start:end])
                self.idx += 1
            return batches
        else:
            self.idx = 0
            raise StopIteration()


