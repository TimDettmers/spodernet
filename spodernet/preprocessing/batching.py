from os.path import join
from threading import Thread, Event
from collections import namedtuple
from torch.autograd import Variable

import torch
import time
import datetime
import numpy as np
import cPickle as pickle
import Queue

from spodernet.util import get_data_path, load_hdf_file, Timer
from spodernet.logger import Logger
from spodernet.global_config import Config, Backends, TensorFlowConfig
from spodernet.hooks import ETAHook
from spodernet.observables import IAtIterEndObservable, IAtEpochEndObservable, IAtEpochStartObservable, IAtBatchPreparedObservable

log = Logger('batching.py.txt')



class TorchConverter(IAtBatchPreparedObservable):
    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        inp = Variable(torch.LongTensor(np.int64(inp)))
        inp_len = Variable(torch.IntTensor(np.int64(inp_len)))
        sup = Variable(torch.LongTensor(np.int64(sup)))
        sup_len = Variable(torch.IntTensor(np.int64(sup_len)))
        t = Variable(torch.LongTensor(np.int64(t)))
        return [inp, inp_len, sup, sup_len, t, idx]

class TorchCUDAConverter(IAtBatchPreparedObservable):
    def __init__(self, device_id):
        self.device_id = device_id

    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        inp = inp.cuda(self.device_id)
        inp_len = inp_len.cuda(self.device_id)
        sup = sup.cuda(self.device_id)
        sup_len = sup_len.cuda(self.device_id)
        t = t.cuda(self.device_id)
        idx = idx
        return [inp, inp_len, sup, sup_len, t, idx]

class TensorFlowConverter(IAtBatchPreparedObservable):

    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        if TensorFlowConfig.inp == None:
            log.error('You need to initialize the batch size via TensorflowConfig.init_batch_size(batchsize)!')
        feed_dict = {}
        feed_dict[TensorFlowConfig.inp] = inp
        feed_dict[TensorFlowConfig.support] = sup
        feed_dict[TensorFlowConfig.input_length] = inp_len
        feed_dict[TensorFlowConfig.support_length] = sup_len
        feed_dict[TensorFlowConfig.target] = t
        feed_dict[TensorFlowConfig.index] = idx
        return feed_dict

class BatcherState(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.loss = None
        self.argmax = None
        self.pred = None
        self.batch_size = None
        self.current_idx = None
        self.current_epoch = None
        self.targets = None
        self.num_batches = None
        self.timer = None


class DataLoaderSlave(Thread):
    def __init__(self, stream_batcher, batchidx2paths, batchidx2start_end, randomize=False, paths=None, shard2batchidx=None, seed=None, shard_fractions=None):
        super(DataLoaderSlave, self).__init__()
        if randomize:
            assert seed is not None, 'For randomized data loadingg a seed needs to be set!'
        self.stream_batcher = stream_batcher
        self.batchidx2paths = batchidx2paths
        self.batchidx2start_end = batchidx2start_end
        self.current_data = {}
        self.randomize = randomize
        self.num_batches = len(batchidx2paths.keys())
        self.rdm = np.random.RandomState(234)
        self.shard_fractions = shard_fractions
        self.shard2batchidx = shard2batchidx
        self.paths = paths
        self.stopping = False
        self._stop = Event()
        self.daemon = True

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def load_files_if_needed(self, current_paths):
        if isinstance(current_paths[0], list):
            for paths in current_paths:
                for path in paths:
                    if path not in self.current_data:
                        self.current_data[path] = load_hdf_file(path)
        else:
            for path in current_paths:
                if path not in self.current_data:
                    self.current_data[path] = load_hdf_file(path)

    def create_batch_parts(self, current_paths, start, end):
        # index loaded data for minibatch
        batch_parts = []
        if isinstance(current_paths[0], list):
            start = start[0]
            end = end[1]
            for i in range(len(current_paths[0])):
                x1 = self.current_data[current_paths[0][i]][start:]
                x2 = self.current_data[current_paths[1][i]][:end]
                if len(x1.shape) == 1:
                    x = np.hstack([x1, x2])
                else:
                    x = np.vstack([x1, x2])
                batch_parts.append(x)
        else:
            for path in current_paths:
                batch_parts.append(self.current_data[path][start:end])

        return batch_parts

    def clean_cache(self, current_paths):
        # delete unused cached data
        if isinstance(current_paths[0], list):
            current_paths = current_paths[0] + current_paths[1]


        for old_path in self.current_data.keys():
            if old_path not in current_paths:
                self.current_data.pop(old_path, None)

    def publish_at_prepared_batch_event(self, batch_parts):
        for obs in self.stream_batcher.at_batch_prepared_observers:
            batch_parts = obs.at_batch_prepared(batch_parts)
        return batch_parts

    def run(self):
        while not self.stopped():

            # we have this to terminate threads gracefully
            # if we use daemons then the terminational signal might not be heard while loading files
            # thus causing ugly exceptions
            try:
                batch_idx = self.stream_batcher.work.get(block=False, timeout=1.0)
            except:
                continue

            if self.randomize:
                shard_idx = self.rdm.choice(len(self.shard2batchidx.keys()), 1, p=self.shard_fractions)[0]
                current_paths = self.paths[shard_idx]

                self.load_files_if_needed(current_paths)

                n = self.current_data[current_paths[0]].shape[0]
                start = self.rdm.randint(0, n-self.stream_batcher.batch_size+1)
                end = start + self.stream_batcher.batch_size

                batch_parts = self.create_batch_parts(current_paths, start, end)
            else:
                if batch_idx not in self.batchidx2paths:
                    log.error('{0}, {1}', batch_idx, self.batchidx2paths.keys())
                current_paths = self.batchidx2paths[batch_idx]
                start, end = self.batchidx2start_end[batch_idx]

                self.load_files_if_needed(current_paths)
                batch_parts = self.create_batch_parts(current_paths, start, end)


            batch_parts = self.publish_at_prepared_batch_event(batch_parts)
            # pass data to streambatcher
            self.stream_batcher.prepared_batches[batch_idx] = batch_parts
            self.stream_batcher.prepared_batchidx.put(batch_idx)

            self.clean_cache(current_paths)


class StreamBatcher(object):
    def __init__(self, pipeline_name, name, batch_size, loader_threads=8, randomize=False):
        config_path = join(get_data_path(), pipeline_name, name, 'hdf5_config.pkl')
        config = pickle.load(open(config_path))
        self.paths = config['paths']
        self.fractions = config['fractions']
        self.num_batches = int(np.sum(config['counts']) / batch_size)
        self.batch_size = batch_size
        self.batch_idx = 0
        self.prefetch_batch_idx = 0
        self.loaders = []
        self.prepared_batches = {}
        self.prepared_batchidx = Queue.Queue()
        self.work = Queue.Queue()
        self.cached_batches = {}
        eta = ETAHook(name, print_every_x_batches=1000)
        self.end_iter_observers = [eta]
        self.end_epoch_observers = [eta]
        self.start_epoch_observers = [eta]
        self.at_batch_prepared_observers = []
        self.state = BatcherState()
        self.current_iter = 0
        self.current_epoch = 0
        self.timer = Timer()
        if Config.backend == Backends.TORCH:
            self.subscribe_to_batch_prepared_event(TorchConverter())
            if Config.cuda:
                self.subscribe_to_batch_prepared_event(TorchCUDAConverter(torch.cuda.current_device()))
        elif Config.backend == Backends.TENSORFLOW:
            self.subscribe_to_batch_prepared_event(TensorFlowConverter())
        else:
            raise Exception('Backend has unsupported value {0}'.format(Config.backend))


        batchidx2paths, batchidx2start_end, shard2batchidx = self.create_batchidx_maps(config['counts'])

        for i in range(loader_threads):
            seed = None
            if randomize:
                seed = 23432435345 % ((i+1)*17)
            self.loaders.append(DataLoaderSlave(self, batchidx2paths, batchidx2start_end, randomize, self.paths, shard2batchidx, seed, self.fractions))
            self.loaders[-1].start()

        while self.prefetch_batch_idx < loader_threads:
            self.work.put(self.prefetch_batch_idx)
            self.prefetch_batch_idx += 1

    def __del__(self):
        log.debug('Stopping threads...')
        for worker in self.loaders:
            worker.stop()

        log.debug('Waiting for threads to finish...')
        for worker in self.loaders:
            worker.join()

    def subscribe_end_of_iter_event(self, observer):
        self.end_iter_observers.append(observer)

    def subscribe_end_of_epoch_event(self, observer):
        self.end_epoch_observers.append(observer)

    def subscribe_to_events(self, observer):
        self.subscribe_end_of_iter_event(observer)
        self.subscribe_end_of_epoch_event(observer)

    def subscribe_to_batch_prepared_event(self, observer):
        self.at_batch_prepared_observers.append(observer)

    def subscribe_to_start_of_epoch_event(self, observer):
        self.start_epoch_observers.append(observer)

    def publish_end_of_iter_event(self):
        self.state.current_idx = self.batch_idx
        self.state.current_epoch = self.current_epoch
        self.state.num_batches = self.num_batches

        if self.batch_idx == 0:
            self.current_iter += 1
            for obs in self.start_epoch_observers:
                obs.at_start_of_epoch_event(self.state)
            return
        for obs in self.end_iter_observers:
            obs.at_end_of_iter_event(self.state)
        self.state.clear()
        self.current_iter += 1

    def publish_end_of_epoch_event(self):
        self.state.current_idx = self.batch_idx
        self.state.current_epoch = self.current_epoch
        self.state.num_batches = self.num_batches
        self.state.timer = self.timer
        for obs in self.end_epoch_observers:
            obs.at_end_of_epoch_event(self.state)
        self.state.clear()
        self.current_epoch += 1

    def create_batchidx_maps(self, counts):
        counts_cumulative = np.cumsum(counts)
        counts_cumulative_offset = np.cumsum([0] + counts)
        batchidx2paths = {}
        batchidx2start_end = {}
        shard2batchidx = { 0 : []}
        paths = self.paths
        file_idx = 0
        for i in range(self.num_batches):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            if end > counts_cumulative[file_idx] and file_idx+1 < len(paths):
                start_big_batch = start - counts_cumulative_offset[file_idx]
                end_big_batch = end - counts_cumulative_offset[file_idx+1]
                batchidx2start_end[i] = ((start_big_batch, None), (None, end_big_batch))
                batchidx2paths[i] = (paths[file_idx], paths[file_idx+1])

                shard2batchidx[file_idx].append(i)
                file_idx += 1
                shard2batchidx[file_idx] = [i]
            else:
                start_big_batch = start - counts_cumulative_offset[file_idx]
                end_big_batch = end - counts_cumulative_offset[file_idx]
                batchidx2start_end[i] = (start_big_batch, end_big_batch)
                batchidx2paths[i] = paths[file_idx]
                shard2batchidx[file_idx].append(i)

        return batchidx2paths, batchidx2start_end, shard2batchidx


    def get_next_batch_parts(self):
        if self.batch_idx in self.cached_batches:
            return self.cached_batches.pop(self.batch_idx)
        else:
            batch_idx = self.prepared_batchidx.get()
            if self.batch_idx == batch_idx:
                return self.prepared_batches.pop(self.batch_idx)
            else:
                self.cached_batches[batch_idx] = self.prepared_batches.pop(batch_idx)
                return self.get_next_batch_parts()

    def __iter__(self):
        return self

    def next(self):
        if self.batch_idx + 1 < self.num_batches:
            batch_parts = self.get_next_batch_parts()
            self.publish_end_of_iter_event()

            self.batch_idx += 1
            self.work.put(self.prefetch_batch_idx)
            self.prefetch_batch_idx +=1
            if self.prefetch_batch_idx == self.num_batches:
                self.prefetch_batch_idx = 0

            return batch_parts
        else:
            self.batch_idx = 0
            self.publish_end_of_epoch_event()
            raise StopIteration()
