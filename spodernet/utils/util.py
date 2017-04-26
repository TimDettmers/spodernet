from os.path import join

import h5py
import os
import time
import torch

from spodernet.utils.logger import Logger
from torch.cuda import Event
log = Logger('util.py.txt')

def write_to_hdf(path, data):
    '''Writes a numpy array to a hdf5 file under the given path.'''
    log.debug_once('Saving hdf5 file to: {0}', path)
    h5file = h5py.File(path, "w")
    h5file.create_dataset("default", data=data)
    h5file.close()


def load_hdf_file(path, keyword='default'):
    '''Reads and returns a numpy array for a hdf5 file'''
    log.debug_once('Reading hdf5 file from: {0}', path)
    h5file = h5py.File(path, 'r')
    dset = h5file.get(keyword)
    data = dset[:]
    h5file.close()
    return data

def load_hdf5_paths(paths, limit=None):
    data = []
    for path in paths:
        if limit != None:
            data.append(load_hdf_file(path)[:limit])
        else:
            data.append(load_hdf_file(path))
    return data

def get_home_path():
    return os.environ['HOME']

def get_data_path():
    return join(os.environ['HOME'], '.data')

def make_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Timer(object):
    def __init__(self, silent=False):
        self.cumulative_secs = {}
        self.current_ticks = {}
        self.silent = silent

    def tick(self, name='default'):
        if name not in self.current_ticks:
            self.current_ticks[name] = time.time()

            return 0.0
        else:
            if name not in self.cumulative_secs:
                self.cumulative_secs[name] = 0
            t = time.time()
            self.cumulative_secs[name] += t - self.current_ticks[name]
            self.current_ticks.pop(name)

            return self.cumulative_secs[name]

    def tock(self, name='default'):
        self.tick(name)
        value = self.cumulative_secs[name]
        if not self.silent:
            log.info('Time taken for {0}: {1:.8f}s'.format(name, value))
        self.cumulative_secs.pop(name)
        self.current_ticks.pop(name, None)

        return value



class CUDATimer(object):
    def __init__(self, silent=False):
        self.cumulative_secs = {}
        self.current_ticks = {}
        self.silent = silent
        self.end = Event(enable_timing=True, blocking=True)

    def tick(self, name='default'):
        if name not in self.current_ticks:
            start = Event(enable_timing=True, blocking=True)
            start.record()
            self.current_ticks[name] = start

            return 0.0
        else:
            if name not in self.cumulative_secs:
                self.cumulative_secs[name] = 0
            self.end.record()
            self.end.synchronize()
            self.cumulative_secs[name] += self.current_ticks[name].elapsed_time(self.end)/1000.
            self.current_ticks.pop(name)

            return self.cumulative_secs[name]

    def tock(self, name='default'):
        self.tick(name)
        value = self.cumulative_secs[name]
        if not self.silent:
            log.info('Time taken for {0}: {1:.8f}s'.format(name, value))
        self.cumulative_secs.pop(name)
        if name in self.current_ticks:
            del self.current_ticks[name]
        self.current_ticks.pop(name, None)

        return value
