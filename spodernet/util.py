from os.path import join

import h5py
import os
import time


def numpy2hdf(path, data):
    '''Write a numpy array to a hdf5 file under the given path.'''
    h5file = h5py.File(path, "w")
    h5file.create_dataset("default", data=data)


def hdf2numpy(path, keyword='default'):
    '''Reads and returns a numpy array for a hdf5 file'''
    h5file = h5py.File(path, 'r')
    dset = h5file.get(keyword)
    return dset[:]


def load_hdf5_paths(paths, limit=None):
    data = []
    for path in paths:
        if limit != None:
            data.append(hdf2numpy(path)[:limit])
        else:
            data.append(hdf2numpy(path))
    return data

def get_home_path():
    return os.environ['HOME']

def get_logger_path():
    return join(get_home_path(), '.data', 'logger')

def make_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Timer(object):
    def __init__(self):
        self.cumulative_secs = {}
        self.current_ticks = {}
        pass

    def tick(self, name='default'):
        if name not in self.current_ticks:
            self.current_ticks[name] = time.time()
        else:
            if name not in self.cumulative_secs:
                self.cumulative_secs[name] = 0
            t = time.time()
            self.cumulative_secs[name] += t - self.current_ticks[name]
            self.current_ticks.pop(name)

    def tock(self, name='default'):
        self.tick(name)
        print('Time taken for {0}: {1:.1f}s'.format(name, self.cumulative_secs[name]))
        self.cumulative_secs.pop(name)
        self.current_ticks.pop(name, None)

