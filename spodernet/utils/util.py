from os.path import join
from scipy.sparse import csr_matrix, spmatrix

import h5py
import os
import time
import os
import numpy as np

from spodernet.utils.logger import Logger
log = Logger('util.py.txt')

def save_dense_hdf(path, data):
    '''Writes a numpy array to a hdf5 file under the given path.'''
    log.debug_once('Saving hdf5 file to: {0}', path)
    h5file = h5py.File(path, "w")
    h5file.create_dataset("default", data=data)
    h5file.close()


def load_dense_hdf(path, keyword='default'):
    '''Reads and returns a numpy array for a hdf5 file'''
    log.debug_once('Reading hdf5 file from: {0}', path)
    h5file = h5py.File(path, 'r')
    dset = h5file.get(keyword)
    data = dset[:]
    h5file.close()
    return data

def save_sparse_hdf(path, data):
    shape = data.shape
    sparse = csr_matrix(data)
    folder, filename = os.path.split(path)
    save_dense_hdf(join(folder, 'data_' + filename), sparse.data)
    save_dense_hdf(join(folder, 'indices_' + filename), sparse.indices)
    save_dense_hdf(join(folder, 'indptr_' + filename), sparse.indptr)
    save_dense_hdf(join(folder, 'shape_dense_' + filename), shape)
    save_dense_hdf(join(folder, 'shape_sparse_' + filename), sparse.shape)

def load_sparse_hdf(path, keyword='default'):
    folder, filename = os.path.split(path)
    data = load_dense_hdf(join(folder, 'data_' + filename))
    indices = load_dense_hdf(join(folder, 'indices_' + filename))
    indptr = load_dense_hdf(join(folder, 'indptr_' + filename))
    shape = load_dense_hdf(join(folder, 'shape_dense_' + filename))
    shape_sparse = load_dense_hdf(join(folder, 'shape_sparse_' + filename))
    return csr_matrix((data, indices, indptr), shape=shape_sparse).toarray().reshape(shape)

def load_data(path):
    folder, filename = os.path.split(path)
    if os.path.exists(join(folder, 'indptr_' + filename)):
        data = load_sparse_hdf(path)
        return data
    else:
        return load_dense_hdf(path)

def save_data(path, data):
    assert data.size > 0
    is_sparse = isinstance(data, spmatrix)
    if is_sparse:
        save_sparse_hdf(path, data)
        return

    zero = (data == 0.0).sum()
    percent = zero/float(data.size)
    if percent > 0.5:
        save_sparse_hdf(path, data)
    else:
        save_dense_hdf(path, data)


def load_hdf5_paths(paths, limit=None):
    data = []
    for path in paths:
        if limit != None:
            data.append(load_data(path)[:limit])
        else:
            data.append(load_data(path))
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
