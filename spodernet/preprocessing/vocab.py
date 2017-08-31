from collections import Counter

import numpy as np
import os
import time
import datetime
import pickle
import urllib
import bashmagic
import time
import json

from spodernet.utils.util import get_data_path, save_data, xavier_uniform_weight
from os.path import join

from spodernet.utils.util import Logger
log = Logger('vocab.py.txt')

'''This models the vocabulary and token embeddings'''

class Vocab(object):
    '''Class that manages work/char embeddings'''

    def __init__(self, path, vocab = Counter(), labels = {}):
        '''Constructor.
        Args:
            vocab: Counter object with vocabulary.
        '''
        token2idx = {}
        idx2token = {}
        self.label2idx = {}
        self.idx2label = {}
        for i, item in enumerate(vocab.items()):
            token2idx[item[0]] = i+1
            idx2token[i+1] = item[0]

        for idx in labels:
            self.label2idx[labels[idx]] = idx
            self.idx2label[idx] = labels[idx]

        # out of vocabulary token
        token2idx['OOV'] = int(0)
        idx2token[int(0)] = 'OOV'
        # empty = 0
        token2idx[''] = int(1)
        idx2token[int(1)] = ''

        self.token2idx = token2idx
        self.idx2token = idx2token
        self.path = path
        if len(idx2token.keys()) > 0:
            self.next_idx = int(np.max(list(idx2token.keys())) + 1)
        else:
            self.next_idx = int(2)

        if len(self.idx2label.keys()) > 0:
            self.next_label_2dx = int(int(np.max(self.idx2label.keys())) + 1)
        else:
            self.next_label_idx = int(0)

    @property
    def num_token(self):
        return len(self.token2idx)

    @property
    def num_labels(self):
        return len(self.label2idx)

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.next_idx
            self.idx2token[self.next_idx] = token
            self.next_idx += 1

    def add_label(self, label):
        if label not in self.label2idx:
            self.label2idx[label] = self.next_label_idx
            self.idx2label[self.next_label_idx] = label
            self.next_label_idx += 1

    def get_idx(self, word):
        '''Gets the idx if it exists, otherwise returns -1.'''
        if word in self.token2idx:
            return self.token2idx[word]
        else:
            return self.token2idx['OOV']

    def get_idx_label(self, label):
        '''Gets the idx of the label'''
        return self.label2idx[label]

    def get_word(self, idx):
        '''Gets the word if it exists, otherwise returns OOV.'''
        if idx in self.idx2token:
            return self.idx2token[idx]
        else:
            return self.idx2token[0]

    def save_to_disk(self, name=''):
        log.info('Saving vocab to: {0}'.format(self.path))
        pickle.dump([self.token2idx, self.idx2token, self.label2idx,
            self.idx2label], open(self.path + name, 'wb'))

    def load_from_disk(self, name=''):
        if not os.path.exists(self.path + name):
            return False
        timestamp = time.ctime(os.path.getmtime(self.path + name))
        timestamp = datetime.datetime.strptime(timestamp, '%a %b %d %H:%M:%S %Y')
        age_in_hours = (datetime.datetime.now() - timestamp).seconds/60./60.
        log.info('Loading vocab from: {0}'.format(self.path + name))
        self.token2idx, self.idx2token, self.label2idx, self.idx2label = pickle.load(open(self.path, 'rb'))
        if age_in_hours > 12:
            log.info('Vocabulary outdated: {0}'.format(self.path + name))
            return False
        else:
            return True

    def download_glove(self):
        if not os.path.exists(join(get_data_path(), 'glove')):
            log.info('Glove data is missing, dowloading data now...')
            os.mkdir(join(get_data_path(), 'glove'))
            bashmagic.wget("http://nlp.stanford.edu/data/glove.6B.zip", join(get_data_path(),'glove'))
            bashmagic.unzip(join(get_data_path(), 'glove', 'glove.6B.zip'), join(get_data_path(), 'glove'))

    def prepare_glove(self, dimension):
        if not os.path.exists(join(get_data_path(), 'glove', 'index_50.p')):
            dims = [50, 100, 200, 300]
            base_filename = 'glove.6B.{0}d.txt'
            paths = [join(get_data_path(), 'glove', base_filename.format(dim)) for dim in dims]
            for path, dim in zip(paths, dims):
                index = {}
                index = {'PATH' : path}
                with open(path) as f:
                    log.info('Building index for {0}', path)
                    while True:
                        prev_pos = f.tell()
                        line = f.readline()
                        if line == '': break
                        next_pos = f.tell()
                        data = line.strip().split(' ')
                        token = data[0]
                        index[token] = (prev_pos, next_pos)

                log.info('Saving glove index...')
                json.dump(index, open(join(get_data_path(), 'glove', 'index_{0}.p'.format(dim)), 'wb'))

        log.info('Loading glove index...')
        index = json.load(open(join(get_data_path(), 'glove', 'index_{0}.p'.format(dimension)), 'rb'))

        return index


    def load_matrix(self, index, dim):
        p = index['PATH']
        log.info('Initializing glove matrix...')
        X = xavier_uniform_weight(len(self.token2idx), dim)
        log.info('Loading vectors into glove matrix with dimension: {0}', X.shape)
        pretrained_count = 0
        n = len(self.token2idx)-2
        with open(p) as f:
            for i, (token, idx) in enumerate(self.token2idx.items()):
                if i % 10000 == 0: print(i)
                if token in index:
                    start, end = index[token]
                    f.seek(start)
                    line = f.read(end-start)
                    data = line.strip().split(' ')
                    vec = data[1:]
                    X[idx] = vec
                    pretrained_count += 1
        log.info('Filled matrix with {0} pretrained embeddings and {1} xavier uniform initialized embeddings.', pretrained_count, n-pretrained_count)
        return X


    def get_glove_matrix(self, dimension):
        assert dimension in [50, 100, 200, 300], 'Dimension not supported! Only dimension 50, 100, 200, and 300 are supported!'
        self.download_glove()
        index = self.prepare_glove(dimension)
        return self.load_matrix(index, dimension)




