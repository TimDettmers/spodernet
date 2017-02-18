'''This models manages the vocabulary and token embeddings'''
import cPickle as pickle


class Vocab(object):
    '''Class that manages work/char embeddings'''

    def __init__(self, vocab, path):
        '''Constructor.
        Args:
            vocab: Counter object with vocabulary.
        '''
        token2idx = {}
        idx2token = {}
        for i, item in enumerate(vocab.items()):
            token2idx[item[0]] = i+1
            idx2token[i+1] = item[0]

        # out of vocabulary token
        token2idx['OOV'] = -1
        idx2token[-1] = 'OOV'
        # empty = 0
        token2idx[''] = 0
        idx2token[0] = ''

        self.token2idx = token2idx
        self.idx2token = idx2token
        self.path = path

    def get_idx(self, word):
        '''Gets the idx if it exists, otherwise returns -1.'''
        if word in self.token2idx:
            return self.token2idx[word]
        else:
            return self.token2idx['OOV']

    def get_word(self, idx):
        '''Gets the word if it exists, otherwise returns OOV.'''
        if idx in self.idx2token:
            return self.idx2token[idx]
        else:
            return self.idx2token[-1]

    def save_to_disk(self):
        print('Saving vocab to: {0}'.format(self.path))
        pickle.dump([self.token2idx, self.idx2token], open(self.path, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def load_from_disk(self):
        print('Loading vocab from: {0}'.format(self.path))
        self.token2idx, self.idx2token = pickle.load(open(self.path))
