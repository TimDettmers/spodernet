from os.path import join

import numpy as np

from spodernet.util import get_data_path, numpy2hdf, make_dirs_if_not_exists

from spodernet.logger import Logger
log = Logger('processors.py.txt')

class AbstractProcessor(object):
    def __init__(self):
        self.state = None
        pass


    def link_with_pipeline(self, state):
        self.state = state

    def process(self, inputs, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractProcessor need to implement the process method')

class AbstractLoopLevelTokenProcessor(AbstractProcessor):
    def __init__(self):
        super(AbstractLoopLevelTokenProcessor, self).__init__()
        self.successive_for_loops_to_tokens = None

    def process_token(self, token, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelTokenProcessor need to implement the process_token method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops_to_tokens == None:
            i = 0
            level = sample
            while not (isinstance(level, basestring) or isinstance(level, long)):
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_tokens = i

        if self.successive_for_loops_to_tokens == 0:
            ret = self.process_token(sample, inp_type)

        elif self.successive_for_loops_to_tokens == 1:
            new_tokens = []
            for token in sample:
                new_tokens.append(self.process_token(token, inp_type))
            ret = new_tokens

        elif self.successive_for_loops_to_tokens == 2:
            new_sents = []
            for sent in sample:
                new_tokens = []
                for token in sent:
                    new_tokens.append(self.process_token(token, inp_type))
                new_sents.append(new_tokens)
            ret = new_sents

        return ret

class AbstractLoopLevelListOfTokensProcessor(AbstractProcessor):
    def __init__(self):
        super(AbstractLoopLevelListOfTokensProcessor, self).__init__()
        self.successive_for_loops_to_list_of_tokens = None

    def process_list_of_tokens(self, tokens, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelListOfTokensProcessor need to implement the process_list_of_tokens method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops_to_list_of_tokens == None:
            i = 0
            level = sample
            while not (isinstance(level, basestring)
                       or isinstance(level, long)
                       or isinstance(level, np.int64)):
                    print(type(level))
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_list_of_tokens = i-1

        if self.successive_for_loops_to_list_of_tokens == 0:
            ret = self.process_list_of_tokens(sample, inp_type)

        elif self.successive_for_loops_to_list_of_tokens == 1:
            new_sents = []
            for sent in sample:
                new_sents.append(self.process_list_of_tokens(sent, inp_type))
            ret = new_sents

        return ret

class Tokenizer(AbstractProcessor):
    def __init__(self, tokenizer_method):
        super(Tokenizer, self).__init__()
        self.tokenize = tokenizer_method

    def process(self, sentence, inp_type):
        return self.tokenize(sentence)

class AddToVocab(AbstractProcessor):
    def __init__(self):
        super(AddToVocab, self).__init__()

    def process(self, token, inp_type):
        if inp_type == 'target':
            self.state['vocab'].add_label(token)
        else:
            self.state['vocab'].add_token(token)
        return token

class ToLower(AbstractProcessor):
    def __init__(self):
        super(ToLower, self).__init__()

    def process(self, token, inp_type):
        return token.lower()

class ConvertTokenToIdx(AbstractLoopLevelTokenProcessor):
    def __init__(self):
        super(ConvertTokenToIdx, self).__init__()

    def process_token(self, token, inp_type):
        if inp_type != 'target':
            log.statistical('a label {0}', token)
            return self.state['vocab'].get_idx(token)
        else:
            log.statistical('a non-label token {0}', token)
            return self.state['vocab'].get_idx_label(token)

class SaveStateToList(AbstractProcessor):
    def __init__(self, name):
        super(SaveStateToList, self).__init__()
        self.name = name

    def link_with_pipeline(self, state):
        self.state = state
        if self.name not in self.state['data']:
            self.state['data'][self.name] = {}
        self.data = self.state['data'][self.name]

    def process(self, data, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(data)
        return data

class SaveLengthsToState(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self):
        super(SaveLengthsToState, self).__init__()

    def link_with_pipeline(self, state):
        self.state = state
        if 'lengths' not in self.state['data']:
            self.state['data']['lengths'] = {}
        self.data = self.state['data']['lengths']

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(len(tokens))
        log.statistical('A list of tokens: {0}', tokens)
        return tokens

class StreamToHDF5(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, name, samples_per_file=50000, max_length=None):
        super(StreamToHDF5, self).__init__()
        self.max_length = max_length
        self.samples_per_file = samples_per_file
        self.path = join(get_data_path(), name)
        make_dirs_if_not_exists(self.path)

    def link_with_pipeline(self, state):
        self.state = state
        self.shard_id = {'target' : 0, 'input' : 0, 'support' : 0}
        self.root = self.state['path']
        self.data = {'target' : [], 'input' : [], 'support' : []}
        self.idx = 0
        self.num_samples = None

    def process_list_of_tokens(self, tokens, inp_type):
        if self.max_length is None:
            if 'lengths' not in self.state['data']:
                log.error(('Do a first pass to produce lengths first, that is use the "SaveLengths" ',
                           'processor, execute, clean processors, then rerun the pipeline with hdf5 streaming.'))
            if inp_type not in self.state['data']['lengths']:
                log.error(('Do a first pass to produce lengths first, that is use the "SaveLengths" ',
                           'processor, execute, clean processors, then rerun the pipeline with hdf5 streaming.'))
            if self.num_samples == None:
                self.num_samples = len(self.state['data']['lengths']['input'])

            max_length = np.max(self.state['data']['lengths'][inp_type])
            log.statistical('max length of the dataset: {0}', max_length)
            log.debug_once('Using type int32 for inputs and supports for now, but this may not be correct in the future')
            x = np.zeros((max_length), dtype=np.int32)
            x[:len(tokens)] = tokens
            if len(tokens) == 1:
                self.data[inp_type].append(x[0])
            else:
                self.data[inp_type].append(x)

        if len(self.data[inp_type]) == self.samples_per_file or self.idx == self.num_samples:
            self.save_to_hdf5(inp_type)

        return tokens

    def save_to_hdf5(self, inp_type):
        idx = self.shard_id[inp_type]
        X = np.array(self.data[inp_type])
        file_name = inp_type + '_' + str(idx+1) + '.hdf5'
        numpy2hdf(join(self.path, file_name), X)

        if inp_type != 'target':
            start = idx*self.samples_per_file
            end = (idx+1)*self.samples_per_file
            X_len = np.array(self.state['data']['lengths'][inp_type][start:end])
            file_name_len = inp_type + '_lengths_' + str(idx+1) + '.hdf5'
            numpy2hdf(join(self.path, file_name_len), X_len)

        self.shard_id[inp_type] += 1
        del self.data[inp_type][:]
