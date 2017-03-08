from os.path import join

import numpy as np
import cPickle as pickle
import os

from spodernet.util import get_data_path, numpy2hdf, make_dirs_if_not_exists, hdf2numpy

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
            while not (   isinstance(level, basestring)
                       or isinstance(level, long)):
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
                    level = level[0]
                    i+=1
            self.successive_for_loops_to_list_of_tokens = i-1

        if self.successive_for_loops_to_list_of_tokens == 0:
            ret = self.process_list_of_tokens(sample, inp_type, samples_idx)

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
    def __init__(self, name, samples_per_file=50000):
        super(StreamToHDF5, self).__init__()
        self.max_length = None
        self.samples_per_file = samples_per_file
        self.path = join(get_data_path(), name)
        make_dirs_if_not_exists(self.path)
        self.shard_id = {'target' : 0, 'input' : 0, 'support' : 0}
        self.data = {'target' : [], 'input' : [], 'support' : []}
        self.num_samples = None
        self.idx = 0

    def link_with_pipeline(self, state):
        self.state = state
        self.root = self.state['path']

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



class CreateBinsByNestedLength(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, name, min_batch_size=128, bins_of_same_length=True, raise_on_throw_away_fraction=0.2):
        super(CreateBinsByNestedLength, self).__init__()
        self.min_batch_size = min_batch_size
        self.pure_bins = bins_of_same_length
        self.raise_fraction = raise_on_throw_away_fraction
        self.bin_fractions = []
        self.bin_idx2length_tuple = {}
        self.length_key2bin_idx = {}
        self.performed_search = False
        self.name = name
        self.max_sample_idx = -1
        self.inp_type2idx = {'support' : 0, 'input' : 0}
        self.idx2data = {'support' : {}, 'input' : {}}
        self.binidx2data = {'support' : {}, 'input' : {}}
        self.binidx2bincount = {}
        self.binidx2numprocessed = {'support' : {}, 'input' : {}}

    def link_with_pipeline(self, state):
        self.state = state
        self.base_path = join(self.state['path'], self.name)
        self.temp_file_path = join(self.base_path, 'remaining_data.tmp')
        make_dirs_if_not_exists(self.base_path)

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type not in self.inp_type2idx: return tokens
        if 'lengths' not in self.state['data']:
            log.error(('Do a first pass to produce lengths first, that is use the "SaveLengths" ',
                       'processor, execute, clean processors, then rerun the pipeline with this module.'))
        if inp_type not in self.state['data']['lengths']:
            log.error(('Do a first pass to produce lengths first, that is use the "SaveLengths" ',
                       'processor, execute, clean processors, then rerun the pipeline with this module.'))
        if not self.performed_search:
            self.perform_bin_search()

            assert (   isinstance(tokens[0], long)
                    or isinstance(tokens[0], np.int64)
                    or isinstance(tokens[0], int)
                    or isinstance(tokens[0], np.int32)), \
                    'Token need to be either int or longs (or numpy int or longs) for binning to work!'

        idx = self.inp_type2idx[inp_type]
        self.idx2data[inp_type][idx] = tokens
        if idx in self.idx2data['input'] and idx in self.idx2data['support']:
            x1 = self.idx2data['input'][idx]
            x2 = self.idx2data['support'][idx]
            l1 = len(x1)
            l2 = len(x2)
            key = str(l1) + ',' + str(l2)
            if key not in self.length_key2bin_idx:
                self.inp_type2idx[inp_type] += 1
                return
            bin_idx = self.length_key2bin_idx[key]
            self.binidx2data['input'][bin_idx].append(np.array(x1))
            self.binidx2data['support'][bin_idx].append(np.array(x2))
            self.binidx2numprocessed[bin_idx] += 1
            self.inp_type2idx[inp_type] += 1

            if (len(self.binidx2data['input']) % 100 == 0
                    or (     self.binidx2numprocessed[bin_idx] == self.binidx2bincount[bin_idx]
                         and len(self.binidx2data['input'][bin_idx]) > 0)):
                X_new = np.array(self.binidx2data['input'][bin_idx])
                S_new = np.array(self.binidx2data['support'][bin_idx])
                pathX = join(self.base_path, 'input_bin_{0}.hdf5'.format(bin_idx))
                pathS = join(self.base_path, 'support_bin_{0}.hdf5'.format(bin_idx))
                if os.path.exists(pathX):
                    X_old = hdf2numpy(pathX)
                    S_old = hdf2numpy(pathS)
                    X = np.vstack([X_old, X_new])
                    S = np.vstack([S_old, S_new])
                else:
                    X = X_new
                    S = S_new


                numpy2hdf(pathX, X)
                numpy2hdf(pathS, S)
                del self.binidx2data['input'][bin_idx][:]
                del self.binidx2data['support'][bin_idx][:]

        else:
            self.inp_type2idx[inp_type] += 1

        return tokens


    def perform_bin_search(self):
        l1 = np.array(self.state['data']['lengths']['input'])
        l2 = np.array(self.state['data']['lengths']['support'])
        if self.pure_bins == False:
            raise NotImplementedError('Bin search currently only works for bins that feature samples of the same length')
        if self.pure_bins:
            self.wasted_lengths, self.length_tuple2bin_size = self.calculate_wastes(l1, l2)

        total_count = 0.0
        for i, ((l1, l2), count) in enumerate(self.length_tuple2bin_size):
            total_count += count

        for i, ((l1, l2), count) in enumerate(self.length_tuple2bin_size):
            key = str(l1) + ',' + str(l2)
            self.length_key2bin_idx[key] = i
            self.binidx2data['input'][i] = []
            self.binidx2data['support'][i] = []
            self.binidx2numprocessed[i] = 0
            self.binidx2bincount[i] = count
            self.bin_idx2length_tuple[i] = [(l1, l2), count]
            self.bin_fractions.append(count/total_count)
        config = {}
        config['bin_fractions'] = self.bin_fractions
        config['bin_idx2length_tuple'] = self.bin_idx2length_tuple
        self.config = config
        pickle.dump(config, open(join(self.base_path, 'bin_config.pkl'), 'w'), pickle.HIGHEST_PROTOCOL)

        self.performed_search = True
        self.max_sample_idx = l1.size-1


    def calculate_wastes(self, l1, l2):
        wasted_samples = 0.0
        # get non-zero bin count, and the lengths corresponding to the bins
        counts_unfiltered = np.bincount(l1)
        lengths = np.arange(counts_unfiltered.size)
        counts = counts_unfiltered[counts_unfiltered > 0]
        lengths = lengths[counts_unfiltered > 0]
        indices = np.argsort(counts)
        # from smallest bin_counts to largest
        # look how many bins of l2 (support) are smaller than the min_batch_size
        wasted_lengths = []
        bin_by_size = []
        total_bin_count = 0.0
        for idx in indices:
            l1_waste = lengths[idx]
            l2_index = np.where(l1==l1_waste)[0]
            l2_counts_unfiltered = np.bincount(l2[l2_index])
            lengths2 = np.arange(l2_counts_unfiltered.size)
            l2_counts = l2_counts_unfiltered[l2_counts_unfiltered > 0]
            lengths2 = lengths2[l2_counts_unfiltered > 0]
            # keep track of the size of nested bins which will be included
            for length, bin_count in zip(lengths2, l2_counts):
                if bin_count >= self.min_batch_size:
                    bin_by_size.append(((l1_waste, length), bin_count))
                    total_bin_count += bin_count
            l2_waste = lengths2[l2_counts < self.min_batch_size]
            wasted_lengths.append([l1_waste, l2_waste])
            wasted_samples += np.sum(l2_counts[l2_counts < self.min_batch_size])
        wasted_fraction = wasted_samples / l1.size
        log.info('Wasted fraction for batch size {0} is {1}', self.min_batch_size, wasted_fraction)
        if wasted_fraction > self.raise_fraction:
            raise Exception('Wasted fraction higher than the raise error threshold of {0}!'.format(self.raise_fraction))

        # assign this here for testing purposes
        self.wasted_fraction = wasted_fraction
        self.total_bin_count = total_bin_count

        return wasted_lengths, bin_by_size



