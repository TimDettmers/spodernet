from io import StringIO
from os.path import join

import uuid
import os
import nltk
import pytest
import json
import numpy as np
import shutil
import cPickle as pickle

from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import Tokenizer, SaveStateToList, AddToVocab, ToLower, ConvertTokenToIdx, SaveLengthsToState
from spodernet.preprocessing.processors import StreamToHDF5, CreateBinsByNestedLength
from spodernet.preprocessing.vocab import Vocab
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.util import get_data_path, hdf2numpy

from spodernet.logger import Logger, LogLevel

log = Logger('test_pipeline.py.txt')

Logger.GLOBAL_LOG_LEVEL = LogLevel.STATISTICAL
Logger.LOG_PROPABILITY = 0.1

def get_test_data_path_dict():
    paths = {}
    paths['snli'] = './tests/test_data/snli.json'
    paths['snli3k'] = './tests/test_data/snli_3k.json'
    paths['wiki'] = './tests/test_data/wiki.json'

    return paths

def test_tokenization():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    state = p.execute()

    inp_sents = state['data']['tokens']['input']
    sup_sents = state['data']['tokens']['support']
    sents = inp_sents + sup_sents
    log.statistical('input sentence of tokens: {0}', inp_sents[0])
    log.statistical('support sentence of tokens: {0}', sup_sents[0])

    # 2. setup nltk tokenization
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)
            tokenized_sents['input'].append(tokenizer.tokenize(inp))
            tokenized_sents['support'].append(tokenizer.tokenize(sup))
            log.statistical('input sentence of tokens: {0}', tokenized_sents['input'][-1])
            log.statistical('support sentence of tokens: {0}', tokenized_sents['support'][-1])

    sents_nltk = tokenized_sents['input'] + tokenized_sents['support']
    # 3. test equality
    assert len(sents) == len(sents_nltk), 'Sentence count differs!'
    log.debug('count should be 200: {0}', len(sents))
    for sent1, sent2 in zip(sents, sents_nltk):
        assert len(sent1) == len(sent2), 'Token count differs!'
        log.statistical('a sentence of tokens: {0}', sent1)
        for token1, token2 in zip(sent1, sent2):
            assert token1 == token2, 'Token values differ!'
            log.statistical('a token: {0}', token1)

def test_path_creation():
    names = []
    for i in range(100):
        names.append(str(uuid.uuid4()))

    for name in names:
        p = Pipeline(name)

    home = os.environ['HOME']
    paths = [ join(home, '.data', name) for name in names]
    for path in paths:
        assert os.path.exists(path)
        os.rmdir(path)

def test_vocab():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    state = p.execute()

    # 1. use Vocab manually and test it against manual vocabulary
    idx2token = {}
    token2idx = {}
    token2idx['OOV'] = -1
    idx2token[-1] = 'OOV'
    # empty = 0
    token2idx[''] = 0
    idx2token[0] = ''
    idx = 1
    v = Vocab('test')
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1
                    log.statistical('uncommon word if high number: {0}, {1}', token, idx)
                    log.statistical('uncommon word if high number: {0}, {1}', token, v.get_idx(token))

            for token in tokenizer.tokenize(sup):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1
                    log.statistical('uncommon word if high number: {0}, {1}', token, idx)
                    log.statistical('uncommon word if high number: {0}, {1}', token, v.get_idx(token))

            v.add_label(t)
            log.statistical('label vocab index, that is small numbers: {0}', v.idx2label.keys())


    # 3. Compare vocabs
    v2 = state['vocab']
    for token in v.token2idx:
        assert v.token2idx[token] == v2.token2idx[token], 'Index for token not the same!'
        assert v.token2idx[token] == token2idx[token], 'Index for token not the same!'

    for idx in v.idx2token:
        assert v.idx2token[idx] == v2.idx2token[idx], 'Token for index not the same!'
        assert v.idx2token[idx] == idx2token[idx], 'Token for index not the same!'

    for label in v.label2idx:
        log.statistical('a label: {0}', label)
        assert v.label2idx[label] == v2.label2idx[label], 'Index for label not the same!'

    for idx in v.idx2label:
        assert v.idx2label[idx] == v2.idx2label[idx], 'Label for index not the same!'


def test_to_lower_sent():
    path = get_test_data_path_dict()['snli']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute()

    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    sents = inp_sents + sup_sents

    # 2. test lowercase
    assert len(sents) == 200 # we have 100 samples for snli
    for sent in sents:
        log.statistical('lower case sentence {0}', sent)
        assert sent == sent.lower(), 'Sentence is not lower case'

def test_to_lower_token():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    path = get_test_data_path_dict()['snli']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(ToLower())
    p.add_token_processor(SaveStateToList('tokens'))
    state = p.execute()

    inp_tokens = state['data']['tokens']['input']
    sup_tokens = state['data']['tokens']['support']
    tokens = inp_tokens + sup_tokens

    # 2. test lowercase
    for token in tokens:
        log.statistical('lower case token: {0}', token)
        assert token == token.lower(), 'Token is not lower case'

def test_save_to_list_text():
    path = get_test_data_path_dict()['wiki']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_text_processor(SaveStateToList('text'))
    state = p.execute()

    inp_texts = state['data']['text']['input']
    sup_texts = state['data']['text']['support']
    with open(path) as f:
        for inp1, sup1, line in zip(inp_texts, sup_texts, f):
            inp2, sup2, t = json.loads(line)
            log.statistical('a wikipedia paragraph: {0}', sup1)
            assert inp1 == inp2, 'Saved text data not the same!'
            assert sup1 == sup2, 'Saved text data not the same!'


def test_save_to_list_sentences():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_text_processor(Tokenizer(sent_tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute()

    # 2. setup manual sentence processing
    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    inp_sents2 = []
    sup_sents2 = []
    with open(path) as f:
        for line in f:
            inp, sup, t = json.loads(line)
            sup_sents2 += sent_tokenizer.tokenize(sup)
            inp_sents2 += sent_tokenizer.tokenize(inp)
            log.statistical('a list of sentences: {0}', sup_sents)

    # 3. test equivalence
    assert len(inp_sents) == len(inp_sents2), 'Sentence count differs!'
    assert len(sup_sents) == len(sup_sents2), 'Sentence count differs!'

    for sent1, sent2 in zip(inp_sents, inp_sents2):
        assert sent1 == sent2, 'Saved sentence data not the same!'

    for sent1, sent2 in zip(sup_sents, sup_sents2):
        log.statistical('a sentence from a wiki paragraph: {0}', sent1)
        assert sent1 == sent2, 'Saved sentence data not the same!'


def test_save_to_list_post_process():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_text_processor(Tokenizer(sent_tokenizer.tokenize))
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveStateToList('samples'))
    state = p.execute()

    # 2. setup manual sentence -> token processing
    inp_samples = state['data']['samples']['input']
    sup_samples = state['data']['samples']['support']
    inp_samples2 = []
    sup_samples2 = []
    with open(path) as f:
        for line in f:
            sup_sents = []
            inp_sents = []
            inp, sup, t = json.loads(line)
            for sent in sent_tokenizer.tokenize(sup):
                sup_sents.append(tokenizer.tokenize(sent))
            for sent in sent_tokenizer.tokenize(inp):
                inp_sents.append(tokenizer.tokenize(sent))
            inp_samples2.append(inp_sents)
            sup_samples2.append(sup_sents)


    # 3. test equivalence
    for sample1, sample2 in zip(inp_samples, inp_samples2):
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                assert token1 == token2, 'Tokens differ!'

    for sample1, sample2 in zip(sup_samples, sup_samples2):
        log.statistical('a wiki paragraph {0}', sample1)
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            log.statistical('a sentence of tokens of a wiki paragraph {0}', sent1)
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                log.statistical('a token from a sentence of a wiki paragraph {0}', token1)
                assert token1 == token2, 'Tokens differ!'



def test_convert_token_to_idx_no_sentences():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute()

    inp_indices = state['data']['idx']['input']
    label_idx = state['data']['idx']['target']
    log.statistical('a list of about 10 indices: {0}', inp_indices[0])

    # 2. use Vocab manually
    v = Vocab('test')
    with open(get_test_data_path_dict()['snli']) as f:
        for line in f:
            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                v.add_token(token)

            for token in tokenizer.tokenize(sup):
                v.add_token(token)

            v.add_label(t)

    # 3. index manually
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : [], 'target' : []}
        for line in f:
            inp_idx = []
            sup_idx = []

            inp, sup, t = json.loads(line)

            for token in tokenizer.tokenize(inp):
                inp_idx.append(v.get_idx(token))

            for token in tokenizer.tokenize(sup):
                sup_idx.append(v.get_idx(token))

            log.statistical('a list of about 10 indices {0}', inp_idx)
            tokenized_sents['target'].append(v.get_idx_label(t))
            tokenized_sents['input'].append(inp_idx)
            tokenized_sents['support'].append(sup_idx)


    # 4. Compare idx
    assert len(tokenized_sents['input']) == len(inp_indices), 'Sentence count differs!'
    for sent1, sample in zip(tokenized_sents['input'], inp_indices):
        sent2 = sample[0] # in this case we do not have sentences
        assert len(sent1) == len(sent2), 'Index count (token count) differs!'
        for idx1, idx2 in zip(sent1, sent2):
            assert idx1 == idx2, 'Index for token differs!'

    # 5. Compare label idx
    for idx1, sample in zip(tokenized_sents['target'], label_idx):
        # sample[0] == sent
        # sent[0] = idx
        assert idx1 == sample[0][0], 'Index for label differs!'

def test_save_lengths():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    state = p.execute()

    lengths_inp = state['data']['lengths']['input']
    lengths_sup = state['data']['lengths']['support']
    log.statistical('a list of length values {0}', lengths_inp)
    lengths1 = lengths_inp + lengths_sup

    # 2. generate lengths manually
    lengths_inp2 = []
    lengths_sup2 = []
    with open(get_test_data_path_dict()['snli']) as f:
        for line in f:
            inp, sup, t = json.loads(line)

            lengths_inp2.append(len(tokenizer.tokenize(inp)))
            lengths_sup2.append(len(tokenizer.tokenize(sup)))

    lengths2 = lengths_inp2 + lengths_sup2

    # 3. test for equal lengths
    assert len(lengths1) == len(lengths2), 'Count of lengths differs!'
    assert len(lengths1) == 200, 'Count of lengths not as expected for SNLI test data!'
    for l1, l2 in zip(lengths1, lengths2):
        assert l1 == l2, 'Lengths of sentence differs!'


def test_stream_to_hdf5():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=2)
    p.add_post_processor(streamer)
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    t = np.zeros((n), dtype=np.int64)
    index = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        t[i] = sample_t[0]
        index[i] = i

    # 4. setup expected paths
    inp_paths = [join(base_path, 'input_' + str(i) + '.hdf5') for i in range(1, 50)]
    sup_paths = [join(base_path, 'support_' + str(i) + '.hdf5') for i in range(1, 50)]
    target_paths = [join(base_path, 'target_' + str(i) + '.hdf5') for i in range(1, 50)]
    inp_len_paths = [join(base_path, 'input_lengths_' + str(i) + '.hdf5') for i in range(1, 50)]
    sup_len_paths = [join(base_path, 'support_lengths_' + str(i) + '.hdf5') for i in range(1, 50)]
    index_paths = [join(base_path, 'index_' + str(i) + '.hdf5') for i in range(1, 50)]
    zip_iter = zip([X, S, t, X_len, S_len, index], [inp_paths, sup_paths, target_paths, inp_len_paths, sup_len_paths, index_paths])

    # 5. Compare data
    for data, paths in zip_iter:
        data_idx = 0
        for path in paths:
            assert os.path.exists(path), 'This file should have been created by the HDF5Streamer: {0}'.format(path)
            shard = hdf2numpy(path)
            start = data_idx*2
            end = (data_idx + 1)*2
            np.testing.assert_array_equal(shard, data[start:end], 'HDF5 Stream data not equal for path {0}'.format(path))
            data_idx += 1

    # 6. compare config
    config_path = join(base_path, 'hdf5_config.pkl')
    config_reference = streamer.config
    assert os.path.exists(config_path), 'No HDF5 config exists under the path: {0}'.format(config_path)
    config_dict = pickle.load(open(config_path))
    assert 'paths' in config_dict, 'paths key not found in config dict!'
    assert 'fractions' in config_dict, 'fractions key not found in config dict!'
    assert 'counts' in config_dict, 'counts key not found in config dict!'
    for paths1, paths2 in zip(config_dict['paths'], streamer.config['paths']):
        for path1, path2 in zip(paths1, paths2):
            assert path1 == path2, 'Paths differ from HDF5 config!'
    np.testing.assert_array_equal(config_dict['fractions'], streamer.config['fractions'], 'Fractions for HDF5 samples per file not equal!')
    np.testing.assert_array_equal(config_dict['counts'], streamer.config['counts'], 'Counts for HDF5 samples per file not equal!')

    path_types = ['input', 'support', 'input_length', 'support_length', 'target', 'index']
    for i, paths in enumerate(streamer.config['paths']):
        assert len(paths) == 6, 'One path type is missing! Required path types {0}, existing paths {1}.'.format(path_types, paths)

    # 7. clean up
    shutil.rmtree(base_path)


def test_bin_search():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli3k_bins'
    total_samples = 30000.0
    base_path = join(get_data_path(), 'test_pipeline', data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli3k'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Execute the binning procedure
    p.add_path(get_test_data_path_dict()['snli3k'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    bin_creator = CreateBinsByNestedLength(data_folder_name, min_batch_size=4)
    p.add_post_processor(bin_creator)
    state = p.execute()

    # 3. We proceed to test if the bin sizes are correct, the config is correct, 
    #    if the calculated fraction of wasted samples is correct.
    #    This makes use of the state of the CreateBins class itself which
    #    thus biases this test. Use statiatical logging for 
    #    additional verification of correctness.

    # 3.1 Test config equality
    config_path = join(base_path, 'bin_config.pkl')
    assert os.path.exists(base_path), 'Base path for binning does not exist!'
    assert os.path.exists(config_path), 'Config file for binning not found!'
    config_dict = pickle.load(open(config_path))
    assert 'paths' in config_dict, 'paths key not found in config dict!'
    assert 'fractions' in config_dict, 'fractions key not found in config dict!'
    assert 'counts' in config_dict, 'counts key not found in config dict!'
    for paths1, paths2 in zip(config_dict['paths'], bin_creator.config['paths']):
        for path1, path2 in zip(paths1, paths2):
                assert path1 == path2, 'Paths differ from bin config!'
    np.testing.assert_array_equal(config_dict['fractions'], bin_creator.config['fractions'], 'Fractions for HDF5 samples per file not equal!')
    np.testing.assert_array_equal(config_dict['counts'], bin_creator.config['counts'], 'Counts for HDF5 samples per file not equal!')

    path_types = ['input', 'support', 'input_length', 'support_length', 'target', 'index']
    for i, paths in enumerate(bin_creator.config['paths']):
        assert len(paths) == 6, 'One path type is missing! Required path types {0}, existing paths {1}.'.format(path_types, paths)
    num_idxs = len(bin_creator.config['paths'])
    paths_inp = [join(base_path, 'input_bin_{0}.hdf5'.format(i)) for i in range(num_idxs)]
    paths_sup = [join(base_path, 'support_bin_{0}.hdf5'.format(i)) for i in range(num_idxs)]

    # 3.2 Test length, count and total count equality
    num_samples_bins = bin_creator.total_bin_count
    cumulative_count = 0.0
    for i, (path_inp, path_sup) in enumerate(zip(paths_inp, paths_sup)):
        inp = hdf2numpy(path_inp)
        sup = hdf2numpy(path_sup)
        l1 = bin_creator.config['path2len'][path_inp]
        l2 = bin_creator.config['path2len'][path_sup]
        count = bin_creator.config['path2count'][path_sup]

        expected_bin_fraction = count/num_samples_bins
        actual_bin_fraction = bin_creator.config['fractions'][i]

        assert actual_bin_fraction == expected_bin_fraction, 'Bin fraction for bin {0} not equal'.format(i)
        assert inp.shape[0] == count, 'Count for input bin at {0} not as expected'.format(path_inp)
        assert sup.shape[0] == count, 'Count for support bin at {0} not as expected'.format(path_sup)
        assert inp.shape[1] == l1, 'Input data sequence length for {0} not as expected'.format(path_inp)
        assert sup.shape[1] == l2, 'Support data sequence length for {0} not as expected'.format(path_sup)

        cumulative_count += count

    assert cumulative_count == num_samples_bins, 'Number of total bin samples not as expected!'

    shutil.rmtree(base_path)


def test_non_random_stream_batcher():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli3k'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=500)
    p.add_post_processor(streamer)
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    batch_size = 128
    epochs = 5
    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8)

    # 4. test data equality
    for epoch in range(epochs):
        for x, x_len, s, s_len, t, idx in batcher:
            np.testing.assert_array_equal(X[idx], x, 'Input data not equal!')
            np.testing.assert_array_equal(S[idx], s, 'Support data not equal!')
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

    # 5. clean up
    shutil.rmtree(base_path)


def test_random_stream_batcher():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli3k'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    # 2 samples per file -> 50 files
    streamer = StreamToHDF5(data_folder_name, samples_per_file=500)
    p.add_post_processor(streamer)
    state = p.execute()

    # 2. Load data from the SaveStateToList hook
    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['support']
    t_indices = state['data']['idx']['target']
    max_inp_len = np.max(state['data']['lengths']['input'])
    max_sup_len = np.max(state['data']['lengths']['support'])
    # For SNLI the targets consist of single words'
    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'

    # 3. parse data to numpy
    n = len(inp_indices)
    X = np.zeros((n, max_inp_len), dtype=np.int64)
    X_len = np.zeros((n), dtype=np.int64)
    S = np.zeros((n, max_sup_len), dtype=np.int64)
    S_len = np.zeros((n), dtype=np.int64)
    T = np.zeros((n), dtype=np.int64)

    for i in range(len(inp_indices)):
        sample_inp = inp_indices[i][0]
        sample_sup = sup_indices[i][0]
        sample_t = t_indices[i][0]
        l = len(sample_inp)
        X_len[i] = l
        X[i, :l] = sample_inp

        l = len(sample_sup)
        S_len[i] = l
        S[i, :l] = sample_sup

        T[i] = sample_t[0]

    batch_size = 128
    epochs = 8
    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8, randomize=True)

    # 4. test data equality
    for epoch in range(epochs):
        for x, x_len, s, s_len, t, idx in batcher:
            np.testing.assert_array_equal(X[idx], x, 'Input data not equal!')
            np.testing.assert_array_equal(S[idx], s, 'Support data not equal!')
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

    # 5. clean up
    shutil.rmtree(base_path)

