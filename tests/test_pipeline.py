from io import StringIO
from os.path import join

import uuid
import os
import nltk
import pytest
import simplejson as json
import numpy as np
import shutil
import cPickle as pickle
import itertools
import scipy.stats
from io import StringIO
import dill
import simplejson

from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import Tokenizer, SaveStateToList, AddToVocab, ToLower, ConvertTokenToIdx, SaveLengthsToState
from spodernet.preprocessing.processors import JsonLoaderProcessors, RemoveLineOnJsonValueCondition, DictKey2ListMapper, RemoveUnnecessaryDimensions
from spodernet.preprocessing.processors import StreamToNumpyTable, StreamToHDF5, CreateBinsByNestedLength
from spodernet.preprocessing.vocab import Vocab
from spodernet.preprocessing.batching import StreamBatcher, BatcherState
from spodernet.utils.util import get_data_path, load_hdf_file
from spodernet.utils.global_config import Config, Backends
from spodernet.hooks import LossHook, AccuracyHook, ETAHook

from diskhash.core import NumpyTable

from spodernet.utils.logger import Logger, LogLevel
log = Logger('test_pipeline.py.txt')

Logger.GLOBAL_LOG_LEVEL = LogLevel.STATISTICAL
Logger.LOG_PROPABILITY = 0.1
Config.backend = Backends.TEST

def get_test_data_path_dict():
    paths = {}
    paths['snli'] = './tests/test_data/snli.json'
    paths['snli3k'] = './tests/test_data/snli_3k.json'
    paths['snli1k'] = './tests/test_data/snli_1k.json'
    paths['wiki'] = './tests/test_data/wiki.json'

    return paths

def test_dict2listmapper():
    with open(join(get_data_path(), 'test.txt'), 'wb') as f:
        for i in range(10):
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = str(i+4)
            f.write(json.dumps(test_dict) + '\n')

    p = Pipeline('abc')
    p.add_path(join(get_data_path(), 'test.txt'))
    p.add_line_processor(JsonLoaderProcessors())
    p.add_line_processor(DictKey2ListMapper(['key3', 'key1', 'key2']))
    p.add_text_processor(SaveStateToList('lines'))
    state = p.execute()
    for i, line in enumerate(state['data']['lines']['input']):
        assert int(line) == i+4, 'Input values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['support']):
        assert int(line) == i+5, 'Support values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['target']):
        assert int(line) == i+3, 'Target values does not correspond to the json key mapping.'

    os.remove(join(get_data_path(), 'test.txt'))
    shutil.rmtree(join(get_data_path(), 'abc'))

def test_remove_on_json_condition():
    with open(join(get_data_path(), 'test.txt'), 'wb') as f:
        for i in range(10):
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = str(i+4)
            f.write(json.dumps(test_dict) + '\n')
            test_dict = {}
            test_dict['key1'] = str(i+5)
            test_dict['key2'] = str(i+3)
            test_dict['key3'] = 'remove me'
            f.write(json.dumps(test_dict) + '\n')

    p = Pipeline('abc')
    p.add_path(join(get_data_path(), 'test.txt'))
    p.add_line_processor(JsonLoaderProcessors())
    p.add_line_processor(RemoveLineOnJsonValueCondition('key3', lambda inp: inp == 'remove me'))
    p.add_line_processor(DictKey2ListMapper(['key3', 'key1', 'key2']))
    p.add_text_processor(SaveStateToList('lines'))
    state = p.execute()

    assert len(state['data']['lines']['input']) == 10, 'Length different from filtered length!'
    for i, line in enumerate(state['data']['lines']['input']):
        assert int(line) == i+4, 'Input values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['support']):
        assert int(line) == i+5, 'Support values does not correspond to the json key mapping.'
    for i, line in enumerate(state['data']['lines']['target']):
        assert int(line) == i+3, 'Target values does not correspond to the json key mapping.'

    os.remove(join(get_data_path(), 'test.txt'))
    shutil.rmtree(join(get_data_path(), 'abc'))


def test_tokenization():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_sent_processor(SaveStateToList('tokens'))
    state = p.execute()

    inp_sents = state['data']['tokens']['input']
    sup_sents = state['data']['tokens']['support']
    sents = inp_sents + sup_sents
    log.statistical('input sentence of tokens: {0}', 0.5, inp_sents[0])
    log.statistical('support sentence of tokens: {0}', 0.5, sup_sents[0])

    # 2. setup nltk tokenization
    with open(get_test_data_path_dict()['snli']) as f:
        tokenized_sents = {'input' : [], 'support' : []}
        for line in f:
            inp, sup, t = json.loads(line)
            tokenized_sents['input'].append(tokenizer.tokenize(inp))
            tokenized_sents['support'].append(tokenizer.tokenize(sup))
            log.statistical('input sentence of tokens: {0}', 0.01, tokenized_sents['input'][-1])
            log.statistical('support sentence of tokens: {0}', 0.01, tokenized_sents['support'][-1])

    sents_nltk = tokenized_sents['input'] + tokenized_sents['support']
    # 3. test equality
    assert len(sents) == len(sents_nltk), 'Sentence count differs!'
    log.debug('count should be 200: {0}', len(sents))
    for sent1, sent2 in zip(sents, sents_nltk):
        assert len(sent1) == len(sent2), 'Token count differs!'
        log.statistical('a sentence of tokens: {0}', 0.01, sent1)
        for token1, token2 in zip(sent1, sent2):
            assert token1 == token2, 'Token values differ!'
            log.statistical('a token: {0}', 0.001, token1)

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
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    state = p.execute()

    # 1. use Vocab manually and test it against manual vocabulary
    idx2token = {}
    token2idx = {}
    token2idx['OOV'] = 0
    idx2token[0] = 'OOV'
    # empty = 0
    token2idx[''] = 1
    idx2token[1] = ''
    idx = 2
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
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, idx)
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, v.get_idx(token))

            for token in tokenizer.tokenize(sup):
                v.add_token(token)
                if token not in token2idx:
                    token2idx[token] = idx
                    idx2token[idx] = token
                    idx += 1
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, idx)
                    log.statistical('uncommon word if high number: {0}, {1}', 0.001, token, v.get_idx(token))

            v.add_label(t)
            log.statistical('label vocab index, that is small numbers: {0}', 0.01, v.idx2label.keys())


    # 3. Compare vocabs
    v2 = state['vocab']['general']
    for token in v.token2idx:
        assert v.token2idx[token] == v2.token2idx[token], 'Index for token not the same!'
        assert v.token2idx[token] == token2idx[token], 'Index for token not the same!'

    for idx in v.idx2token:
        assert v.idx2token[idx] == v2.idx2token[idx], 'Token for index not the same!'
        assert v.idx2token[idx] == idx2token[idx], 'Token for index not the same!'

    for label in v.label2idx:
        log.statistical('a label: {0}', 0.001, label)
        assert v.label2idx[label] == v2.label2idx[label], 'Index for label not the same!'

    for idx in v.idx2label:
        assert v.idx2label[idx] == v2.idx2label[idx], 'Label for index not the same!'


def test_separate_vocabs():

    # 1. write test data
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')
    with open(file_path, 'wb') as f:
        f.write(json.dumps(['0', 'a','-']) + '\n')
        f.write(json.dumps(['1', 'b','&']) + '\n')
        f.write(json.dumps(['2', 'c','#']) + '\n')

    # 2. read test data with pipeline
    p = Pipeline('test_pipeline')

    p.add_path(file_path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_token_processor(AddToVocab())
    state = p.execute()
    vocab = state['vocab']['general']
    inp_vocab = state['vocab']['input']
    sup_vocab = state['vocab']['support']
    tar_vocab = state['vocab']['target']

    # 6 token + empty and unknown = 8 
    assert vocab.num_token == 6 + 2, 'General vocab token count should be 8, but was {0} instead.'.format(vocab.num_token)
    assert vocab.num_labels == 3, 'General vocab token count should be 3, but was {0} instead.'.format(vocab.num_labels)

    assert inp_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(inp_vocab.num_token)
    assert inp_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(inp_vocab.num_labels)
    assert sup_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(sup_vocab.num_token)
    assert sup_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(sup_vocab.num_labels)
    assert tar_vocab.num_token == 3 + 2, 'General vocab token count should be 5, but was {0} instead.'.format(tar_vocab.num_token)
    assert tar_vocab.num_labels == 0, 'General vocab token count should be 0, but was {0} instead.'.format(tar_vocab.num_labels)

    for token in ['0', '1', '2']:
        assert token in vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in inp_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)

    for token in ['a', 'b', 'c']:
        assert token in vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in sup_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)

    for token in ['-', '&', '#']:
        assert token in vocab.label2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)
        assert token in tar_vocab.token2idx, 'Token {0} not found in the vocabulary when it should have been there!'.format(token)


def test_to_lower_sent():
    path = get_test_data_path_dict()['snli']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(ToLower())
    p.add_sent_processor(SaveStateToList('sents'))
    state = p.execute()

    inp_sents = state['data']['sents']['input']
    sup_sents = state['data']['sents']['support']
    sents = inp_sents + sup_sents

    # 2. test lowercase
    assert len(sents) == 200 # we have 100 samples for snli
    for sent in sents:
        log.statistical('lower case sentence {0}', 0.001, sent)
        assert sent == sent.lower(), 'Sentence is not lower case'

def test_to_lower_token():
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    path = get_test_data_path_dict()['snli']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(ToLower())
    p.add_token_processor(SaveStateToList('tokens'))
    state = p.execute()

    inp_tokens = state['data']['tokens']['input']
    sup_tokens = state['data']['tokens']['support']
    tokens = inp_tokens + sup_tokens

    # 2. test lowercase
    for token in tokens:
        log.statistical('lower case token: {0}', 0.0001, token)
        assert token == token.lower(), 'Token is not lower case'

def test_save_to_list_text():
    path = get_test_data_path_dict()['wiki']

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_text_processor(SaveStateToList('text'))
    state = p.execute()

    inp_texts = state['data']['text']['input']
    sup_texts = state['data']['text']['support']
    assert len(inp_texts) == 3, 'The input data size should be three samples, but found {0}'.format(len(inp_texts))
    assert len(inp_texts) == 3, 'The input data size should be three samples, but found {0}'.format(len(sup_texts))
    with open(path) as f:
        for inp1, sup1, line in zip(inp_texts, sup_texts, f):
            inp2, sup2, t = json.loads(line)
            log.statistical('a wikipedia paragraph: {0}', 0.5, sup1)
            assert inp1 == inp2, 'Saved text data not the same!'
            assert sup1 == sup2, 'Saved text data not the same!'


def test_save_to_list_sentences():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
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
            log.statistical('a list of sentences: {0}', 0.3, sup_sents)

    # 3. test equivalence
    assert len(inp_sents) == len(inp_sents2), 'Sentence count differs!'
    assert len(sup_sents) == len(sup_sents2), 'Sentence count differs!'

    for sent1, sent2 in zip(inp_sents, inp_sents2):
        assert sent1 == sent2, 'Saved sentence data not the same!'

    for sent1, sent2 in zip(sup_sents, sup_sents2):
        log.statistical('a sentence from a wiki paragraph: {0}', 0.3, sent1)
        assert sent1 == sent2, 'Saved sentence data not the same!'


def test_save_to_list_post_process():
    path = get_test_data_path_dict()['wiki']
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(path)
    p.add_line_processor(JsonLoaderProcessors())
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
        log.statistical('a wiki paragraph {0}', 0.1,  sample1)
        assert len(sample1) == len(sample2), 'Sentence count differs!'
        for sent1, sent2, in zip(sample1, sample2):
            log.statistical('a sentence of tokens of a wiki paragraph {0}', 0.01, sent1)
            assert len(sent1) == len(sent2), 'Token count differs!'
            for token1, token2 in zip(sent1, sent2):
                log.statistical('a token from a sentence of a wiki paragraph {0}', 0.001, token1)
                assert token1 == token2, 'Tokens differ!'



def test_convert_token_to_idx_no_sentences():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute()

    inp_indices = state['data']['idx']['input']
    label_idx = state['data']['idx']['target']
    log.statistical('a list of about 10 indices: {0}', 0.5, inp_indices[0])

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

            log.statistical('a list of about 10 indices {0}', 0.01, inp_idx)
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


def test_convert_to_idx_with_separate_vocabs():

    # 1. write test data
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')
    with open(file_path, 'wb') as f:
        f.write(json.dumps(['0', 'a','-']) + '\n')
        f.write(json.dumps(['1', 'b','&']) + '\n')
        f.write(json.dumps(['2', 'c','#']) + '\n')

    # 2. read test data with pipeline
    p = Pipeline('test_pipeline')

    p.add_path(file_path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_token_processor(AddToVocab())
    p.add_post_processor(ConvertTokenToIdx(keys=['input', 'support']))
    p.add_post_processor(SaveStateToList('idx'))
    state = p.execute()

    inp_indices = state['data']['idx']['input']
    sup_indices = state['data']['idx']['input']

    # 0 = UNK, 1 = '', 2,3,4 -> max index is 4
    assert np.max(inp_indices) == 2 + 2, 'Max idx should have been 2 if the vocabularies were separates!'
    assert np.max(sup_indices) == 2 + 2, 'Max idx should have been 2 if the vocabularies were separates!'

def test_save_lengths():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_line_processor(JsonLoaderProcessors())
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    p.add_post_processor(SaveLengthsToState())
    state = p.execute()

    lengths_inp = state['data']['lengths']['input']
    lengths_sup = state['data']['lengths']['support']
    log.statistical('a list of length values {0}', 0.5, lengths_inp)
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


names = ['loss', 'accuracy']
print_every = [20, 7, 13, 2000]
test_data = [r for r in itertools.product(names, print_every)]
ids = ['name={0}, print_every={1}'.format(name, print_every) for name, print_every in test_data]
@pytest.mark.parametrize("hook_name, print_every", test_data, ids=ids)
def test_hook(hook_name, print_every):
    def calc_confidence_interval(expected_loss):
        mean = np.mean(expected_loss)
        std = np.std(expected_loss)
        z = scipy.stats.norm.ppf(0.99)
        se = z*std/np.sqrt(print_every)
        lower_expected = mean-se
        upper_expected = mean+se
        return lower_expected, upper_expected, mean, n

    def generate_loss():
        loss = np.random.rand()
        state = BatcherState()
        state.loss = loss
        return loss, state

    def generate_accuracy():
        target = np.random.randint(0,3,print_every)
        argmax = np.random.randint(0,3,print_every)
        state = BatcherState()
        state.targets = target
        state.argmax = argmax
        accuracy = np.mean(target==argmax)
        return accuracy, state

    if hook_name == 'loss':
        hook = LossHook(print_every_x_batches=print_every)
        gen_func = generate_loss
    elif hook_name == 'accuracy':
        hook = AccuracyHook(print_every_x_batches=print_every)
        gen_func = generate_accuracy

    expected_loss = []
    state = BatcherState()
    for epoch in range(2):
        for i in range(100):
            metric, state = gen_func()
            expected_loss.append(metric)
            lower, upper, m, n = hook.at_end_of_iter_event(state)
            if (i+1) % print_every == 0:
                lower_expected, upper_expected, mean, n2 = calc_confidence_interval(expected_loss)
                assert n == n2, 'Sample size not equal!'
                assert np.allclose(m, mean), 'Mean not equal!'
                assert np.allclose(lower, lower_expected), 'Lower confidence bound not equal!'
                assert np.allclose(upper, upper_expected), 'Upper confidence bound not equal!'
                del expected_loss[:]

        lower, upper, m, n = hook.at_end_of_epoch_event(state)
        lower_expected, upper_expected, mean, n2 = calc_confidence_interval(expected_loss)
        del expected_loss[:]




def test_stream_to_numpytable():
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
    p.add_line_processor(JsonLoaderProcessors())
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
    streamer = StreamToNumpyTable(data_folder_name, ['input', 'support'])
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

    data = [X, S, t, X_len, S_len]
    tbls = []
    tbls.append(NumpyTable(data_folder_name + '_input', fixed_length=False, base_path=base_path))
    tbls.append(NumpyTable(data_folder_name + '_support', fixed_length=False, base_path=base_path))
    tbls.append(NumpyTable(data_folder_name + '_target', fixed_length=False, base_path=base_path))
    tbls.append(NumpyTable(data_folder_name + '_input_length', fixed_length=True, base_path=base_path))
    tbls.append(NumpyTable(data_folder_name + '_support_length', fixed_length=True, base_path=base_path))
    for tbl in tbls: tbl.init()
    # 5. Compare data
    for i, (var, tbl) in enumerate(zip(data, tbls)):
        idx = range(var.shape[0])
        np.testing.assert_array_equal(tbl[idx], var.reshape(100, -1), 'NumpyTable Stream data not equal')

    # 7. clean up
    shutil.rmtree(base_path)

test_data = [17, 128]
ids = ['batch_size={0}'.format(batch_size) for batch_size in test_data]
@pytest.mark.parametrize("batch_size", test_data, ids=ids)
def test_numpytable_batch_streamer(batch_size):
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    data_folder_name = 'snli_test'
    pipeline_folder = 'test_pipeline'
    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline(pipeline_folder)
    p.add_path(get_test_data_path_dict()['snli1k'])
    p.add_line_processor(JsonLoaderProcessors())
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
    streamer = StreamToNumpyTable(data_folder_name, ['input', 'support'])
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
    tbl_config_path  = join(base_path, 'tbl_config.pkl')
    tbl_config = simplejson.load(open(tbl_config_path))
    main_tbl = NumpyTable(tbl_config[0][1], fixed_length=False, base_path=base_path)
    main_tbl.init()
    assert len(main_tbl) == 1000, 'There should be 1000 samples for this dataset, but found {0}!'.format(len(main_tbl))
    tbls = [main_tbl]
    for i in range(1, len(tbl_config)):
        tbl = NumpyTable(tbl_config[i][1], fixed_length=False, base_path=base_path)
        tbl.init()
        tbls.append(tbl)

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

    epochs = 2
    if batch_size > 20:
        batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=4, same_length=False)
    else:
        batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=4)
    del batcher.at_batch_prepared_observers[:]
    batcher.at_batch_prepared_observers.append(RemoveUnnecessaryDimensions())

    # 4. test data equality
    for epoch in range(epochs):
        for x, x_len, s, s_len, t, idx in batcher:
            max_length_inp = x.shape[1]
            max_length_sup = s.shape[1]
            assert np.int32 == x_len.dtype, 'Input length type should be int32!'
            assert np.int32 == s_len.dtype, 'Support length type should be int32!'
            assert np.int32 == x.dtype, 'Input type should be int32!'
            assert np.int32 == s.dtype, 'Input type should be int32!'
            assert np.int32 == t.dtype, 'Target type should be int32!'
            assert np.int32 == idx.dtype, 'Index type should be int32!'
            np.testing.assert_array_equal(X[idx, :max_length_inp], x, 'Input data not equal!')
            assert np.sum(X[idx, max_length_inp:]) == 0.0, 'Padded region non-zero!'
            np.testing.assert_array_equal(S[idx, :max_length_sup], s, 'Support data not equal!')
            assert np.sum(S[idx, max_length_sup:]) == 0.0, 'Padded region non-zero!'
            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')

    # 5. clean up
    shutil.rmtree(base_path)
    for tbl in tbls:
        tbl.clear_table()


def test_abitrary_input_data():
    base_path = join(get_data_path(), 'test_keys')
    # clean all data from previous failed tests   
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    file_path = join(get_data_path(), 'test_pipeline', 'test_data.json')

    questions = [['bla bla Q1', 'this is q2', 'q3'], ['q4 set2', 'or is it q1?']]
    support = [['I like', 'multiple supports'], ['yep', 'they are pretty cool', 'yeah, right?']]
    answer = [['yes', 'absolutly', 'not really'], ['you bet', 'nah']]
    pos_tag = [['t1', 't2'], ['t1', 't2', 't3']]

    with open(file_path, 'w') as f:
        for i in range(2):
            f.write(json.dumps([questions[i], support[i], answer[i], pos_tag[i]]) + '\n')

    p = Pipeline('test_keys', ['question', 'support', 'answer', 'pos'])
    p.add_path(file_path)
    p.add_line_processor(JsonLoaderProcessors())
    p.add_token_processor(AddToVocab())
    p.add_post_processor(SaveLengthsToState())
    p.execute()

    p.clear_processors()
    p.add_token_processor(ConvertTokenToIdx(keys=['answer', 'pos']))
    p.add_post_processor(StreamToHDF5('test'))
    p.add_post_processor(SaveStateToList('data'))
    state = p.execute()

    Q = state['data']['data']['question']
    S = state['data']['data']['support']
    A = state['data']['data']['answer']
    pos = state['data']['data']['pos']
    print(Q)
    print(S)
    print(A)
    print(pos)


    assert False



#batch_size = [17, 128]
#samples_per_file = [500]
#randomize = [True, False]
#test_data = [r for r in itertools.product(samples_per_file, randomize, batch_size)]
#test_data.append((1000000, True, 83))
#str_func = lambda i, j, k: 'samples_per_file={0}, randomize={1}, batch_size={2}'.format(i, j, k)
#ids = [str_func(i,j,k) for i,j,k in test_data]
#test_idx = np.random.randint(0,len(test_data),3)
#@pytest.mark.parametrize("samples_per_file, randomize, batch_size", test_data, ids=ids)
#def test_non_random_stream_batcher(samples_per_file, randomize, batch_size):
#    tokenizer = nltk.tokenize.WordPunctTokenizer()
#    data_folder_name = 'snli_test'
#    pipeline_folder = 'test_pipeline'
#    base_path = join(get_data_path(), pipeline_folder, data_folder_name)
#    # clean all data from previous failed tests   
#    if os.path.exists(base_path):
#        shutil.rmtree(base_path)
#
#    # 1. Setup pipeline to save lengths and generate vocabulary
#    p = Pipeline(pipeline_folder)
#    p.add_path(get_test_data_path_dict()['snli1k'])
#    p.add_line_processor(JsonLoaderProcessors())
#    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
#    p.add_post_processor(SaveLengthsToState())
#    p.execute()
#    p.clear_processors()
#
#    # 2. Process the data further to stream it to hdf5
#    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
#    p.add_token_processor(AddToVocab())
#    p.add_post_processor(ConvertTokenToIdx())
#    p.add_post_processor(SaveStateToList('idx'))
#    # 2 samples per file -> 50 files
#    streamer = StreamToHDF5(data_folder_name, samples_per_file=samples_per_file)
#    p.add_post_processor(streamer)
#    state = p.execute()
#
#    # 2. Load data from the SaveStateToList hook
#    inp_indices = state['data']['idx']['input']
#    sup_indices = state['data']['idx']['support']
#    t_indices = state['data']['idx']['target']
#    max_inp_len = np.max(state['data']['lengths']['input'])
#    max_sup_len = np.max(state['data']['lengths']['support'])
#    # For SNLI the targets consist of single words'
#    assert np.max(state['data']['lengths']['target']) == 1, 'Max index label length should be 1'
#    assert 'counts' in streamer.config, 'counts key not found in config dict!'
#    assert len(streamer.config['counts']) > 0,'Counts of samples per file must be larger than zero (probably no files have been saved)'
#    if samples_per_file == 100000:
#        count = len(streamer.config['counts'])
#        assert count == 1,'Samples per files is 100000 and there should be one file for 1k samples, but there are {0}'.format(count)
#
#    assert streamer.num_samples == 1000, 'There should be 1000 samples for this dataset, but found {1}!'.format(streamer.num_samples)
#
#
#    # 3. parse data to numpy
#    n = len(inp_indices)
#    X = np.zeros((n, max_inp_len), dtype=np.int64)
#    X_len = np.zeros((n), dtype=np.int64)
#    S = np.zeros((n, max_sup_len), dtype=np.int64)
#    S_len = np.zeros((n), dtype=np.int64)
#    T = np.zeros((n), dtype=np.int64)
#
#    for i in range(len(inp_indices)):
#        sample_inp = inp_indices[i][0]
#        sample_sup = sup_indices[i][0]
#        sample_t = t_indices[i][0]
#        l = len(sample_inp)
#        X_len[i] = l
#        X[i, :l] = sample_inp
#
#        l = len(sample_sup)
#        S_len[i] = l
#        S[i, :l] = sample_sup
#
#        T[i] = sample_t[0]
#
#    epochs = 2
#    batcher = StreamBatcher(pipeline_folder, data_folder_name, batch_size, loader_threads=8, randomize=randomize)
#    del batcher.at_batch_prepared_observers[:]
#
#    # 4. test data equality
#    for epoch in range(epochs):
#        for x, x_len, s, s_len, t, idx in batcher:
#            assert np.int32 == x_len.dtype, 'Input length type should be int32!'
#            assert np.int32 == s_len.dtype, 'Support length type should be int32!'
#            assert np.int32 == x.dtype, 'Input type should be int32!'
#            assert np.int32 == s.dtype, 'Input type should be int32!'
#            assert np.int32 == t.dtype, 'Target type should be int32!'
#            assert np.int32 == idx.dtype, 'Index type should be int32!'
#            np.testing.assert_array_equal(X[idx], x, 'Input data not equal!')
#            np.testing.assert_array_equal(S[idx], s, 'Support data not equal!')
#            np.testing.assert_array_equal(X_len[idx], x_len, 'Input length data not equal!')
#            np.testing.assert_array_equal(S_len[idx], s_len, 'Support length data not equal!')
#            np.testing.assert_array_equal(T[idx], t, 'Target data not equal!')
#
#    # 5. clean up
#    shutil.rmtree(base_path)
