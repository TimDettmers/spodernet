from io import StringIO
from os.path import join

import uuid
import os
import nltk
import pytest
import json

from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import Tokenizer, SaveStateToList, AddToVocab, ToLower, ConvertTokenToIdx, SaveLengths
from spodernet.preprocessing.vocab import Vocab

from spodernet.logger import Logger, LogLevel

log = Logger('test_pipeline.py.txt')

Logger.GLOBAL_LOG_LEVEL = LogLevel.STATISTICAL
Logger.LOG_PROPABILITY = 0.1

def get_test_data_path_dict():
    paths = {}
    paths['snli'] = './tests/test_data/snli.json'
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
    p.add_target_processor(AddToVocab())
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
        print(token)
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

    # 2. setup manual sent -> token processing
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
    p.add_target_processor(AddToVocab())
    # for sents in sample -> for token in sents = 2 for loops to token
    p.add_post_processor(ConvertTokenToIdx(successive_for_loops_to_tokens=2))
    p.add_post_processor(SaveStateToList('idx'))
    p.add_target_processor(ConvertTokenToIdx(successive_for_loops_to_tokens=0, is_label=True))
    p.add_target_processor(SaveStateToList('idx'))
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
    for idx1, idx2 in zip(tokenized_sents['target'], label_idx):
        assert idx1 == idx2, 'Index for label differs!'

def test_save_lengths():
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    # 1. setup pipeline
    p = Pipeline('test_pipeline')
    p.add_path(get_test_data_path_dict()['snli'])
    p.add_sent_processor(Tokenizer(tokenizer.tokenize))
    # for sents in sample -> sents = list of tokens = 1 for loop
    p.add_post_processor(SaveLengths(successive_for_loops_to_sentences=1))
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
    assert False, 'Needs to be implemented first!'
