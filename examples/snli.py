'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function

from spodernet.data.snli2spoder import snli2spoder
from spodernet.preprocessing import spoder2hdf5
from spodernet.util import load_hdf_file, load_hdf5_paths
from spodernet.hooks import AccuracyHook, LossHook, ETAHook
from spodernet.models import DualLSTM
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import AddToVocab, CreateBinsByNestedLength, SaveLengthsToState, ConvertTokenToIdx, StreamToHDF5, Tokenizer, NaiveNCharTokenizer
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.logger import Logger, LogLevel
from spodernet.util import Timer
from spodernet.global_config import Config, Backends
from spodernet.backends.tfbackend import TensorFlowConfig
from spodernet.model import Model


import nltk
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats.mstats import normaltest
#import torch.nn.utils.rnn as rnn_utils
np.set_printoptions(suppress=True)
import time

def preprocess_SNLI(delete_data=False):
    # load data
    names, file_paths = snli2spoder()
    train_path, dev_path, test_path = file_paths
    tokenizer = nltk.tokenize.WordPunctTokenizer()

    not_t = []
    t = ['input', 'support', 'target']
    # tokenize and convert to hdf5
    # 1. Setup pipeline to save lengths and generate vocabulary
    p = Pipeline('snli_example', delete_data)
    p.add_path(train_path)
    p.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p.add_token_processor(AddToVocab())
    p.add_post_processor(SaveLengthsToState())
    p.execute()
    p.clear_processors()
    p.state['vocab'].save_to_disk()

    # 2. Process the data further to stream it to hdf5
    p.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p.add_post_processor(ConvertTokenToIdx())
    p.add_post_processor(CreateBinsByNestedLength('snli_train', min_batch_size=128))
    state = p.execute()

    # dev and test data
    p2 = Pipeline('snli_example')
    p2.add_vocab(p)
    p2.add_path(dev_path)
    p2.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p2.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p2.add_post_processor(SaveLengthsToState())
    p2.execute()

    p2.clear_processors()
    p2.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p2.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p2.add_post_processor(ConvertTokenToIdx())
    p2.add_post_processor(StreamToHDF5('snli_dev'))
    p2.execute()

    p3 = Pipeline('snli_example')
    p3.add_vocab(p)
    p3.add_path(test_path)
    p3.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p3.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p3.add_post_processor(SaveLengthsToState())
    p3.execute()

    p3.clear_processors()
    p3.add_sent_processor(Tokenizer(tokenizer.tokenize), t)
    #p3.add_sent_processor(NaiveNCharTokenizer(3), not_t)
    p3.add_post_processor(ConvertTokenToIdx())
    p3.add_post_processor(StreamToHDF5('snli_test'))
    p3.execute()

def train_torch(train_batcher, train_batcher_error, dev_batcher, test_batcher, model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    t0 = time.time()
    print('starting training...')
    t0= Timer()
    print(epochs)
    for epoch in range(epochs):
        print(epoch)
        model.train()
        t0.tick()
        for i, (inp, inp_len, sup, sup_len, t, idx) in enumerate(train_batcher):
            t0.tick()

            optimizer.zero_grad()
            pred = model.forward_to_output(inp, sup, inp_len, sup_len, t)
            #print(pred)
            loss = F.nll_loss(pred, t)
            #print(loss)
            loss.backward()
            optimizer.step()
            maxiumum, argmax = torch.topk(pred.data, 1)
            #train_batcher.add_to_hook_histories(t, argmax, loss)
            train_batcher.state.argmax = argmax
            train_batcher.state.targets = t
            t0.tick()
        t0.tick()
        t0.tock()

        model.eval()
        for i, (inp, inp_len, sup, sup_len, t, idx) in enumerate(train_batcher_error):
            pred = model.forward_to_output(inp, sup, inp_len, sup_len, t)
            maxiumum, argmax = torch.topk(pred.data, 1)
            train_batcher_error.state.argmax = argmax
            train_batcher_error.state.targets = t

            if i == 1010: break

        for inp, inp_len, sup, sup_len, t, idx in dev_batcher:
            pred = model.forward_to_output(inp, sup, inp_len, sup_len, t)
            maxiumum, argmax = torch.topk(pred.data, 1)
            dev_batcher.state.argmax = argmax
            dev_batcher.state.targets = t
            #dev_batcher.add_to_hook_histories(t, argmax, loss)


def main():
    Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG
    Config.backend = Backends.TENSORFLOW
    Config.cuda = True
    Config.dropout = 0.2
    Config.l2 = 0.0001

    do_process = False
    if do_process:
        preprocess_SNLI(delete_data=False)


    p = Pipeline('snli_example')
    vocab = p.state['vocab']
    vocab.load_from_disk()

    batch_size = 128
    TensorFlowConfig.init_batch_size(batch_size)
    train_batcher = StreamBatcher('snli_example', 'snli_train', batch_size, randomize=True, loader_threads=2)
    train_batcher_error = StreamBatcher('snli_example', 'snli_train', batch_size, randomize=True, loader_threads=8, seed=2345)
    dev_batcher = StreamBatcher('snli_example', 'snli_dev', batch_size)
    test_batcher  = StreamBatcher('snli_example', 'snli_test', batch_size)

    train_batcher.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=10))
    dev_batcher.subscribe_to_events(AccuracyHook('Dev', print_every_x_batches=10000))

    if Config.backend == Backends.TORCH:
        print('using torch backend')
        model = SNLIClassification(batch_size, vocab, use_cuda=Config.cuda)
        if Config.cuda:
            model.cuda()

        train_torch(train_batcher, train_batcher_error, dev_batcher, test_batcher, model, epochs=100)
    else:
        print('using tensorflow backend')
        from spodernet.backends.tfbackend import train_classification
        from spodernet.backends.tfmodels import Embedding, PairedBiDirectionalLSTM, SoftmaxCrossEntropy
        model = Model()
        model.add(Embedding(embedding_size=256, num_embeddings=vocab.num_embeddings))
        model.add(PairedBiDirectionalLSTM(hidden_size=128))
        model.add(SoftmaxCrossEntropy(num_labels=3))

        train_classification(model, train_batcher, dev_batcher, test_batcher, epochs=100)


if __name__ == '__main__':
    main()
