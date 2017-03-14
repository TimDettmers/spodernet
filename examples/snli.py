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
from spodernet.global_config import Config, Backends, TensorFlowConfig

#import tensorflow as tf

import nltk
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats.mstats import normaltest
#import torch.nn.utils.rnn as rnn_utils
np.set_printoptions(suppress=True)
import time

import tensorflow as tf
from tensorflow import placeholder

class TFSNLI(object):
    def __init__(self, batch_size, vocab):
        self.batch_size = batch_size
        self.vocab = vocab

    def forward(self, embedding_size=256, output_size=128, scope=None):
        Q = TensorFlowConfig.inp
        S = TensorFlowConfig.support
        Q_len = TensorFlowConfig.input_length
        S_len = TensorFlowConfig.support_length
        t = TensorFlowConfig.target

        embeddings = tf.get_variable("embeddings", [self.vocab.num_embeddings, embedding_size],
                                initializer=tf.random_normal_initializer(0., 1./np.sqrt(embedding_size)),
                                trainable=True, dtype="float32")

        with tf.variable_scope("embedders") as varscope:
            seqQ = tf.nn.embedding_lookup(embeddings, Q)
            varscope.reuse_variables()
            seqS = tf.nn.embedding_lookup(embeddings, S)

        with tf.variable_scope(scope or "conditional_reader_seq1") as varscope1:
            #seq1_states: (c_fw, h_fw), (c_bw, h_bw)
            _, seq1_states = self.reader(seqQ, Q_len, output_size, scope=varscope1)
        with tf.variable_scope(scope or "conditional_reader_seq2") as varscope2:
            varscope1.reuse_variables()
            # each [batch_size x max_seq_length x output_size]
            outputs, states = self.reader(seqS, S_len, output_size, seq1_states, scope=varscope2)

        output = tf.concat([states[0][1], states[1][1]], 1)

        logits, loss, predict = self.predictor(output, t, 3)

        return logits, loss, predict

    def reader(self, inputs, lengths, output_size, contexts=(None, None), scope=None):
        with tf.variable_scope(scope or "reader") as varscope:

            cell = tf.contrib.rnn.LSTMCell(output_size, state_is_tuple=True,initializer=tf.contrib.layers.xavier_initializer())

            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-Config.dropout)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell,
                cell,
                inputs,
                sequence_length=lengths,
                initial_state_fw=contexts[0],
                initial_state_bw=contexts[1],
                dtype=tf.float32)

            return outputs, states

    def predictor(self, inputs, targets, target_size):
        init = tf.contrib.layers.xavier_initializer(uniform=True) #uniform=False for truncated normal
        logits = tf.contrib.layers.fully_connected(inputs, target_size, weights_initializer=init, activation_fn=None)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                labels=targets), name='predictor_loss')
        predict = tf.arg_max(tf.nn.softmax(logits), 1, name='prediction')
        return logits, loss, predict

class SNLIClassification(torch.nn.Module):
    def __init__(self, batch_size, vocab, use_cuda=False):
        super(SNLIClassification, self).__init__()
        self.batch_size = batch_size
        input_dim = 256
        hidden_dim = 128
        num_directions = 2
        layers = 1
        self.emb= torch.nn.Embedding(vocab.num_embeddings,
                input_dim, padding_idx=0)#, scale_grad_by_freq=True, padding_idx=0)
        self.projection_to_labels = torch.nn.Linear(2 * num_directions * hidden_dim, 3)
        self.dual_lstm = DualLSTM(self.batch_size,input_dim,
                hidden_dim,layers=layers,
                bidirectional=True,to_cuda=use_cuda )
        #self.init_weights()

        #print(i.size(1), input_dim)
        self.b1 = None
        self.b2 = None
        self.use_cuda = use_cuda

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.projection_to_labels.weight.data.uniform_(-initrange, initrange)

    def forward_to_output(self, input_seq, support_seq, inp_len, sup_len, targets):
        inputs = self.emb(input_seq)
        support = self.emb(support_seq)
        #inputs_packed = rnn_utils.pack_padded_sequence(inputs, inp_len.data.tolist(), True)
        #support_packed = rnn_utils.pack_padded_sequence(support, sup_len.data.tolist(), True)
        #(out_all1_packed, out_all2_packed), (h1, h2) = self.dual_lstm(inputs_packed, support_packed)
        #out_all1, lengths = rnn_utils.pad_packed_sequence(out_all1_packed, True)
        #out_all2, lengths = rnn_utils.pad_packed_sequence(out_all2_packed, True)
        (out_all1, out_all2), (h1, h2) = self.dual_lstm(inputs, support)

        if self.b1 == None:
            b1 = torch.ByteTensor(out_all1.size())
            b2 = torch.ByteTensor(out_all2.size())
            if self.use_cuda:
                b1 = b1.cuda()
                b2 = b2.cuda()
        #out1 = torch.index_select(out_all1,1,inp_len)
        #out2 = torch.index_select(out_all2,1,sup_len)
        b1.fill_(0)
        for i, num in enumerate(inp_len.data):
            b1[i,num-1,:] = 1
        out1 = out_all1[b1].view(self.batch_size,-1)

        b2.fill_(0)
        for i, num in enumerate(sup_len.data):
            b2[i,num-1,:] = 1
        out2 = out_all2[b2].view(self.batch_size,-1)

        out = torch.cat([out1,out2],1)

        #out1 = torch.transpose(out1,1,0).resize(self.b.batch_size,4*256)
        #out2 = torch.transpose(out2,1,0).resize(self.b.batch_size,4*256)
        #out1 = out1.view(self.b.batch_size,-1)
        #out2 = out2.view(self.b.batch_size,-1)
        #output = torch.cat([out1, out2],1)
        #output = torch.cat([out1[0], out2[0]],1)
        projected = self.projection_to_labels(out)
        #print(output)
        pred = F.log_softmax(projected)
        #print(pred[0:5])
        return pred

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

def train_tensorflow(train_batcher, train_batcher_error, dev_batcher, test_batcher, model, epochs=5):
    optimizer = tf.train.AdamOptimizer(0.001)
    t0 = time.time()
    print('starting training...')
    t0= Timer()
    sess = tf.Session()

    logits, loss, predict = model.forward()

    if Config.L2 != 0.0:
        loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * Config.L2


    min_op = optimizer.minimize(loss)

    tf.global_variables_initializer().run(session=sess)
    for epoch in range(epochs):
        for i, feed_dict in enumerate(train_batcher):
            _, argmax = sess.run([min_op, predict], feed_dict=feed_dict)

            train_batcher.state.argmax = argmax
            train_batcher.state.targets = feed_dict[TensorFlowConfig.target]

        for i, feed_dict in enumerate(train_batcher_error):
            argmax = sess.run([predict], feed_dict=feed_dict)[0]

            train_batcher_error.state.argmax = argmax
            train_batcher_error.state.targets = feed_dict[TensorFlowConfig.target]

            if i == 1010: break

        for i, feed_dict in enumerate(dev_batcher):
            argmax = sess.run([predict], feed_dict=feed_dict)[0]

            dev_batcher.state.argmax = argmax
            dev_batcher.state.targets = feed_dict[TensorFlowConfig.target]


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

    train_batcher.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=1000))
    train_batcher_error.subscribe_to_events(AccuracyHook('Train', print_every_x_batches=1000))
    dev_batcher.subscribe_to_events(AccuracyHook('Dev', print_every_x_batches=10000))
    #eta = ETAHook('Train', 10)
    #train_batcher.subscribe_to_events(eta)
    #train_batcher.subscribe_to_start_of_epoch_event(eta)

    if Config.backend == Backends.TORCH:
        print('using torch backend')
        model = SNLIClassification(batch_size, vocab, use_cuda=Config.cuda)
        if Config.cuda:
            model.cuda()

        train_torch(train_batcher, train_batcher_error, dev_batcher, test_batcher, model, epochs=100)
    else:
        print('using tensorflow backend')
        model = TFSNLI(batch_size=128, vocab=vocab)

        train_tensorflow(train_batcher, train_batcher_error, dev_batcher, test_batcher, model, epochs=100)


if __name__ == '__main__':
    main()
