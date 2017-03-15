import tensorflow as tf

from spodernet.interfaces import IAtBatchPreparedObservable
from spodernet.utils.util import Timer
from spodernet.utils.global_config import Config

class TensorFlowConfig:
    inp = None
    support = None
    input_length = None
    support_length = None
    target = None
    index = None

    @staticmethod
    def init_batch_size(batch_size):
        TensorFlowConfig.inp = tf.placeholder(tf.int64, [batch_size, None])
        TensorFlowConfig.support = tf.placeholder(tf.int64, [batch_size, None])
        TensorFlowConfig.input_length = tf.placeholder(tf.int64, [batch_size,])
        TensorFlowConfig.support_length = tf.placeholder(tf.int64, [batch_size,])
        TensorFlowConfig.target = tf.placeholder(tf.int64, [batch_size])
        TensorFlowConfig.index = tf.placeholder(tf.int64, [batch_size])



class TensorFlowConverter(IAtBatchPreparedObservable):

    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        if TensorFlowConfig.inp == None:
            log.error('You need to initialize the batch size via TensorflowConfig.init_batch_size(batchsize)!')
        feed_dict = {}
        feed_dict[TensorFlowConfig.inp] = inp
        feed_dict[TensorFlowConfig.support] = sup
        feed_dict[TensorFlowConfig.input_length] = inp_len
        feed_dict[TensorFlowConfig.support_length] = sup_len
        feed_dict[TensorFlowConfig.target] = t
        feed_dict[TensorFlowConfig.index] = idx
        return feed_dict

def train_classification(model, train_batcher, dev_batcher, test_batcher=None, epochs=5):
    optimizer = tf.train.AdamOptimizer(0.001)
    print('starting training...')
    t0 = Timer()
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

            if i == 500: break

        for i, feed_dict in enumerate(dev_batcher):
            argmax = sess.run([predict], feed_dict=feed_dict)[0]

            dev_batcher.state.argmax = argmax
            dev_batcher.state.targets = feed_dict[TensorFlowConfig.target]


