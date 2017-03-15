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

