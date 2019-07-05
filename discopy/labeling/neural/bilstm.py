import logging
import os

import numpy as np
from keras.regularizers import l2
from sklearn.utils import compute_class_weight

logger = logging.getLogger('discopy')

from keras import Input, Model
from keras.layers import Bidirectional, Dense, Dropout, SpatialDropout1D
from keras.layers import CuDNNLSTM as LSTM


import tensorflow as tf


def get_balanced_class_weights(y):
    y = y.argmax(-1).flatten()
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes, y)
    return {c: w for c, w in zip(classes, class_weights)}


def class_weighted_loss(class_weights):
    def loss(onehot_labels, logits):
        c_weights = np.array([class_weights[i] for i in range(4)])
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=[onehot_labels], logits=[logits])
        weights = tf.reduce_sum(tf.multiply(onehot_labels, c_weights), axis=-1)
        weighted_losses = unweighted_losses * weights  # reduce the result to get your final loss
        total_loss = tf.reduce_mean(weighted_losses)
        return total_loss

    return loss


class BiLSTMx:

    def __init__(self, embd_layer, max_seq_len, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, nb_classes):
        # learn_mode = 'join'
        # test_mode = 'viterbi'
        # learn_mode = 'marginal'
        # test_mode = 'marginal'
        # self.no_crf = no_crf

        x = Input(shape=(max_seq_len,), name='window-input')
        y = embd_layer(x)
        y = SpatialDropout1D(0.2)(y)
        if not no_rnn:
            y = Bidirectional(LSTM(rnn_dim, return_sequences=True, name='hidden-rnn'))(y)
        if not no_dense:
            y = Dense(hidden_dim, activation='relu', name='hidden-dense', kernel_regularizer=l2(0.001))(y)
            y = Dropout(0.2)(y)
        # if no_crf:
        y1 = Dense(nb_classes, activation='softmax', name='args')(y)
        # else:
        #     raise NotImplementedError('CRF')
        # y1 = CRF(nb_classes, test_mode=test_mode, learn_mode=learn_mode, name='args')(y)

        self.x = x
        self.y_args = y1

        self.model = Model(self.x, self.y_args)

    def compile(self, class_weights):
        # loss = {'args': crf_loss}
        # metrics = {'args': crf_marginal_accuracy}

        # if self.no_crf:
        self.model.compile(loss=class_weighted_loss(class_weights), optimizer='adam',
                           metrics=['accuracy'])
        # else:
        #     self.model.compile(optimizer="adam",
        #                        loss=class_weighted_loss(class_weights),
        #                        metrics=[crf_marginal_accuracy])

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size, callbacks):
        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       verbose=1,
                       callbacks=callbacks)

    def predict(self, X, batch_size=64):
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def summary(self, fh=None):
        if fh:
            self.model.summary(print_fn=lambda line: fh.write("{}\n".format(line)))
        else:
            self.model.summary()


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embedding_matrix(vector_path, word_index, emb_dim):
    save_path = "{}.npy".format(vector_path)
    if os.path.exists(save_path):
        return np.load(save_path)
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(vector_path, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    filled = 0
    all_embs = np.stack(list(embeddings_index.values()))
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), (len(word_index), emb_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            filled += 1
    print('Filled {} of {} word vectors.'.format(filled, len(word_index)))
    np.save(save_path, embedding_matrix)
    return embedding_matrix
