import logging
import os
from collections import Counter

import numpy as np
from keras.regularizers import l2
from sklearn.utils import compute_class_weight

logger = logging.getLogger('discopy')

from keras import Input, Model
from keras.layers import Bidirectional, Dense, Dropout, SpatialDropout1D
from keras.layers import CuDNNLSTM as LSTM


import tensorflow as tf


def get_class_weights(y, smooth_factor=0.0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)
    if smooth_factor > 0.0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}


def get_balanced_class_weights(y):
    y = y.argmax(-1).flatten()
    return get_class_weights(y, 0.1)


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

    def __init__(self, embd_layer, max_seq_len, hidden_dim, rnn_dim, no_rnn, no_dense, nb_classes):
        x = Input(shape=(max_seq_len,), name='window-input')
        y = embd_layer(x)
        y = SpatialDropout1D(0.2)(y)
        if not no_rnn:
            y = Bidirectional(LSTM(rnn_dim, return_sequences=True, name='hidden-rnn'))(y)
        if not no_dense:
            y = Dense(hidden_dim, activation='relu', name='hidden-dense', kernel_regularizer=l2(0.001))(y)
            y = Dropout(0.2)(y)
        y = Dense(nb_classes, activation='softmax', name='args')(y)

        self.x = x
        self.y = y

        self.model = Model(self.x, self.y)

    def compile(self, class_weights):
        self.model.compile(loss=class_weighted_loss(class_weights), optimizer='adam',
                           metrics=['accuracy'])

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
    logger.info('Found %s word vectors.' % len(embeddings_index))

    filled = 0
    all_embs = np.stack([e for e in embeddings_index.values() if len(e) == emb_dim])
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), (len(word_index), emb_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            filled += 1
    logger.info('Filled {} of {} word vectors.'.format(filled, len(word_index)))
    np.save(save_path, embedding_matrix)
    return embedding_matrix
