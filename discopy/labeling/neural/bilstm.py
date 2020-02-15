import logging
import os
from collections import Counter

import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2

logger = logging.getLogger('discopy')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, SpatialDropout1D


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K


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
    return get_class_weights(y, 0)


def class_weighted_loss(class_weights):
    def loss(onehot_labels, logits):
        c_weights = np.array([class_weights[i] for i in range(4)])
        unweighted_losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=[onehot_labels],
                                                                                 logits=[logits])
        weights = tf.reduce_sum(tf.multiply(onehot_labels, c_weights), axis=-1)
        weighted_losses = unweighted_losses * weights  # reduce the result to get your final loss
        total_loss = tf.reduce_mean(weighted_losses)
        return total_loss

    return loss


class ElmoEmbeddingLayer(Layer):
    def __init__(self, vocab, batch_size, max_len, **kwargs):
        self.trainable = True
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_len = max_len
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ipt_shape = input_shape
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        mapping_string = [word for (word, word_idx) in sorted(self.vocab.items(), key=lambda x: x[1])]
        self.table = tf.contrib.lookup.index_to_string_table_from_tensor(mapping_string, default_value="UKN")
        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        K.get_session().run(tf.tables_initializer(name='init_all_tables'))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(
            inputs={
                "tokens": self.table.lookup(tf.cast(x, tf.int64)),
                "sequence_len": self.batch_size * [self.max_len],
            },
            as_dict=True,
            signature='tokens',
        )['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        # return K.not_equal(self.table.lookup(tf.cast(inputs, tf.int64)), '--PAD--')
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1024)


class BiLSTMx:

    def __init__(self, embd_layer, max_seq_len, hidden_dim, rnn_dim, no_rnn, no_dense, nb_classes):
        x = Input(shape=(max_seq_len,), name='window-input')
        y = embd_layer(x)
        y = SpatialDropout1D(0.2)(y)
        y = Dense(hidden_dim, activation='relu')(y)
        if not no_rnn:
            y = Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(rnn_dim, return_sequences=True, name='hidden-rnn1'))(y)
            y = Bidirectional(tf.compat.v1.keras.layers.CuDNNGRU(rnn_dim, return_sequences=True, name='hidden-rnn2'))(y)
        if not no_dense:
            y = Dense(hidden_dim, activation='relu', name='hidden-dense', kernel_regularizer=l2(0.001))(y)
        y = Dropout(0.2)(y)
        y = Dense(nb_classes, activation='softmax', name='args')(y)

        self.x = x
        self.y = y

        self.model = Model(self.x, self.y)

    def compile(self, class_weights):
        optimizer = Adam(lr=0.001, amsgrad=True)
        self.model.compile(loss=class_weighted_loss(class_weights), optimizer=optimizer,
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size, callbacks):
        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       verbose=1,
                       callbacks=callbacks)

    def predict(self, X, batch_size=512):
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def summary(self):
        self.model.summary(print_fn=logger.info)


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
