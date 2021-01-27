import json
import logging
import os
from typing import List

import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from discopy.components.component import Component
from discopy.components.nn.windows import predict_discourse_windows_for_id, \
    reduce_relation_predictions, extract_windows, PDTBWindowSequence
from discopy.data.doc import BertDocument
from discopy.data.loaders.conll import load_bert_conll_dataset
from discopy.data.relation import Relation
from discopy.utils import init_logger

logger = logging.getLogger('discopy')


class SkMetrics(tf.keras.callbacks.Callback):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.concatenate(self.model.predict(self.ds).argmax(-1))
        y = np.concatenate([args for windows, args in self.ds])
        y = np.concatenate(y.argmax(-1))
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report EPOCH {}".format(epoch))
        for line in report.split('\n'):
            logger.info(line)


def get_model(max_seq_len, hidden_dim, rnn_dim, nb_classes):
    x = y = tf.keras.layers.Input(shape=(max_seq_len, 768), name='window-input')
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True), name='rnn')(y)
    # y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True), name='hidden-rnn2')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(hidden_dim, activation='relu', name='dense')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(nb_classes, activation='softmax', name='args')(y)
    model = tf.keras.models.Model(x, y)
    return model


class AbstractArgumentExtractor(Component):

    def __init__(self, window_length, hidden_dim, rnn_dim, explicits_only=False, positives_only=False,
                 fn: str = '', ckpt_path: str = ''):
        super().__init__(used_features=['vectors'])
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.explicits_only = explicits_only
        self.positives_only = positives_only
        self.fn = fn
        self.checkpoint_path = ckpt_path
        self.model = get_model(self.window_length, self.hidden_dim, self.rnn_dim, 4)
        self.compiled = False
        self.sense_map = {}
        self.callbacks = []
        self.epochs = 10
        self.batch_size = 256
        self.metrics = [
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="ROC"),
        ]
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001, amsgrad=True)

    def get_loss(self, precomputed_weights=None):
        def loss(onehot_labels, logits):
            class_weights = precomputed_weights or [1.0, 9.0, 9.5, 110.0]
            c_weights = np.array([class_weights[i] for i in range(4)])
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
            weights = tf.reduce_sum(tf.multiply(tf.cast(onehot_labels, tf.float32), c_weights), axis=-1)
            weighted_losses = unweighted_losses * weights  # reduce the result to get your final loss
            total_loss = tf.reduce_mean(weighted_losses)
            return total_loss

        return loss

    def load(self, path):
        self.sense_map = json.load(open(os.path.join(path, 'senses.json'), 'r'))
        if not os.path.exists(os.path.join(path, self.fn)):
            raise FileNotFoundError("Model not found.")
        self.model = tf.keras.models.load_model(os.path.join(path, self.fn))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, self.fn))
        json.dump(self.sense_map, open(os.path.join(path, 'senses.json'), 'w'))

    def fit(self, docs_train: List[BertDocument], docs_val: List[BertDocument] = None):
        self.sense_map = {v: k for k, v in enumerate(
            ['NoSense'] + sorted({s for doc in docs_train for rel in doc.relations for s in rel.senses}))}
        ds_train = PDTBWindowSequence(docs_train, self.window_length, self.sense_map, batch_size=self.batch_size,
                                      explicits_only=self.explicits_only,
                                      positives_only=self.positives_only)
        ds_val = PDTBWindowSequence(docs_val, self.window_length, self.sense_map, batch_size=self.batch_size,
                                    explicits_only=self.explicits_only,
                                    positives_only=self.positives_only)
        print("train", ds_train.get_balanced_class_weights())
        print("val", ds_val.get_balanced_class_weights())

        if not self.compiled:
            self.model.compile(loss=self.get_loss(ds_train.get_balanced_class_weights()),
                               optimizer=self.optimizer,
                               metrics=self.metrics)
            self.compiled = True
        self.model.summary()
        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=1, min_lr=0.00001,
                                                 verbose=2),
            # tf.keras.callbacks.LearningRateScheduler(scheduler),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True,
                                             verbose=1),
            SkMetrics(ds_val),
        ]
        if self.checkpoint_path:
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path, self.fn + '.ckp'),
                                                   save_best_only=True,
                                                   save_weights_only=True))
            self.callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_path, 'logs.csv')))
        self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=self.epochs,
            callbacks=self.callbacks,
            max_queue_size=10,
        )

    def score(self, docs: List[BertDocument]):
        ds = PDTBWindowSequence(docs, self.window_length, self.sense_map, batch_size=self.batch_size,
                                explicits_only=self.explicits_only,
                                positives_only=self.positives_only)
        y_pred = np.concatenate(self.model.predict(ds).argmax(-1))
        y = np.concatenate([args for windows, args in ds])
        y = np.concatenate(y.argmax(-1))
        logger.info("Evaluation: {}".format(self.fn))
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)

    def parse(self, doc: BertDocument, relations: List[Relation] = None, **kwargs):
        raise NotImplementedError()


class ExplicitArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim):
        super().__init__(window_length, hidden_dim, rnn_dim, fn='nea', explicits_only=True, positives_only=False)

    def parse(self, doc: BertDocument, relations: List[Relation] = None,
              batch_size=64, strides=1, max_distance=0.5, **kwargs):
        offset = self.window_length // 2
        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()
        windows = extract_windows(doc_bert, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=batch_size)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


class FullArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim):
        super().__init__(window_length, hidden_dim, rnn_dim, fn='naa', explicits_only=False, positives_only=False)

    def parse(self, doc: BertDocument, relations: List[Relation] = None,
              batch_size=64, strides=1, max_distance=0.5, **kwargs):
        offset = self.window_length // 2
        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()
        windows = extract_windows(doc_bert, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=batch_size)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


class ConnectiveArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim):
        super().__init__(window_length, hidden_dim, rnn_dim, fn='nca', explicits_only=True, positives_only=True)

    def parse(self, doc: BertDocument, relations: List[Relation] = None, **kwargs):
        offset = self.window_length // 2
        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()
        conn_pos = [min([i.idx for i in r.conn.tokens]) for r in relations]
        word_windows = extract_windows(doc_bert, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=128)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        relations = [relations_hat[i] for i in conn_pos]
        return relations

    def get_window_probs(self, tokens):
        offset = self.window_length // 2
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=64)
        return y_hat

    def get_relations_for_window_probs(self, y_hat, tokens, relations):
        offset = self.window_length // 2
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        probs = y_hat.max(-1).mean(-1)[conn_pos]
        relations = [relations_hat[i] for i in conn_pos]
        return relations, probs


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    logger.info('Load dev data')
    docs_val = load_bert_conll_dataset(os.path.join(conll_path, 'en.dev'),
                                       cache_dir=os.path.join(conll_path, 'en.dev.bert.joblib'))
    logger.info('Init model')
    # clf = ExplicitArgumentExtractor(window_length=150, hidden_dim=64, rnn_dim=128)
    clf = ConnectiveArgumentExtractor(window_length=100, hidden_dim=64, rnn_dim=512)
    # try:
    #     clf.load('models/nn')
    # except FileNotFoundError:
    logger.info('Load train data')
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         cache_dir=os.path.join(conll_path, 'en.train.bert.joblib'))
    logger.info('Train model')
    clf.fit(docs_train, docs_val)
    # logger.info('Evaluation on TRAIN')
    # clf.score(docs_train)
    clf.save('models/nn')
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    # print(docs_val[0].to_json())
    print(clf.parse(docs_val[0], docs_val[0].get_explicit_relations()))


if __name__ == "__main__":
    main()
