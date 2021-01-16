import json
import logging
import os
from typing import List

import click
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import classification_report

from discopy.components.component import Component
from discopy.components.nn.windows import generate_pdtb_features, predict_discourse_windows_for_id, \
    reduce_relation_predictions, extract_windows
from discopy.data.doc import BertDocument
from discopy.data.loaders.conll import load_bert_conll_dataset
from discopy.data.relation import Relation
from discopy.utils import init_logger

logger = logging.getLogger('discopy')


def get_model(max_seq_len, hidden_dim, rnn_dim, nb_classes):
    x = y = tf.keras.layers.Input(shape=(max_seq_len, 768), name='window-input')
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True), name='rnn')(y)
    # y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True), name='hidden-rnn2')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(hidden_dim, activation='relu', name='dense')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(nb_classes, activation='softmax', name='args')(y)
    model = tf.keras.models.Model(x, y)
    optimizer = tf.keras.optimizers.Adam(lr=0.001, amsgrad=True)
    metrics = [
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


class AbstractArgumentExtractor(Component):

    def __init__(self, window_length, hidden_dim, rnn_dim, explicits_only=False, positives_only=False,
                 ckpt_path: str = ''):
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.explicits_only = explicits_only
        self.positives_only = positives_only
        self.fn = ''
        self.checkpoint_path = ckpt_path
        self.model = get_model(self.window_length, self.hidden_dim, self.rnn_dim, 4)
        self.sense_map = {}
        self.callbacks = []
        self.epochs = 25

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
        x_train, y1_train, y2_train = generate_pdtb_features(docs_train, self.window_length, self.sense_map,
                                                             explicits_only=self.positives_only,
                                                             positives_only=self.positives_only)
        x_val, y1_val, y2_val = generate_pdtb_features(docs_val, self.window_length, self.sense_map,
                                                       explicits_only=self.positives_only,
                                                       positives_only=self.positives_only)
        self.model.summary()
        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=2, min_lr=0.00001,
                                                 verbose=2),
            # tf.keras.callbacks.LearningRateScheduler(scheduler),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True,
                                             verbose=1),
        ]
        if self.checkpoint_path:
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path, self.fn + '.ckp'),
                                                   save_best_only=True,
                                                   save_weights_only=True))
            self.callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_path, 'logs.csv')))
        self.model.fit(
            x_train, y1_train,
            validation_data=(x_val, y1_val),
            epochs=self.epochs,
            batch_size=64,
            callbacks=self.callbacks
        )

    def score_on_features(self, x, y):
        y_pred = np.concatenate(self.model.predict(x).argmax(-1))
        y = np.concatenate(y.argmax(-1))
        logger.info("Evaluation: {}".format(self.fn))
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)

    def score(self, docs: List[BertDocument]):
        x, y1, y2 = generate_pdtb_features(docs, self.window_length, self.sense_map, self.positives_only,
                                           self.positives_only)
        self.score_on_features(x, y1)

    def parse(self, doc: BertDocument, relations: List[Relation] = None, **kwargs):
        raise NotImplementedError()


class ExplicitArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim):
        super().__init__(window_length, hidden_dim, rnn_dim, positives_only=False)
        self.fn = 'nea'

    def parse(self, doc: BertDocument, relations: List[Relation] = None,
              batch_size=256, strides=1, max_distance=0.5, **kwargs):
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
        super().__init__(window_length, hidden_dim, rnn_dim, explicits_only=True, positives_only=True)
        self.fn = 'nca'

    def parse(self, doc: BertDocument, relations: List[Relation] = None, **kwargs):
        offset = self.window_length // 2
        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        word_windows = extract_windows(doc_bert, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=256)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        relations = [relations_hat[i] for i in conn_pos]
        return relations

    def get_window_probs(self, tokens):
        offset = self.window_length // 2
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=256)
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
                                       cache_dir=os.path.join(conll_path, 'en.dev.bert.joblib'),
                                       # limit=10,
                                       device='cuda')
    logger.info('Init model')
    # clf = ExplicitArgumentExtractor(window_length=100, hidden_dim=128, rnn_dim=128)
    clf = ConnectiveArgumentExtractor(window_length=100, hidden_dim=128, rnn_dim=128)
    try:
        clf.load('models/nn')
    except FileNotFoundError:
        logger.info('Load train data')
        docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                             cache_dir=os.path.join(conll_path, 'en.train.bert.joblib'),
                                             # limit=30,
                                             device='cuda')
        torch.cuda.empty_cache()
        logger.info('Train model')
        clf.fit(docs_train, docs_val)
        logger.info('Evaluation on TRAIN')
        clf.score(docs_train)
        clf.save('models/nn')
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    # print(docs_val[0].to_json())
    print(clf.parse(docs_val[0], [], ))


if __name__ == "__main__":
    main()
