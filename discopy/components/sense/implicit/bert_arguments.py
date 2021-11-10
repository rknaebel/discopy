import json
import logging
import math
import os
from typing import List

import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score

from discopy.components.component import Component
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_bert_conll_dataset
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')


def get_model(arg_length, embdding_dim, out_size, hidden_rnn_size=512, hidden_dense_size=128):
    arg_lstm = tf.keras.layers.LSTM(hidden_rnn_size, return_sequences=True)
    x1 = tf.keras.layers.Input(shape=(arg_length, embdding_dim), name='arg1')
    y1 = arg_lstm(x1)
    y1 = tf.keras.layers.GlobalMaxPooling1D()(y1)
    x2 = tf.keras.layers.Input(shape=(arg_length, embdding_dim), name='arg2')
    y2 = arg_lstm(x2)
    y2 = tf.keras.layers.GlobalMaxPooling1D()(y2)
    y = tf.keras.layers.Concatenate()([y1, y2])
    y = tf.keras.layers.Dropout(0.25)(y)
    y = tf.keras.layers.Dense(hidden_dense_size, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dense(out_size, activation='softmax')(y)
    model = tf.keras.models.Model([x1, x2], y)
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=[
        "accuracy",
    ])
    return model


def get_sense_mapping(docs):
    sense_map = {
        'NoRel': 0,
        'EntRel': 1,
    }
    senses = sorted({s for doc in docs for rel in doc.relations for s in rel.senses})
    for s in senses:
        if s not in sense_map:
            sense_map[s] = len(sense_map)
    classes = []
    for sense, sense_id in sorted(sense_map.items(), key=lambda x: x[1]):
        if len(classes) > sense_id:
            continue
        classes.append(sense)
    return sense_map, classes


class PDTBSentencesSequence(tf.keras.utils.Sequence):
    def __init__(self, docs: List[Document], argument_length: int, sense_map, batch_size: int):
        self.rng = np.random.default_rng()
        self.argument_length = argument_length
        self.relations = []
        for doc in docs:
            for rel in doc.relations:
                if rel.is_explicit() or rel.type == 'AltLex':
                    continue
                if rel.type == 'EntRel':
                    sense = 'EntRel'
                else:
                    sense = rel.senses[0]
                self.relations.append((doc, rel, sense_map[sense]))

        self.instances = np.arange(len(self.relations))
        self.rng.shuffle(self.instances)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.instances) / self.batch_size)

    def __getitem__(self, idx):
        idxs = self.instances[idx * self.batch_size:(idx + 1) * self.batch_size]
        features = []
        for i in idxs:
            doc, rel, sense = self.relations[i]
            features.append((*get_features(rel, doc, self.argument_length), sense))
        x1, x2, y = list(zip(*features))
        return (np.stack(x1), np.stack(x2)), np.array(y)

    def on_epoch_end(self):
        self.rng.shuffle(self.instances)


def get_features(rel: Relation, doc: Document, length: int):
    arg1_idxs = [t.idx for t in rel.arg1.tokens]
    arg2_idxs = [t.idx for t in rel.arg2.tokens]
    s1 = doc.get_embeddings()[min(arg1_idxs):max(arg1_idxs) + 1]
    s2 = doc.get_embeddings()[min(arg2_idxs):max(arg2_idxs) + 1]
    e1 = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(s1, axis=0), length,
                                                       padding='pre', truncating="pre")
    e2 = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(s2, axis=0), length,
                                                       padding='post', truncating="post")
    return e1[0], e2[0]


def generate_pdtb_features(docs: List[Document], sense_map, argument_length=50):
    features = []
    for doc in docs:
        for rel in doc.relations:
            if rel.is_explicit() or rel.type == 'AltLex':
                continue
            if rel.type == 'EntRel':
                sense = 'EntRel'
            else:
                sense = rel.senses[0]
            features.append((*get_features(rel, doc, argument_length), sense_map[sense]))
    x1, x2, y = list(zip(*features))
    return (np.stack(x1), np.stack(x2)), np.array(y)


class ArgumentSenseClassifier(Component):
    model_name = 'non_explicit_argument_sense_classifier'
    used_features = ['vectors']

    def __init__(self, input_dim, arg_length: int = 50, hidden_dim: int = 512, rnn_dim: int = 256):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.arg_length = arg_length
        self.sense_map = {}
        self.classes = []
        self.model = None
        self.batch_size = 512

    def get_config(self):
        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'rnn_dim': self.rnn_dim,
            'arg_length': self.arg_length,
            'sense_map': self.sense_map,
            'classes': self.classes,
        }

    @staticmethod
    def from_config(config: dict):
        clf = ArgumentSenseClassifier(config['input_dim'], config['arg_length'], config['hidden_dim'],
                                      config['rnn_dim'])
        clf.sense_map = config['sense_map']
        clf.classes = config['classes']
        return clf

    def load(self, path):
        self.sense_map = json.load(open(os.path.join(path, self.model_name, 'senses.json'), 'r'))
        self.classes = []
        for sense, sense_id in sorted(self.sense_map.items(), key=lambda x: x[1]):
            if len(self.classes) > sense_id:
                continue
            self.classes.append(sense)
        if not os.path.exists(os.path.join(path, self.model_name)):
            raise FileNotFoundError("Model not found.")
        self.model = tf.keras.models.load_model(os.path.join(path, self.model_name),
                                                compile=False)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, self.model_name))
        json.dump(self.sense_map, open(os.path.join(path, self.model_name, 'senses.json'), 'w'))

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        if docs_val is None:
            raise ValueError("Validation data is missing.")
        self.sense_map, self.classes = get_sense_mapping(docs_train)
        self.model = get_model(self.arg_length, self.input_dim, len(self.sense_map),
                               hidden_rnn_size=self.rnn_dim, hidden_dense_size=self.hidden_dim)
        self.model.summary()
        # x_train, y_train = generate_pdtb_features(docs_train, self.sense_map, self.arg_length)
        # x_val, y_val = generate_pdtb_features(docs_val, self.sense_map, self.arg_length)
        ds_train = PDTBSentencesSequence(docs_train, self.arg_length, self.sense_map, self.batch_size)
        ds_val = PDTBSentencesSequence(docs_val, self.arg_length, self.sense_map, self.batch_size)
        self.model.fit(ds_train,
                       validation_data=ds_val,
                       epochs=15,
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=7, verbose=0,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, verbose=1)
                       ],
                       max_queue_size=10,
                       workers=3,
                       )

    def score_on_features(self, x, y):
        y_pred = self.model.predict(x).argmax(-1)
        logger.info("Evaluation:")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

    def score(self, docs: List[Document]):
        if not self.model:
            raise ValueError("Score of untrained model.")
        x, y = generate_pdtb_features(docs, self.sense_map, self.arg_length)
        self.score_on_features(x, y)

    def parse(self, doc: Document, relations: List[Relation] = None, **kwargs):
        if len(doc.sentences) <= 1:
            return []
        if relations is None:
            raise ValueError()
        features = []
        rels = [r for r in relations if not r.type or r.type == 'Implicit']
        if rels:
            for r in rels:
                features.append(get_features(r, doc, self.arg_length))
            x1, x2 = list(zip(*features))
            x1, x2 = np.stack(x1), np.stack(x2)
            preds = self.model.predict([x1, x2], batch_size=self.batch_size).argmax(-1).flatten()
            for rel, pred in zip(rels, preds):
                sense = self.classes[pred]
                if sense == 'NoRel':
                    continue
                elif sense == 'EntRel':
                    rel_type = 'EntRel'
                else:
                    rel_type = 'Implicit'
                rel.type = rel_type
                rel.senses = [sense]
        return relations


@click.command()
@click.argument('conll-path')
@click.argument('save-path', type=str)
@click.argument('bert_model', type=str)
@click.option('-s', '--sense-lvl', default=2, type=int)
@click.option('-l', '--argument-length', default=50, type=int)
def main(conll_path, bert_model, save_path, sense_lvl, argument_length):
    logger = init_logger()
    docs_val = load_bert_conll_dataset(os.path.join(conll_path, 'en.dev'),
                                       cache_dir=os.path.join(conll_path, f'en.dev.{bert_model}.joblib'),
                                       bert_model=bert_model,
                                       sense_level=sense_lvl)
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         cache_dir=os.path.join(conll_path, f'en.train.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_lvl)
    clf = ArgumentSenseClassifier(input_dim=docs_val[0].get_embedding_dim(), arg_length=argument_length)
    logger.info('Train model')
    clf.fit(docs_train, docs_val)
    clf.save(save_path)
    json.dump(clf.get_config(), open(os.path.join(save_path, 'config.json'), 'w'))
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)


if __name__ == "__main__":
    main()
