import json
import logging
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
    x1 = tf.keras.layers.Input(shape=(arg_length, embdding_dim), name='arg1')
    y1 = tf.keras.layers.LSTM(hidden_rnn_size, return_sequences=True)(x1)
    y1 = tf.keras.layers.GlobalMaxPooling1D()(y1)
    x2 = tf.keras.layers.Input(shape=(arg_length, embdding_dim), name='arg2')
    y2 = tf.keras.layers.LSTM(hidden_rnn_size, return_sequences=True)(x2)
    y2 = tf.keras.layers.GlobalMaxPooling1D()(y2)
    y = tf.keras.layers.Concatenate()([y1, y2])
    y = tf.keras.layers.Dropout(0.25)(y)
    y = tf.keras.layers.Dense(hidden_dense_size, activation='relu')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dense(out_size, activation='softmax')(y)
    model = tf.keras.models.Model([x1, x2], y)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=[
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


def get_features(sent_idx: int, doc, length: int):
    s1, s2 = doc.sentences[sent_idx:sent_idx + 2]
    e1 = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(s1.get_embeddings(), axis=0), length,
                                                       padding='pre', truncating="pre")
    e2 = tf.keras.preprocessing.sequence.pad_sequences(np.expand_dims(s2.get_embeddings(), axis=0), length,
                                                       padding='post', truncating="post")
    return e1[0], e2[0]


def get_implicit_sentence_relations(doc):
    rels = [(r.arg1.get_sentence_idxs(), r.arg2.get_sentence_idxs(), r) for r in doc.relations if not r.is_explicit()]
    return {s1[0]: r for s1, s2, r in rels if len(s1 + s2) == 2 and (s2[0] - s1[0]) == 1}


def generate_pdtb_features(docs: List[Document], sense_map, argument_length=50):
    features = []
    for doc in docs:
        implicits = get_implicit_sentence_relations(doc)
        for s_i, s1 in enumerate(doc.sentences[:-1]):
            if s_i in implicits:
                relation = implicits[s_i]
                if relation.type == 'EntRel':
                    sense = 'EntRel'
                elif relation.senses[0]:
                    sense = relation.senses[0]
                else:
                    sense = 'NoRel'
            else:
                sense = 'NoRel'
            features.append((*get_features(s_i, doc, argument_length), sense_map[sense]))
    x1, x2, y = list(zip(*features))
    return (np.stack(x1), np.stack(x2)), np.array(y)


class NonExplicitRelationClassifier(Component):
    model_name = 'non_explicit_relation_classifier'
    used_features = ['vectors']

    def __init__(self, input_dim, arg_length: int = 50):
        self.input_dim = input_dim
        self.arg_length = arg_length
        self.sense_map = {}
        self.classes = []
        self.model = None

    def get_config(self):
        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'arg_length': self.arg_length,
            'sense_map': self.sense_map,
            'classes': self.classes,
        }

    @staticmethod
    def from_config(config: dict):
        clf = NonExplicitRelationClassifier(config['input_dim'], config['arg_length'])
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
                               hidden_rnn_size=128, hidden_dense_size=128)
        self.model.summary()
        x_train, y_train = generate_pdtb_features(docs_train, self.sense_map, self.arg_length)
        x_val, y_val = generate_pdtb_features(docs_val, self.sense_map, self.arg_length)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, shuffle=True, epochs=25,
                       batch_size=64,
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
                       ])

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
            relations: List[Relation] = []
        features = []
        for s_i, s1 in enumerate(doc.sentences[:-1]):
            features.append(get_features(s_i, doc, self.arg_length))
        x1, x2 = list(zip(*features))
        x1, x2 = np.stack(x1), np.stack(x2)
        preds = self.model.predict([x1, x2]).argmax(-1).flatten()
        for sent_i, pred in enumerate(preds):
            sense = self.classes[pred]
            if sense == 'NoRel':
                continue
            elif sense == 'EntRel':
                rel_type = 'EntRel'
            else:
                rel_type = 'Implicit'
            relations.append(Relation(
                arg1=doc.sentences[sent_i].tokens,
                arg2=doc.sentences[sent_i + 1].tokens,
                senses=[sense],
                type=rel_type
            ))
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
    clf = NonExplicitRelationClassifier(input_dim=docs_val[0].get_embedding_dim(), arg_length=argument_length)
    logger.info('Train model')
    clf.fit(docs_train, docs_val)
    clf.save(save_path)
    json.dump(clf.get_config(), open(os.path.join(save_path, 'config.json'), 'w'))
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    # logger.info('Parse one document')
    # for doc in docs_val:
    #     for r in clf.parse(doc):
    #         print(r.type, r.senses[0], r.arg1.get_sentence_idxs(), r.arg2.get_sentence_idxs())


if __name__ == "__main__":
    main()
