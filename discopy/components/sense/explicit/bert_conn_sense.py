import json
import logging
import os
from typing import List, Dict

import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from discopy.components.component import Component
from discopy.components.connective.base import get_connective_candidates
from discopy.evaluate.conll import evaluate_docs, print_results
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_bert_conll_dataset
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')


def get_conn_model(in_size, out_size, hidden_size, hidden_size2=256):
    x = y = tf.keras.layers.Input(shape=(in_size,), name='connective')
    y = tf.keras.layers.Dense(hidden_size, kernel_initializer='lecun_normal', activation='selu')(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Dense(hidden_size2, kernel_initializer='lecun_normal', activation='selu')(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    y = tf.keras.layers.Dense(out_size, activation='softmax')(y)
    model = tf.keras.models.Model(x, y)
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=[
        "accuracy",
    ])
    return model


def get_bert_features(idxs, doc_bert, used_context=0):
    idxs = list(idxs)
    pad = np.zeros_like(doc_bert[0])
    embd = doc_bert[idxs].mean(axis=0)
    if used_context > 0:
        left = [doc_bert[i] if i >= 0 else pad for i in range(min(idxs) - used_context, min(idxs))]
        right = [doc_bert[i] if i < len(doc_bert) else pad for i in range(max(idxs) + 1, max(idxs) + 1 + used_context)]
        embd = np.concatenate(left + [embd] + right).flatten()
    return embd


def generate_pdtb_features(docs: List[Document], sense_map: Dict[str, int], used_context=0):
    features = []
    for doc in tqdm(docs):
        doc_bert = doc.get_embeddings()
        global_id_map = {(s_i, t.local_idx): t.idx for s_i, s in enumerate(doc.sentences) for t in s.tokens}
        conns = {tuple(t.idx for t in r.conn.tokens): r.senses[0] for r in doc.get_explicit_relations()}
        for sent_i, sentence in enumerate(doc.sentences):
            for connective_candidate in get_connective_candidates(sentence):
                conn_idxs = tuple(global_id_map[(sent_i, i)] for i, c in connective_candidate)
                if conn_idxs in conns:
                    sense = sense_map.get(conns[conn_idxs])
                    if not sense:
                        continue
                    features.append((get_bert_features(conn_idxs, doc_bert, used_context), sense))
                else:
                    features.append((get_bert_features(conn_idxs, doc_bert, used_context), 0))
    x, y = list(zip(*features))
    return np.stack(x), np.array(y)


def get_sense_mapping(docs):
    sense_map = {
        'NoSense': 0,
    }
    senses = sorted({s for doc in docs for rel in doc.relations for s in rel.senses})
    i = 1
    for s in senses:
        if s in sense_map:
            sense_map[s] = sense_map[s]
        else:
            sense_map[s] = i
            i += 1
    classes = []
    for sense, sense_id in sorted(sense_map.items(), key=lambda x: x[1]):
        if len(classes) > sense_id:
            continue
        classes.append(sense)
    return sense_map, classes


class ConnectiveSenseClassifier(Component):
    model_name = 'explicit_sense_bert_classifier'
    used_features = ['vectors']

    def __init__(self, input_dim, used_context: int = 0, hidden_dim: int = 2048):
        self.input_dim = input_dim
        self.used_context = used_context
        self.in_size = input_dim + 2 * used_context * input_dim
        self.hidden_dim = hidden_dim
        self.sense_map = {}
        self.classes = []
        self.model = None
        self.batch_size = 512

    def get_config(self):
        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'used_context': self.used_context,
            'sense_map': self.sense_map,
            'classes': self.classes,
        }

    @staticmethod
    def from_config(config: dict):
        clf = ConnectiveSenseClassifier(config['input_dim'], config['used_context'], config['hidden_dim'])
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
        self.model = get_conn_model(self.in_size, len(self.sense_map), self.hidden_dim, 128)
        self.model.summary()
        print(self.sense_map, self.classes)
        x_train, y_train = generate_pdtb_features(docs_train, self.sense_map, used_context=self.used_context)
        x_val, y_val = generate_pdtb_features(docs_val, self.sense_map, used_context=self.used_context)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, shuffle=True, epochs=20,
                       batch_size=self.batch_size,
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=7, verbose=0,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, verbose=0)
                       ])

    def score_on_features(self, x, y):
        y_pred = self.model.predict(x, batch_size=self.batch_size).argmax(-1)
        logger.info("Evaluation: Connective")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

    def score(self, docs: List[Document]):
        if not self.model:
            raise ValueError("Score of untrained model.")
        x, y = generate_pdtb_features(docs, self.sense_map, used_context=self.used_context)
        self.score_on_features(x, y)

    def parse(self, doc: Document, relations=None, **kwargs):
        if not self.model:
            raise ValueError("Score of untrained model.")
        relations: List[Relation] = []
        doc_bert = doc.get_embeddings()
        global_id_map = {(s_i, t.local_idx): t.idx for s_i, s in enumerate(doc.sentences) for t in s.tokens}
        for sent_i, sent in enumerate(doc.sentences):
            for connective_candidate in get_connective_candidates(sent):
                conn_idxs = tuple(global_id_map[(sent_i, i)] for i, c in connective_candidate)
                features = get_bert_features(conn_idxs, doc_bert, self.used_context)
                pred = self.model.predict(np.expand_dims(features, axis=0)).argmax(-1).flatten()[0]
                if pred > 0:
                    conn_tokens = [sent.tokens[i] for i, c in connective_candidate]
                    relations.append(Relation(
                        conn=conn_tokens,
                        type='Explicit',
                        senses=[self.classes[pred]]
                    ))
        return relations


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
@click.option('--simple-connectives', is_flag=True)
@click.option('-s', '--sense-lvl', default=2, type=int)
@click.option('--conn-length', default=2, type=int)
@click.option('--conn-hidden-dim', default=2048, type=int)
def main(bert_model, conll_path, simple_connectives, sense_lvl, conn_length, conn_hidden_dim):
    logger = init_logger()
    docs_val = load_bert_conll_dataset(os.path.join(conll_path, 'en.dev'),
                                       simple_connectives=simple_connectives,
                                       cache_dir=os.path.join(conll_path, f'en.dev.{bert_model}.joblib'),
                                       bert_model=bert_model,
                                       sense_level=sense_lvl)
    docs_test = load_bert_conll_dataset(os.path.join(conll_path, 'en.test'),
                                        simple_connectives=simple_connectives,
                                        cache_dir=os.path.join(conll_path, f'en.test.{bert_model}.joblib'),
                                        bert_model=bert_model,
                                        sense_level=sense_lvl)
    docs_blind = load_bert_conll_dataset(os.path.join(conll_path, 'en.blind-test'),
                                         simple_connectives=simple_connectives,
                                         cache_dir=os.path.join(conll_path, f'en.blind-test.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_lvl)
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         simple_connectives=simple_connectives,
                                         cache_dir=os.path.join(conll_path, f'en.train.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_lvl)
    clf = ConnectiveSenseClassifier(input_dim=docs_val[0].get_embedding_dim(), used_context=conn_length,
                                    hidden_dim=conn_hidden_dim)
    logger.info('Train model')
    clf.fit(docs_train, docs_val)
    for title, docs_eval in [('TEST', docs_test), ('BLIND', docs_blind)]:
        logger.info(f'Evaluate parser {title}')
        preds = [d.with_relations(clf.parse(d)) for d in docs_eval]
        for threshold in [0.7, 0.95]:
            res_explicit = evaluate_docs(
                [d.with_relations(d.get_explicit_relations()) for d in docs_eval],
                preds,
                threshold=threshold)
            print_results(res_explicit, title=f'{title}-EXPLICIT-{threshold}')


if __name__ == "__main__":
    main()
