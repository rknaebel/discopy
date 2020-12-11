import copy
import logging
import os
import pickle
from typing import List

import click
import nltk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn_crfsuite import CRF

from discopy.components.component import Component
from discopy.data.doc import Document, DepRel
from discopy.data.relation import Relation
from discopy.data.loaders.conll import load_conll_dataset
from discopy.features import get_compressed_chain
from discopy.utils import init_logger, encode_iob, decode_iob

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')


def get_features(ptree: nltk.ParentedTree, dtree: List[DepRel], indices, sense, offset):
    features_sentence = []
    for i, drel in enumerate(dtree):
        if drel.rel.lower() == 'root':
            mv_position = i
            break
    else:
        mv_position = 0
    main_verb = lemmatizer.lemmatize(ptree.pos()[mv_position][0])

    for i, (word, tag) in enumerate(ptree.pos()):
        tree_pos = ptree.treeposition_spanning_leaves(i, i + 1)[:-2]
        chain = [ptree[tree_pos[:i + 1]].label() for i in range(len(tree_pos))]
        chain = ['S' if c == 'SBAR' else c for c in chain]
        if len(chain) > 0:
            chain = "-".join(get_compressed_chain(chain))
        stem = stemmer.stem(word).lower()

        features_word = {
            'idx': offset + i,
            'BOS': i == 0,
            'word': word.lower(),
            'pos': tag,
            'lemma': lemmatizer.lemmatize(word).lower(),
            'stem': stem.lower(),
            'chain': chain,
            'conn': sense.split('.')[0] if offset + i in indices else "",
            'inflection': word[len(stem):],
            'is_main_verb': i == mv_position,
            'main_verb': main_verb.lower()
        }
        features_sentence.append(features_word)
    return features_sentence


def generate_pdtb_features(docs: List[Document], window_side_size=2):
    arg1_features = []
    arg2_features = []

    for doc in docs:
        for relation in filter(lambda r: r.type == 'Explicit', doc.relations):
            arg1 = set(t.local_idx for t in relation.arg1.tokens)
            arg2 = set(t.local_idx for t in relation.arg2.tokens)
            conn = [t.local_idx for t in relation.conn.tokens]

            arg2_sentence_id = relation.arg2.get_sentence_idxs()[0]
            sent_features = []

            for i in range(-window_side_size, window_side_size + 1):
                sent_i = arg2_sentence_id + i
                if sent_i < 0 or sent_i >= len(doc.sentences):
                    continue
                sent_i = doc.sentences[sent_i]
                ptree_i = sent_i.get_ptree()
                if not ptree_i:
                    continue
                dtree_i = sent_i.dependencies
                sent_features.extend(
                    get_features(ptree_i, dtree_i, conn, relation.senses[0], sent_i.tokens[0].local_idx))

            X_arg1 = copy.deepcopy(sent_features)
            X_arg2 = copy.deepcopy(sent_features)
            labels_arg1 = []
            labels_arg2 = []
            for i, w_arg1 in enumerate(X_arg1):
                if w_arg1['idx'] in arg1:
                    labels_arg1.append("Arg1")
                    labels_arg2.append("NULL")
                    w_arg1['is_arg2'] = False
                elif w_arg1['idx'] in arg2:
                    labels_arg1.append("NULL")
                    labels_arg2.append("Arg2")
                    w_arg1['is_arg2'] = True
                else:
                    labels_arg1.append("NULL")
                    labels_arg2.append("NULL")
                    w_arg1['is_arg2'] = False
                del X_arg1[i]['idx']
                del X_arg2[i]['idx']
            labels_arg1 = encode_iob(labels_arg1)
            labels_arg2 = encode_iob(labels_arg2)
            arg1_features.append((X_arg1, labels_arg1))
            arg2_features.append((X_arg2, labels_arg2))
    return list(zip(*arg1_features)), list(zip(*arg2_features))


class GoshArgumentExtractor(Component):
    def __init__(self, window_side_size=2):
        self.id = 'gosh_arg_extract'
        self.window_side_size = window_side_size
        self.arg1_model = CRF(algorithm='l2sgd', verbose=True, min_freq=5)
        self.arg2_model = CRF(algorithm='l2sgd', verbose=True, min_freq=5)

    def load(self, path):
        self.arg1_model = pickle.load(open(os.path.join(path, "{}.arg1.p".format(self.id)), 'rb'))
        self.arg2_model = pickle.load(open(os.path.join(path, "{}.arg2.p".format(self.id)), 'rb'))

    def save(self, path):
        pickle.dump(self.arg1_model, open(os.path.join(path, "{}.arg1.p".format(self.id)), 'wb'))
        pickle.dump(self.arg2_model, open(os.path.join(path, "{}.arg2.p".format(self.id)), 'wb'))

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        (x_arg1, y_arg1), (x_arg2, y_arg2) = generate_pdtb_features(docs_train, self.window_side_size)
        self.arg1_model.fit(x_arg1, y_arg1)
        self.arg2_model.fit(x_arg2, y_arg2)

    def score_on_features(self, x_arg1, y_arg1, x_arg2, y_arg2):
        y_pred = self.arg1_model.predict(x_arg1)
        y_pred = [decode_iob(s) for s in y_pred]
        y_arg1 = [decode_iob(s) for s in y_arg1]
        y_pred = np.concatenate(y_pred)
        y_arg1 = np.concatenate(y_arg1)
        logger.info("Evaluation: Arg1 Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_arg1, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_arg1, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_arg1, y_pred)))

        y_pred = self.arg2_model.predict(x_arg2)
        y_pred = [decode_iob(s) for s in y_pred]
        y_arg2 = [decode_iob(s) for s in y_arg2]
        y_pred = np.concatenate(y_pred)
        y_arg2 = np.concatenate(y_arg2)
        logger.info("Evaluation: Arg2 Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_arg2, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_arg2, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_arg2, y_pred)))

    def score(self, docs: List[Document]):
        (x_arg1, y_arg1), (x_arg2, y_arg2) = generate_pdtb_features(docs, self.window_side_size)
        self.score_on_features(x_arg1, y_arg1, x_arg2, y_arg2)

    def extract_arguments(self, doc: Document, relation: Relation):
        conn = [t.local_idx for t in relation.conn.tokens]
        arg2_sentence_id = relation.arg2.get_sentence_idxs()[0]
        sent_features = []

        for i in range(-self.window_side_size, self.window_side_size + 1):
            sent_i = arg2_sentence_id + i
            if sent_i < 0 or sent_i >= len(doc.sentences):
                continue
            sent_i = doc.sentences[sent_i]
            ptree_i = sent_i.get_ptree()
            if not ptree_i:
                continue
            dtree_i = sent_i.dependencies
            sent_features.extend(
                get_features(ptree_i, dtree_i, conn, relation.senses[0], sent_i.tokens[0].local_idx))

        indices = []
        for i in sent_features:
            indices.append(i['idx'])
            del i['idx']
        indices = np.array(indices)

        arg2_probs = np.array(
            [[p[c] for c in self.arg2_model.classes_] for p in self.arg2_model.predict_marginals_single(sent_features)])
        arg2_probs_max = arg2_probs.max(1)
        arg2_labels = np.array(self.arg2_model.classes_)[arg2_probs.argmax(axis=1)]
        arg2_labels = np.array(decode_iob(arg2_labels))

        for i, arg2_label in zip(sent_features, arg2_labels):
            i['is_arg2'] = (arg2_label == 'Arg2')

        arg1_probs = np.array(
            [[p[c] for c in self.arg1_model.classes_] for p in self.arg1_model.predict_marginals_single(sent_features)])
        arg1_probs_max = arg1_probs.max(1)
        arg1_labels = np.array(self.arg1_model.classes_)[arg1_probs.argmax(axis=1)]
        arg1_labels = np.array(decode_iob(arg1_labels))

        arg1 = indices[np.where(arg1_labels == 'Arg1')[0]]
        arg2 = indices[np.where(arg2_labels == 'Arg2')[0]]

        arg1_prob = arg1_probs_max[np.where(arg1_labels == 'Arg1')[0]].mean() if len(arg1) else 0.0
        arg2_prob = arg2_probs_max[np.where(arg2_labels == 'Arg2')[0]].mean() if len(arg2) else 0.0

        arg1, arg2 = arg1.tolist(), arg2.tolist()

        return arg1, arg2, arg1_prob, arg2_prob

    def parse(self, doc: Document, relations: List[Relation] = None):
        if relations is None:
            raise ValueError('Component needs connectives already classified.')
        for relation in filter(lambda r: r.type == "Explicit", relations):
            sent_id = relation.conn.get_sentence_idxs()[0]
            sent = doc.sentences[sent_id]
            ptree = sent.get_ptree()
            if ptree is None or len(relation.conn.tokens) == 0:
                continue
            arg1, arg2, arg1_c, arg2_c = self.extract_arguments(ptree, relation)
            relation.arg1.tokens = [sent.tokens[i] for i in arg1]
            relation.arg2.tokens = [sent.tokens[i] for i in arg2]
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = GoshArgumentExtractor()
    logger.info('Train Gosh model')
    clf.fit(docs_train)
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)


if __name__ == "__main__":
    main()
