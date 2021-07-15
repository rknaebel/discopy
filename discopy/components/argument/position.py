import logging
import os
import pickle
from typing import List

import click
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score
from sklearn.pipeline import Pipeline

from discopy.components.component import SubComponent
from discopy.features import get_connective_sentence_position, lca, get_pos_features
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_parsed_conll_dataset

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_features(ptree: nltk.ParentedTree, conn: str, leaf_index: list):
    lca_loc = lca(ptree, leaf_index)
    conn_pos = ptree[lca_loc].label()
    conn_pos_relative = get_connective_sentence_position(leaf_index, ptree)

    prev, prev_conn, prev_pos, prev_pos_conn_pos = get_pos_features(ptree, leaf_index, conn, -1)
    prev2, prev2_conn, prev2_pos, prev2_pos_conn_pos = get_pos_features(ptree, leaf_index, conn, -2)

    prev = lemmatizer.lemmatize(prev)
    prev2 = lemmatizer.lemmatize(prev2)

    feat = {
        'connective': conn, 'connectivePOS': conn_pos, 'cPosition': conn_pos_relative,
        'prevWord': prev, 'prevPOSTag': prev_pos, 'prevWord+c': prev_conn, 'prevPOS+cPOS': prev_pos_conn_pos,
        'prevWord2': prev2, 'prev2Word+c': prev2_conn, 'prev2POSTag': prev2_pos, 'prev2POS+cPOS': prev2_pos_conn_pos
    }

    return feat


def generate_pdtb_features(docs: List[Document]):
    features = []
    for doc in docs:
        for relation in doc.relations:
            if relation.type != 'Explicit':
                continue
            conn_raw = ' '.join(t.surface for t in relation.conn.tokens)
            conn_idxs = [t.local_idx for t in relation.conn.tokens]
            ptree = doc.sentences[relation.conn.get_sentence_idxs()[0]].get_ptree()
            if ptree is None:
                continue

            sents_arg1 = relation.arg1.get_sentence_idxs()
            sents_arg2 = relation.arg2.get_sentence_idxs()
            if not sents_arg1 or not sents_arg2:
                continue

            distance = sents_arg1[-1] - sents_arg2[0]
            if len(sents_arg1) == 1 and len(sents_arg2) == 1 and distance == 0:
                features.append((get_features(ptree, conn_raw, conn_idxs), 'SS'))
            elif distance == -1:
                features.append((get_features(ptree, conn_raw, conn_idxs), 'PS'))

    return list(zip(*features))


class ArgumentPositionClassifier(SubComponent):
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('variance', VarianceThreshold(threshold=0.0001)),
            ('model',
             SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                           class_weight='balanced', random_state=0))
        ])

    def load(self, path: str):
        self.model = pickle.load(open(os.path.join(path, 'position_clf.p'), 'rb'))

    def save(self, path: str):
        pickle.dump(self.model, open(os.path.join(path, 'position_clf.p'), 'wb'))

    def fit(self, docs: List[Document]):
        X, y = generate_pdtb_features(docs)
        self.model.fit(X, y)

    def score_on_features(self, X, y):
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: ArgPos")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, docs: List[Document]):
        X, y = generate_pdtb_features(docs)
        self.score_on_features(X, y)

    def get_argument_position(self, ptree: nltk.ParentedTree, connective: str, leaf_index):
        x = get_features(ptree, connective, leaf_index)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_parsed_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = ArgumentPositionClassifier()
    logger.info('Train model')
    clf.fit(docs_train)
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)


if __name__ == "__main__":
    main()
