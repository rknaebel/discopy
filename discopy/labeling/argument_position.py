import logging
import os
import pickle
import sys

import nltk
from nltk.tree import ParentedTree
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score
from sklearn.pipeline import Pipeline

from discopy.data.conll16 import get_conll_dataset
from discopy.features import get_connective_sentence_position, lca, get_pos_features
from discopy.utils import init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_features(ptree: ParentedTree, conn: str, leaf_index: list):
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


def generate_pdtb_features(pdtb, parses):
    features = []
    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc = relation['DocID']
        connective = relation['Connective']['TokenList']
        connective_raw = relation['Connective']['RawText']
        leaf_indices = [token[4] for token in connective]
        ptree = parses[doc]['sentences'][connective[0][3]]['parsetree']
        if not ptree:
            continue

        arg1 = sorted({i[3] for i in relation['Arg1']['TokenList']})
        arg2 = sorted({i[3] for i in relation['Arg2']['TokenList']})
        if not arg1 or not arg2:
            continue

        distance = arg1[-1] - arg2[0]
        if len(arg1) == 1 and len(arg2) == 1 and distance == 0:
            features.append((get_features(ptree, connective_raw, leaf_indices), 'SS'))
        elif distance == -1:
            features.append((get_features(ptree, connective_raw, leaf_indices), 'PS'))

    return list(zip(*features))


class ArgumentPositionClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('variance', VarianceThreshold(threshold=0.0001)),
            ('model',
             SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                           class_weight='balanced', random_state=0))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'position_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'position_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)

    def score_on_features(self, X, y):
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: ArgPos")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.score_on_features(X, y)

    def get_argument_position(self, parse, connective: str, leaf_index):
        x = get_features(parse, connective, leaf_index)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


if __name__ == "__main__":
    logger = init_logger()

    data_path = sys.argv[1]
    parses_train, pdtb_train = get_conll_dataset(data_path, 'en.train', load_trees=True, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(data_path, 'en.dev', load_trees=True, connective_mapping=True)

    clf = ArgumentPositionClassifier()
    logger.info('Train model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
