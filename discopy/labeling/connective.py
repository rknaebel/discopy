import logging
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List

import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline

from discopy.data.conll16 import get_conll_dataset
from discopy.features import get_compressed_chain, get_pos_features, get_sibling_label
from discopy.utils import single_connectives, multi_connectives_first, multi_connectives, distant_connectives, \
    init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_connective_candidates(sentence):
    chosen_indices = set()
    candidates = []
    sentence = [w.lower() for w, wd in sentence]
    for word_idx, word in enumerate(sentence):
        if word_idx in chosen_indices:
            continue

        for conn in distant_connectives:
            if word == conn[0]:
                try:
                    i = sentence.index(conn[1], word_idx)
                    candidates.append([(word_idx, conn[0]), (i, conn[1])])
                    break
                except ValueError:
                    continue

        if word in multi_connectives_first:
            for multi_conn in multi_connectives:
                if (word_idx + len(multi_conn)) < len(sentence) and all(
                        c == sentence[word_idx + i] for i, c in enumerate(multi_conn)):
                    chosen_indices.update([word_idx + i for i in range(len(multi_conn))])
                    candidates.append([(word_idx + i, c) for i, c in enumerate(multi_conn)])
                    break

        if word in single_connectives:
            chosen_indices.add(word_idx)
            candidates.append([(word_idx, word)])

    return candidates


def get_features(ptree, leaf_index):
    leave_list = ptree.leaves()
    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)[:-1]

    self_category = ptree[lca_loc].label()
    parent_category = ptree[lca_loc].parent().label() if lca_loc else self_category

    left_sibling = get_sibling_label(ptree[lca_loc], 'left')
    right_sibling = get_sibling_label(ptree[lca_loc], 'right')

    labels = {n.label() for n in ptree.subtrees(lambda t: t.height() > 2)}
    bool_vp = 'VP' in labels
    bool_trace = 'T' in labels

    c = ' '.join(leave_list[leaf_index[0]:leaf_index[-1] + 1]).lower()
    prev, prev_conn, prev_pos, prev_pos_conn_pos = get_pos_features(ptree, leaf_index, c, -1)
    next, next_conn, next_pos, next_pos_conn_pos = get_pos_features(ptree, leaf_index, c, 1)
    prev = lemmatizer.lemmatize(prev)
    next = lemmatizer.lemmatize(next)

    r2l = [ptree[lca_loc[:i + 1]].label() for i in range(len(lca_loc))]
    r2lcomp = get_compressed_chain(r2l)

    feat = {'connective': c, 'connectivePOS': self_category,
            'prevWord': prev, 'prevPOSTag': prev_conn, 'prevPOS+cPOS': prev_pos_conn_pos,
            'nextWord': next, 'nextPOSTag': next_pos, 'cPOS+nextPOS': next_pos_conn_pos,
            'root2LeafCompressed': ','.join(r2lcomp), 'root2Leaf': ','.join(r2l),
            'left_sibling': left_sibling, 'right_sibling': right_sibling,
            'parentCategory': parent_category, 'boolVP': bool_vp, 'boolTrace': bool_trace}

    return feat


def group_by_doc_id(pdtb: list) -> Dict[str, list]:
    pdtb_by_doc = defaultdict(list)
    for r in filter(lambda r: r['Type'] == 'Explicit', pdtb):
        pdtb_by_doc[r['DocID']].append(r)
    return pdtb_by_doc


def generate_pdtb_features(pdtb, parses):
    features = []
    pdtb = group_by_doc_id(pdtb)
    for doc_id, doc in parses.items():
        if doc_id not in pdtb:
            continue
        # process document
        for sent_i, sentence in enumerate(doc['sentences']):
            ptree = sentence['parsetree']
            if not ptree:
                continue
            conns_sent = {'-'.join([str(t[4]) for t in r['Connective']['TokenList']]) for r in pdtb[doc_id] if
                          sent_i in [t[3] for t in r['Connective']['TokenList']]}
            for connective_candidate in get_connective_candidates(sentence['words']):
                conn_idxs = [i for i, c in connective_candidate]
                if '-'.join([str(i) for i, c in connective_candidate]) in conns_sent:
                    features.append((get_features(ptree, conn_idxs), 1))
                else:
                    features.append((get_features(ptree, conn_idxs), 0))
    return list(zip(*features))


class ConnectiveClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('variance', VarianceThreshold(threshold=0.0001)),
            ('model',
             SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                           class_weight='balanced', random_state=0))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'connective_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'connective_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)

    def score_on_features(self, X, y):
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Connective")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.score_on_features(X, y)

    def get_connective(self, parsetree, sentence, word_idx) -> (List[str], float):
        candidates = get_connective_candidates(sentence)
        for candidate in candidates:
            conn_idxs = [i for i, c in candidate]
            if word_idx == conn_idxs[0]:
                x = get_features(parsetree, conn_idxs)
                probs = self.model.predict_proba([x])[0]
                if probs.argmax() == 0:
                    return [], probs[0]
                else:
                    return candidate, probs[1]
        return [], 0.0


if __name__ == "__main__":
    logger = init_logger()
    data_path = sys.argv[1]
    parses_train, pdtb_train = get_conll_dataset(data_path, 'en.train', load_trees=True, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(data_path, 'en.dev', load_trees=True, connective_mapping=True)

    clf = ConnectiveClassifier()
    logger.info('Train model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
