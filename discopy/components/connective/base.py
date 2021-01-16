import logging
import os
import pickle
from typing import List

import click
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline

from discopy.components.component import Component
from discopy.data.doc import ParsedDocument
from discopy.data.loaders.conll import load_parsed_conll_dataset
from discopy.data.relation import Relation
from discopy.data.sentence import Sentence
from discopy.features import get_compressed_chain, get_pos_features, get_sibling_label
from discopy.utils import single_connectives, multi_connectives_first, multi_connectives, distant_connectives, \
    init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_connective_candidates(sentence: Sentence):
    chosen_indices = set()
    candidates = []
    sentence = [w.surface.lower() for w in sentence.tokens]
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


def get_features(ptree: nltk.ParentedTree, conn_idxs):
    leave_list = ptree.leaves()
    lca_loc = ptree.treeposition_spanning_leaves(conn_idxs[0], conn_idxs[-1] + 1)[:-1]

    self_category = ptree[lca_loc].label()
    parent_category = ptree[lca_loc].parent().label() if lca_loc else self_category

    left_sibling = get_sibling_label(ptree[lca_loc], 'left')
    right_sibling = get_sibling_label(ptree[lca_loc], 'right')

    labels = {n.label() for n in ptree.subtrees(lambda t: t.height() > 2)}
    bool_vp = 'VP' in labels
    bool_trace = 'T' in labels

    c = ' '.join(leave_list[conn_idxs[0]:conn_idxs[-1] + 1]).lower()
    prev, prev_conn, prev_pos, prev_pos_conn_pos = get_pos_features(ptree, conn_idxs, c, -1)
    next, next_conn, next_pos, next_pos_conn_pos = get_pos_features(ptree, conn_idxs, c, 1)
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


def generate_pdtb_features(docs: List[ParsedDocument]):
    features = []
    for doc in docs:
        for sent_i, sentence in enumerate(doc.sentences):
            ptree = sentence.get_ptree()
            if not ptree:
                continue
            conns_sent = {'-'.join([str(t.local_idx) for t in r.conn.tokens]) for r in doc.relations if
                          sent_i in [t.sent_idx for t in r.conn.tokens]}
            for connective_candidate in get_connective_candidates(sentence):
                conn_idxs = [i for i, c in connective_candidate]
                if '-'.join([str(i) for i, c in connective_candidate]) in conns_sent:
                    features.append((get_features(ptree, conn_idxs), 1))
                else:
                    features.append((get_features(ptree, conn_idxs), 0))
    return list(zip(*features))


class ConnectiveClassifier(Component):
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
        if not os.path.exists(path):
            os.makedirs(path)
        pickle.dump(self.model, open(os.path.join(path, 'connective_clf.p'), 'wb'))

    def fit(self, docs_train: List[ParsedDocument], docs_val: List[ParsedDocument] = None):
        x, y = generate_pdtb_features(docs_train)
        self.model.fit(x, y)

    def score_on_features(self, x, y):
        y_pred = self.model.predict_proba(x)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Connective")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, docs):
        x, y = generate_pdtb_features(docs)
        self.score_on_features(x, y)

    def parse(self, doc: ParsedDocument, relations=None, **kwargs):
        relations: List[Relation] = []
        for sent_id, sent in enumerate(doc.sentences):
            ptree = sent.get_ptree()
            if ptree is None:
                logger.warning('Failed on empty tree')
                continue
            for candidate in get_connective_candidates(sent):
                conn_idxs = [i for i, c in candidate]
                x = get_features(ptree, conn_idxs)
                probs = self.model.predict_proba([x])[0]
                if probs.argmax() == 1:
                    conn_tokens = [sent.tokens[i] for i, c in candidate]
                    relations.append(Relation(
                        conn=conn_tokens,
                        type='Explicit'
                    ))
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_parsed_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = ConnectiveClassifier()
    logger.info('Train model')
    clf.fit(docs_train)
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    print(clf.parse(docs_val[0], [], ))


if __name__ == "__main__":
    main()
