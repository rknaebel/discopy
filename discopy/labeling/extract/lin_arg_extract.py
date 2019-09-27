import logging
import os
import pickle
import sys

import nltk
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline

from discopy.data.conll16 import get_conll_dataset
from discopy.features import get_clause_context, get_connective_category, get_relative_position, \
    get_clause_direction_path, lca, get_index_tree
from discopy.features import get_root_path
from discopy.features import get_sibling_counts, get_clauses
from discopy.utils import init_logger

logger = logging.getLogger('discopy')


def simplify_tree(ptree, collapse_root=False):
    ptree._label = 'S'
    tree = nltk.Tree.convert(ptree)

    if not collapse_root and isinstance(tree, nltk.Tree) and len(tree) == 1:
        nodes = [tree[0]]
    else:
        nodes = [tree]

    # depth-first traversal of tree
    while nodes:
        node = nodes.pop()
        if isinstance(node, nltk.Tree):
            if (
                    len(node) == 1
                    and isinstance(node[0], nltk.Tree)
                    and isinstance(node[0, 0], nltk.Tree)
            ):
                if node.label() != node[0].label():
                    node.set_label(node.label() + '+' + node[0].label())
                else:
                    node.set_label(node.label())
                node[0:] = [child for child in node[0]]
                # since we assigned the child's children to the current node,
                # evaluate the current node again
                nodes.append(node)
            else:
                for child in node:
                    nodes.append(child)

    return nltk.ParentedTree.convert(tree)


def get_features(conn_head, indices, ptree):
    features = []
    lca_loc = lca(ptree, indices)
    connective = ptree[lca_loc]

    left_siblings_no, right_siblings_no = get_sibling_counts(ptree[lca_loc])

    conn_cat = get_connective_category(conn_head)

    conn_pos = ptree.treeposition_spanning_leaves(indices[0], max(indices[0] + 1, indices[-1]))[:-1]

    for num, (clause, n_pos) in enumerate(get_clauses(ptree)):
        relative_position = get_relative_position(n_pos, conn_pos)
        clause_context = get_clause_context(clause)

        conn2clause_path = get_clause_direction_path(connective, clause)
        conn2root_path = get_root_path(clause)

        feat = {'connectiveString': conn_head, 'connectivePOS': str(connective.label()), 'leftSibNo': left_siblings_no,
                'rightSibNo': right_siblings_no, 'connCategory': conn_cat, 'clauseRelPosition': relative_position,
                'clauseContext': clause_context, 'conn2clausePath': str(conn2clause_path),
                'conn2rootPath': conn2root_path}
        features.append((feat, n_pos))
    return features


def extract_ss_arguments(conn_head, indices, ptree, arg1, arg2):
    feature_set = []
    ptree_ids = get_index_tree(ptree)
    clause_features = get_features(conn_head, indices, ptree)
    for feature, clause in clause_features:
        clause_indices = set(ptree_ids[clause].leaves())
        if clause_indices.issubset(arg1):
            label = 'Arg1'
        elif clause_indices.issubset(arg2):
            label = 'Arg2'
        else:
            label = 'NULL'
        feature_set.append((feature, label))
    return feature_set


def extract_ps_arguments(conn_head, indices, ptree, arg2):
    """
    If Arg1 is in the previous sentence relative to Arg2, a majority classifier in Lin et al. gave the full previous
    sentence as Arg1, which already gives good results in argument extraction
    """
    feature_set = []
    ptree_ids = get_index_tree(ptree)
    clause_features = get_features(conn_head, indices, ptree)
    for feature, clause in clause_features:
        clause_indices = set(ptree_ids[clause].leaves())
        if clause_indices.issubset(arg2):
            label = 'Arg2'
        else:
            label = 'NULL'
        feature_set.append((feature, label))
    return feature_set


def generate_pdtb_features(pdtb, parses):
    ss_features = []
    ps_features = []

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc_id = relation['DocID']
        conn = relation['Connective']['RawText']
        arg1_sentence_id = relation['Arg1']['TokenList'][0][3]
        arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
        ptree = parses[doc_id]['sentences'][arg2_sentence_id]['parsetree']
        if not ptree:
            continue
        ptree = simplify_tree(ptree)

        arg1 = set([i[4] for i in relation['Arg1']['TokenList']])
        arg2 = set([i[4] for i in relation['Arg2']['TokenList']])

        indices = [token[4] for token in relation['Connective']['TokenList']]

        # Arg1 is in the same sentence (SS)
        if arg1_sentence_id == arg2_sentence_id:
            ss_features.extend(extract_ss_arguments(conn, indices, ptree, arg1, arg2))
        # Arg1 is in the previous sentence (PS)
        elif (arg2_sentence_id - arg1_sentence_id) == 1:
            ps_features.extend(extract_ps_arguments(conn, indices, ptree, arg2))
    return list(zip(*ss_features)), list(zip(*ps_features))


class LinArgumentExtractClassifier:
    def __init__(self):
        self.ss_model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('variance', VarianceThreshold(threshold=0.0001)),
            ('model',
             SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                           class_weight='balanced', random_state=0))
        ])
        self.ps_model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('variance', VarianceThreshold(threshold=0.0001)),
            ('model',
             SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                           class_weight='balanced', random_state=0))
        ])

    def load(self, path):
        self.ss_model = pickle.load(open(os.path.join(path, 'ss_extract_clf.p'), 'rb'))
        self.ps_model = pickle.load(open(os.path.join(path, 'ps_extract_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.ss_model, open(os.path.join(path, 'ss_extract_clf.p'), 'wb'))
        pickle.dump(self.ps_model, open(os.path.join(path, 'ps_extract_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        (X_ss, y_ss), (X_ps, y_ps) = generate_pdtb_features(pdtb, parses)
        self.ss_model.fit(X_ss, y_ss)
        self.ps_model.fit(X_ps, y_ps)

    def score_on_features(self, X_ss, y_ss, X_ps, y_ps):
        y_pred = self.ss_model.predict_proba(X_ss)
        y_pred_c = self.ss_model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: SS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ss, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ss, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ss, y_pred_c)))

        y_pred = self.ps_model.predict_proba(X_ps)
        y_pred_c = self.ps_model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: PS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ps, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ps, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ps, y_pred_c)))

    def score(self, pdtb, parses):
        (X_ss, y_ss), (X_ps, y_ps) = generate_pdtb_features(pdtb, parses)
        self.score_on_features(X_ss, y_ss, X_ps, y_ps)

    def extract_arguments(self, ptree, relation, arg_pos):
        indices = [token[4] for token in relation['Connective']['TokenList']]
        conn_head = relation['Connective']['RawText']
        ptree = simplify_tree(ptree)
        ptree_ids = get_index_tree(ptree)

        fs = sorted(get_features(conn_head, indices, ptree), key=lambda x: x[1])
        if not fs:
            print("Empty feature sequence", conn_head, indices, ptree)
            return [], [], 0.0, 0.0
        X, position = zip(*fs)
        if arg_pos == 'SS':
            probs = self.ss_model.predict_proba(X)

            indices_sorted = np.where(probs.argsort().T[-1] == 1)[0]
            if len(indices_sorted):
                arg2_max_idx = indices_sorted[0]
            else:
                arg2_max_idx = probs[:, 1].argmax()
            arg2_prob = probs[arg2_max_idx, 1]

            probs[arg2_max_idx, :] = 0
            indices_sorted = np.where(probs.argsort().T[-1] == 0)[0]
            if len(indices_sorted):
                arg1_max_idx = indices_sorted[0]
            else:
                arg1_max_idx = probs[:, 0].argmax()
            arg1_prob = probs[arg1_max_idx, 0]

            arg2 = set(ptree_ids[position[arg2_max_idx]].leaves())
            arg1 = set(ptree_ids[position[arg1_max_idx]].leaves())
            if arg1.issubset(arg2):
                logger.debug('arg1 is subset of arg2')
                arg2 = arg2 - arg1
            elif arg1.issuperset(arg2):
                logger.debug('arg1 is superset of arg2')
                arg1 = arg1 - arg2
            if not arg1:
                logger.warning("SS Empty Arg1")
            if not arg2:
                logger.warning("SS Empty Arg2")
        elif arg_pos == 'PS':
            probs = self.ps_model.predict_proba(X)

            indices_sorted = np.where(probs.argsort().T[-1] == 0)[0]
            if len(indices_sorted):
                arg2_max_idx = indices_sorted[0]
            else:
                arg2_max_idx = probs[:, 0].argmax()
            arg2_prob = probs[arg2_max_idx, 0]
            arg1_prob = 1.0

            arg1 = set()
            arg2 = set(ptree_ids[position[arg2_max_idx]].leaves())
            if not arg2:
                logger.warning("PS Empty Arg2")
        else:
            raise NotImplementedError('Unknown argument position')

        return sorted(arg1), sorted(arg2), arg1_prob, arg2_prob


if __name__ == "__main__":
    logger = init_logger()

    data_path = sys.argv[1]
    parses_train, pdtb_train = get_conll_dataset(data_path, 'en.train', load_trees=True, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(data_path, 'en.dev', load_trees=True, connective_mapping=True)

    clf = LinArgumentExtractClassifier()
    logger.info('Train model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
