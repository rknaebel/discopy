import logging
import os
import pickle
from typing import List

import click
import nltk
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline

from discopy.components.argument.position import ArgumentPositionClassifier
from discopy.components.component import Component
from discopy.features import get_clause_context, get_connective_category, get_relative_position, \
    get_clause_direction_path, lca, get_index_tree
from discopy.features import get_root_path
from discopy.features import get_sibling_counts, get_clauses
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_parsed_conll_dataset
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')


def simplify_tree(ptree: nltk.ParentedTree, collapse_root=False):
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


def generate_pdtb_features(docs: List[Document]):
    ss_features = []
    ps_features = []
    for doc in docs:
        for relation in filter(lambda r: r.type == 'Explicit', doc.relations):
            conn = ' '.join(t.surface for t in relation.conn.tokens)
            arg1_sentence_id = relation.arg1.get_sentence_idxs()[0]
            arg2_sentence_id = relation.arg2.get_sentence_idxs()[0]
            ptree = doc.sentences[arg2_sentence_id].get_ptree()
            if ptree is None:
                continue
            ptree = simplify_tree(ptree)
            conn_idxs = list(t.local_idx for t in relation.conn.tokens)
            arg1_idxs = set(t.local_idx for t in relation.arg1.tokens)
            arg2_idxs = set(t.local_idx for t in relation.arg2.tokens)
            # Arg1 is in the same sentence (SS)
            if arg1_sentence_id == arg2_sentence_id:
                ss_features.extend(extract_ss_arguments(conn, conn_idxs, ptree, arg1_idxs, arg2_idxs))
            # Arg1 is in the previous sentence (PS)
            elif (arg2_sentence_id - arg1_sentence_id) == 1:
                ps_features.extend(extract_ps_arguments(conn, conn_idxs, ptree, arg2_idxs))
    return list(zip(*ss_features)), list(zip(*ps_features))


class ExplicitArgumentExtractor(Component):
    model_name = 'explicit_argument_extractor'
    used_features = ['ptree']

    def __init__(self):
        self.arg_pos_clf = ArgumentPositionClassifier()
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
        self.arg_pos_clf.load(path)
        self.ss_model = pickle.load(open(os.path.join(path, 'ss_extract_clf.p'), 'rb'))
        self.ps_model = pickle.load(open(os.path.join(path, 'ps_extract_clf.p'), 'rb'))

    def save(self, path):
        self.arg_pos_clf.save(path)
        pickle.dump(self.ss_model, open(os.path.join(path, 'ss_extract_clf.p'), 'wb'))
        pickle.dump(self.ps_model, open(os.path.join(path, 'ps_extract_clf.p'), 'wb'))

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        self.arg_pos_clf.fit(docs_train)
        (X_ss, y_ss), (X_ps, y_ps) = generate_pdtb_features(docs_train)
        self.ss_model.fit(X_ss, y_ss)
        self.ps_model.fit(X_ps, y_ps)

    def score_on_features(self, x_ss, y_ss, x_ps, y_ps):
        y_pred = self.ss_model.predict_proba(x_ss)
        y_pred_c = self.ss_model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: SS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ss, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ss, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ss, y_pred_c)))

        y_pred = self.ps_model.predict_proba(x_ps)
        y_pred_c = self.ps_model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: PS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ps, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ps, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ps, y_pred_c)))

    def score(self, docs: List[Document]):
        self.arg_pos_clf.score(docs)
        (x_ss, y_ss), (x_ps, y_ps) = generate_pdtb_features(docs)
        self.score_on_features(x_ss, y_ss, x_ps, y_ps)

    def extract_arguments(self, ptree: nltk.ParentedTree, relation: Relation, arg_pos: str):
        conn = ' '.join(t.surface for t in relation.conn.tokens)
        ptree = simplify_tree(ptree)
        conn_idxs = list(t.local_idx for t in relation.conn.tokens)
        ptree_ids = get_index_tree(ptree)
        fs = sorted(get_features(conn, conn_idxs, ptree), key=lambda x: x[1])
        if not fs:
            print("Empty feature sequence", conn, conn_idxs, ptree)
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
            arg1_prob, arg2_prob = 1.0, probs[arg2_max_idx, 0]
            arg1, arg2 = set(), set(ptree_ids[position[arg2_max_idx]].leaves())
            if not arg2:
                logger.warning("PS Empty Arg2")
        else:
            raise NotImplementedError('Unknown argument position')

        return sorted(arg1), sorted(arg2), arg1_prob, arg2_prob

    def parse(self, doc: Document, relations: List[Relation] = None, **kwargs):
        if relations is None:
            raise ValueError('Component needs connectives already classified.')
        for relation in filter(lambda r: r.type == "Explicit", relations):
            sent_id = relation.conn.get_sentence_idxs()[0]
            sent = doc.sentences[sent_id]
            ptree = sent.get_ptree()
            if ptree is None or len(relation.conn.tokens) == 0:
                continue
            # ARGUMENT POSITION
            conn_raw = ' '.join(t.surface for t in relation.conn.tokens)
            conn_idxs = [t.local_idx for t in relation.conn.tokens]
            arg_pos, arg_pos_confidence = self.arg_pos_clf.get_argument_position(ptree, conn_raw, conn_idxs)
            # If position poorly classified as PS, go to the next relation
            if arg_pos == 'PS' and sent_id == 0:
                continue
            # ARGUMENT EXTRACTION
            arg1, arg2, arg1_c, arg2_c = self.extract_arguments(ptree, relation, arg_pos)
            if arg_pos == 'PS':
                prev_sent = doc.sentences[sent_id]
                relation.arg1.tokens = prev_sent.tokens
                relation.arg2.tokens = [sent.tokens[i] for i in arg2]
            elif arg_pos == 'SS':
                relation.arg1.tokens = [sent.tokens[i] for i in arg1]
                relation.arg2.tokens = [sent.tokens[i] for i in arg2]
            else:
                logger.error('Unknown Argument Position: ' + arg_pos)

        return relations


class ImplicitArgumentExtractor(Component):
    model_name = "implicit_argument_extractor"

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        pass

    def score(self, docs: List[Document]):
        pass

    def parse(self, doc: Document, relations: List[Relation] = None, **kwargs):
        if relations is None:
            relations: List[Relation] = []
        inter_relations = set()
        for relation in relations:
            arg1_idxs = relation.arg1.get_sentence_idxs()
            arg2_idxs = relation.arg2.get_sentence_idxs()
            if not arg1_idxs or not arg2_idxs:
                continue
            elif max(arg1_idxs) == min(arg2_idxs) - 1:
                inter_relations.add(min(arg2_idxs) - 1)
            elif max(arg2_idxs) == min(arg1_idxs) - 1:
                inter_relations.add(min(arg1_idxs) - 1)
        if len(doc.sentences) <= 1:
            return []
        for sent_id, sent in enumerate(doc.sentences[1:]):
            if sent_id - 1 in inter_relations:
                continue
            relations.append(Relation(
                arg1=doc.sentences[sent_id - 1].tokens,
                arg2=doc.sentences[sent_id].tokens,
                type='Implicit'
            ))
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_parsed_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = ExplicitArgumentExtractor()
    logger.info('Train model')
    clf.fit(docs_train)
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)


if __name__ == "__main__":
    main()
