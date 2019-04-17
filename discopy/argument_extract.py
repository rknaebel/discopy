import json
import os
import pickle

import nltk
import numpy as np

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.features import get_clause_context, get_connective_category, get_relative_position, \
    get_clause_direction_path
from discopy.features import get_root_path
from discopy.features import get_sibling_counts, get_clauses


def add_connective_heads(pdtb):
    chm = ConnHeadMapper()
    for number, relation in enumerate(pdtb):
        if relation['Type'] == 'Explicit':
            head, indices = chm.map_raw_connective(relation['Connective']['RawText'])
            pdtb[number]['ConnectiveHead'] = head
    return pdtb


def lca(ptree, leaf_index):
    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)
    if type(ptree[lca_loc]) == str:
        lca_loc = lca_loc[:-1]
    return lca_loc


def get_index_tree(ptree):
    tree = ptree.copy(deep=True)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        tree[tree_location[:-1]][0] = idx
    return tree


def get_features(clauses, conn_head, indices, ptree):
    features = []
    lca_loc = lca(ptree, indices)
    connective = ptree[lca_loc]

    left_siblings_no, right_siblings_no = get_sibling_counts(ptree[lca_loc])

    conn_cat = get_connective_category(conn_head)

    conn_pos = ptree.treeposition_spanning_leaves(indices[0], max(indices[0] + 1, indices[-1]))[:-1]

    for num, (clause, n_pos) in enumerate(clauses):
        relative_position = get_relative_position(n_pos, conn_pos)
        clause_context = get_clause_context(clause)

        conn2clause_path = get_clause_direction_path(connective, clause)
        conn2root_path = get_root_path(clause)

        feat = {'connectiveString': conn_head, 'connectivePOS': connective.label(), 'leftSibNo': left_siblings_no,
                'rightSibNo': right_siblings_no, 'connCategory': conn_cat, 'clauseRelPosition': relative_position,
                'clauseContext': clause_context, 'conn2clausePath': conn2clause_path,
                'conn2rootPath': conn2root_path}

        clause = ' '.join(clause.leaves())
        features.append((feat, clause))
    return features


def extract_ss_arguments(clauses, conn_head, indices, ptree, arg1, arg2):
    feature_set = []
    clause_features = get_features(clauses, conn_head, indices, ptree)
    for feature, clause in clause_features:
        # TODO comparison is not correct. replace by tree position
        if clause in arg1:
            label = 'Arg1'
        elif clause in arg2:
            label = 'Arg2'
        else:
            label = 'NULL'
        feature_set.append((feature, label))
    return feature_set


def extract_ps_arguments(clauses, conn_head, indices, ptree, arg2):
    """
    If Arg1 is in the previous sentence relative to Arg2, a majority classifier in Lin et al. gave the full previous
    sentence as Arg1, which already gives good results in argument extraction
    """
    feature_set = []
    clause_features = get_features(clauses, conn_head, indices, ptree)
    for feature, clause in clause_features:
        # TODO comparison is not correct. replace by tree position
        if clause in arg2:
            label = 'Arg2'
        else:
            label = 'NULL'
        feature_set.append((feature, label))
    return feature_set


def generate_pdtb_features(pdtb, parses):
    ss_features = []
    ps_features = []

    add_connective_heads(pdtb)

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc_id = relation['DocID']
        arg1_sentence_id = relation['Arg1']['TokenList'][0][3]
        arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
        s = parses[doc_id]['sentences'][arg2_sentence_id]['parsetree']
        ptree = nltk.ParentedTree.fromstring(s)

        if not ptree.leaves():
            continue

        indices = [token[4] for token in relation['Connective']['TokenList']]
        clauses = get_clauses(ptree)

        # Arg1 is in the same sentence (SS)
        if arg1_sentence_id == arg2_sentence_id:
            ss_features.extend(
                extract_ss_arguments(clauses, relation['ConnectiveHead'], indices, ptree,
                                     relation['Arg1']['RawText'], relation['Arg2']['RawText']))
        # Arg1 is in the previous sentence (PS)
        elif (arg2_sentence_id - arg1_sentence_id) == 1:
            ps_features.extend(
                extract_ps_arguments(clauses, relation['ConnectiveHead'], indices, ptree,
                                     relation['Arg2']['RawText']))
    return ss_features, ps_features


class ArgumentExtractClassifier:
    def __init__(self):
        self.ss_model = None
        self.ps_model = None

    def load(self, path):
        self.ss_model = pickle.load(open(os.path.join(path, 'ss_extract_clf.p'), 'rb'))
        self.ps_model = pickle.load(open(os.path.join(path, 'ps_extract_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.ss_model, open(os.path.join(path, 'ss_extract_clf.p'), 'wb'))
        pickle.dump(self.ps_model, open(os.path.join(path, 'ps_extract_clf.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        ss_features, ps_features = generate_pdtb_features(pdtb, parses)
        self.fit_on_features(ss_features, ps_features, max_iter=max_iter)

    def fit_on_features(self, ss_features, ps_features, max_iter=5):
        self.ss_model = nltk.MaxentClassifier.train(ss_features, max_iter=max_iter)
        self.ps_model = nltk.MaxentClassifier.train(ps_features, max_iter=max_iter)

    def predict(self, X):
        pass

    def extract_arguments(self, ptree, relation):
        indices = [token[4] for token in relation['Connective']['TokenList']]
        chm = ConnHeadMapper()
        conn_head, _ = chm.map_raw_connective(relation['Connective']['RawText'])
        ptree._label = 'S'
        clauses = get_clauses(ptree)

        features = [i[0] for i in get_features(clauses, conn_head, indices, ptree)]

        if relation['ArgPos'] == 'SS':
            preds = self.ss_model.prob_classify_many(features)

        elif relation['ArgPos'] == 'PS':
            preds = self.ps_model.prob_classify_many(features)

        else:
            raise NotImplementedError('Unknown argument position')

        arg1_probs = np.array([pred.prob('Arg1') for pred in preds])
        arg2_probs = np.array([pred.prob('Arg2') for pred in preds])

        ptree_ids = get_index_tree(ptree)

        arg1 = set(ptree_ids[clauses[arg1_probs.argmax()][1]].leaves())
        arg2 = set(ptree_ids[clauses[arg2_probs.argmax()][1]].leaves())
        arg1 = arg1 - arg2

        return list(arg1), list(arg2), arg1_probs.max(), arg2_probs.max()


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in
                 open('../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = ArgumentExtractClassifier()
    train_ss_data, train_ps_data = generate_pdtb_features(trainpdtb, trainparses)
    clf.fit_on_features(train_ss_data, train_ps_data, max_iter=5)
    clf.save('tmp')
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ss_model, train_ss_data)))
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ps_model, train_ps_data)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    val_ss_data, val_ps_data = generate_pdtb_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ss_model, val_ss_data)))
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ps_model, val_ps_data)))
