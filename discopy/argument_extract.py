import os
import pickle
import ujson as json

import nltk
import sklearn
import sklearn.pipeline

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.features import get_clause_context, get_connective_category, get_relative_position, \
    get_clause_direction_path, lca, get_index_tree
from discopy.features import get_root_path
from discopy.features import get_sibling_counts, get_clauses


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

        feat = {'connectiveString': conn_head, 'connectivePOS': str(connective.label()), 'leftSibNo': left_siblings_no,
                'rightSibNo': right_siblings_no, 'connCategory': conn_cat, 'clauseRelPosition': relative_position,
                'clauseContext': clause_context, 'conn2clausePath': str(conn2clause_path),
                'conn2rootPath': conn2root_path}

        features.append((feat, n_pos))
    return features


def extract_ss_arguments(clauses, conn_head, indices, ptree, arg1, arg2):
    feature_set = []
    ptree_ids = get_index_tree(ptree)
    clause_features = get_features(clauses, conn_head, indices, ptree)
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


def extract_ps_arguments(clauses, conn_head, indices, ptree, arg2):
    """
    If Arg1 is in the previous sentence relative to Arg2, a majority classifier in Lin et al. gave the full previous
    sentence as Arg1, which already gives good results in argument extraction
    """
    feature_set = []
    ptree_ids = get_index_tree(ptree)
    clause_features = get_features(clauses, conn_head, indices, ptree)
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
        chm = ConnHeadMapper()
        conn_head, _ = chm.map_raw_connective(relation['Connective']['RawText'])
        try:
            arg1_sentence_id = relation['Arg1']['TokenList'][0][3]
            arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
            s = parses[doc_id]['sentences'][arg2_sentence_id]['parsetree']
            ptree = nltk.ParentedTree.fromstring(s)
        except ValueError:
            continue
        except IndexError:
            continue

        if not ptree.leaves():
            continue

        arg1 = set([i[4] for i in relation['Arg1']['TokenList']])
        arg2 = set([i[4] for i in relation['Arg2']['TokenList']])

        indices = [token[4] for token in relation['Connective']['TokenList']]
        clauses = get_clauses(ptree)

        # Arg1 is in the same sentence (SS)
        if arg1_sentence_id == arg2_sentence_id:
            ss_features.extend(extract_ss_arguments(clauses, conn_head, indices, ptree, arg1, arg2))
        # Arg1 is in the previous sentence (PS)
        elif (arg2_sentence_id - arg1_sentence_id) == 1:
            ps_features.extend(extract_ps_arguments(clauses, conn_head, indices, ptree, arg2))
    return list(zip(*ss_features)), list(zip(*ps_features))


class ArgumentExtractClassifier:
    def __init__(self):
        self.ss_model = sklearn.pipeline.Pipeline([
            ('vectorizer', sklearn.feature_extraction.DictVectorizer(sparse=False)),
            ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=100)),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                                              n_jobs=-1, max_iter=150))
        ])
        self.ps_model = sklearn.pipeline.Pipeline([
            ('vectorizer', sklearn.feature_extraction.DictVectorizer(sparse=False)),
            ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=100)),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                                              n_jobs=-1, max_iter=150))
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

    def extract_arguments(self, ptree, relation):
        indices = [token[4] for token in relation['Connective']['TokenList']]
        chm = ConnHeadMapper()
        conn_head, _ = chm.map_raw_connective(relation['Connective']['RawText'])
        ptree._label = 'S'
        clauses = get_clauses(ptree)
        ptree_ids = get_index_tree(ptree)

        X = [i[0] for i in get_features(clauses, conn_head, indices, ptree)]
        if relation['ArgPos'] == 'SS':
            probs = self.ss_model.predict_proba(X)
            _, arg1_max_idx, arg2_max_idx = probs.argmax(axis=0)
            _, arg1_prob, arg2_prob = probs.max(axis=0)

            arg1 = set(ptree_ids[clauses[arg1_max_idx][1]].leaves())
            arg2 = set(ptree_ids[clauses[arg2_max_idx][1]].leaves())
            if arg1.issubset(arg2):
                arg2 = arg2 - arg1
            elif arg1.issuperset(arg2):
                arg1 = arg1 - arg2
        elif relation['ArgPos'] == 'PS':
            probs = self.ps_model.predict_proba(X)
            _, arg2_max_idx = probs.argmax(axis=0)
            _, arg2_prob = probs.max(axis=0)
            arg1_prob = 1.0

            arg1 = set()
            arg2 = set(ptree_ids[clauses[arg2_max_idx][1]].leaves())
        else:
            raise NotImplementedError('Unknown argument position')

        return sorted(arg1), sorted(arg2), arg1_prob, arg2_prob


if __name__ == "__main__":
    pdtb_train = [json.loads(s) for s in
                  open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    parses_val = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = ArgumentExtractClassifier()
    (X_ss, y_ss), (X_ps, y_ps) = generate_pdtb_features(pdtb_train, parses_train)
    clf.ss_model.fit(X_ss, y_ss)
    clf.ps_model.fit(X_ps, y_ps)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(clf.ss_model.score(X_ss, y_ss)))
    print('ACCURACY {}'.format(clf.ps_model.score(X_ps, y_ps)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    (X_val_ss, y_val_ss), (X_val_ps, y_val_ps) = generate_pdtb_features(pdtb_val, parses_val)
    print('ACCURACY {}'.format(clf.ss_model.score(X_val_ss, y_val_ss)))
    print('ACCURACY {}'.format(clf.ps_model.score(X_val_ps, y_val_ps)))
    clf.save('../tmp')
