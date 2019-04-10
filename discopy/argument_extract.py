import os
import pickle

import nltk

from discopy.confusion_matrix import *
from discopy.conn_head_mapper import *
from discopy.utils import discourse_adverbial, coordinating_connective, subordinating_connective


def rootpath(clause):
    path = ''
    while clause.parent():
        clause = clause.parent()
        path += clause.label() + '-'
    path = path.strip('-')
    return path


def findHead(pdtb):
    chm = ConnHeadMapper()
    for number, relation in enumerate(pdtb):
        if relation['Type'] == 'Explicit':
            head, indices = chm.map_raw_connective(relation['Connective']['RawText'])
            pdtb[number]['ConnectiveHead'] = head
    return pdtb


def lca(ptree, leaf_index):
    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)[:-1]
    if not lca_loc:
        lca_loc = (0,)
    return lca_loc


def height(phrase):
    i = 0
    while phrase.parent() and (phrase.parent().label() != ''):
        phrase = phrase.parent()
        i += 1
    return i


def path(conn, clause):
    if height(conn) == height(clause):
        return conn.label() + 'U' + conn.parent().label() + 'D' + clause.label()
    elif height(conn) > height(clause):
        distance = height(conn) - height(clause) + 1
        p = conn.label()
        parent = conn
        while distance != 0:
            parent = parent.parent()
            p += 'U' + parent.label()
            distance -= 1
        distance = height(clause) - height(parent)
        parent = clause
        down = []
        while distance != 0:
            parent = parent.parent()
            down.append(parent.label())
            distance -= 1
        d = 'D' + clause.label()
        p += d
        return p


def get_index_tree(ptree):
    tree = ptree.copy(deep=True)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        tree[tree_location[:-1]][0] = idx
    return tree


def extract_clause_features(clauses, conn_head, indices, ptree):
    features = []
    if len(indices) == 1:
        lca_loc = ptree.leaf_treeposition(indices[0])[:-1]
        c = ptree[lca_loc]
        left_sib_no = 0
        while c.left_sibling():
            c = c.left_sibling()
            left_sib_no += 1
        c = ptree[lca_loc]
        right_sib_no = 0
        while c.right_sibling():
            c = c.right_sibling()
            right_sib_no += 1
        connective = ptree[lca_loc]
    else:
        lca_loc = lca(ptree, indices)
        conn_last_word = ptree.leaf_treeposition(indices[-1])[:-1]
        conn_first_word = ptree.leaf_treeposition(indices[0])[:-1]
        c = ptree[conn_first_word]
        left_sib_no = 0
        while c.left_sibling():
            c = c.left_sibling()
            left_sib_no += 1
        c = ptree[conn_last_word]
        right_sib_no = 0
        while c.right_sibling():
            c = c.right_sibling()
            right_sib_no += 1
        connective = ptree[lca_loc]
    if conn_head in discourse_adverbial:
        conn_cat = 'Adverbial'
    elif conn_head in coordinating_connective:
        conn_cat = 'Coordinating'
    elif conn_head in subordinating_connective:
        conn_cat = 'Subordinating'
    else:
        conn_cat = None

    conn_pos = ptree.treeposition_spanning_leaves(indices[0], max(indices[0] + 1, indices[-1]))[:-1]

    for num, (clause, n_pos) in enumerate(clauses):
        relative_position = get_relative_position(n_pos, conn_pos)
        clause_pos = clause.label()
        clause_parent_pos = clause.parent().label()
        if clause.left_sibling():
            clause_ls_pos = clause.left_sibling().label()
        else:
            clause_ls_pos = 'NULL'
        if clause.right_sibling():
            clause_rs_pos = clause.right_sibling().label()
        else:
            clause_rs_pos = 'NULL'
        clause_context = clause_pos + '-' + clause_parent_pos + '-' + clause_ls_pos + '-' + clause_rs_pos

        conn2clause_path = path(connective, clause)
        conn2root_path = rootpath(clause)

        featureVector = {'connectiveString': conn_head, 'connectivePOS': connective.label(), 'leftSibNo': left_sib_no,
                         'rightSibNo': right_sib_no, 'connCategory': conn_cat, 'clauseRelPosition': relative_position,
                         'clauseContext': clause_context, 'conn2clausePath': conn2clause_path,
                         'conn2rootPath': conn2root_path}

        clause = ' '.join(clause.leaves())
        features.append((featureVector, clause))
    return features


def extract_ss_arguments(clauses, conn_head, indices, ptree, arg1, arg2):
    feature_set = []
    clause_features = extract_clause_features(clauses, conn_head, indices, ptree)
    for f in clause_features:
        if f[1] in arg1:
            label = 'Arg1'
        elif f[1] in arg2:
            label = 'Arg2'
        else:
            label = 'NULL'
        feature_set.append((f[0], label))
    return feature_set


def extract_ps_arguments(clauses, conn_head, indices, ptree, arg2):
    """
    If Arg1 is in the previous sentence relative to Arg2, a majority classifier in Lin et al. gave the full previous
    sentence as Arg1, which already gives good results in argument extraction
    """
    feature_set = []
    clause_features = extract_clause_features(clauses, conn_head, indices, ptree)
    for f in clause_features:
        if f[1] in arg2:
            label = 'Arg2'
        else:
            label = 'NULL'
        feature_set.append((f[0], label))
    return feature_set


def generate_pdtb_features(pdtb, parses):
    ss_features = []
    ps_features = []

    findHead(pdtb)

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc_id = relation['DocID']
        arg1_sentence_id = relation['Arg1']['TokenList'][0][3]
        arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
        s = parses[doc_id]['sentences'][arg2_sentence_id]['parsetree']
        ptree = nltk.ParentedTree.fromstring(s)

        if not ptree.leaves():
            continue

        indices = [token[4] for token in relation['Connective']['TokenList']]
        clauses = [(ptree[pos], pos) for pos in ptree.treepositions() if type(ptree[pos]) != str and len(pos) > 0]

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


def get_relative_position(pos1, pos2):
    for i in range(min(len(pos1), len(pos2))):
        if pos1[i] < pos2[i]:
            return 'left'
        if pos1[i] > pos2[i]:
            return 'right'
    # if pos2 is contained by pos1
    return 'contains'


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
        clauses = [(ptree[pos], pos) for pos in ptree.treepositions() if type(ptree[pos]) != str and len(pos) > 0]
        features = [i[0] for i in extract_clause_features(clauses, conn_head, indices, ptree)]

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

        return list(arg1), list(arg2)


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = ArgumentExtractClassifier()
    train_ss_data, train_ps_data = generate_pdtb_features(trainpdtb, trainparses)
    clf.fit_on_features(train_ss_data, train_ps_data)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ss_model, train_ss_data)))
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ps_model, train_ps_data)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    val_ss_data, val_ps_data = generate_pdtb_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ss_model, val_ss_data)))
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.ps_model, val_ps_data)))
