import json
import os
import pickle

import nltk
from nltk.tree import ParentedTree

import discopy.conn_head_mapper
from discopy.features import get_connective_sentence_position, lca, get_pos_features


def get_features(ptree: ParentedTree, connective: str, leaf_index: list):
    chm = discopy.conn_head_mapper.ConnHeadMapper()
    head, connective_head_index = chm.map_raw_connective(connective)
    connective_head_index = [leaf_index[i] for i in connective_head_index]

    lca_loc = lca(ptree, leaf_index)
    conn_tag = ptree[lca_loc].label()
    conn_pos_relative = get_connective_sentence_position(connective_head_index, ptree)

    prev, prev_conn, prev_pos, prev_pos_conn_pos = get_pos_features(ptree, leaf_index, head, -1)
    prev2, prev2_conn, prev2_pos, prev2_pos_conn_pos = get_pos_features(ptree, leaf_index, head, -2)

    feat = {'connective': head, 'connectivePOS': conn_tag, 'cPosition': conn_pos_relative, 'prevWord+c': prev_conn,
            'prevPOSTag': prev_pos, 'prevPOS+cPOS': prev_pos_conn_pos, 'prevWord': prev, 'prev2Word+c': prev2_conn,
            'prev2POSTag': prev2_pos, 'prev2POS+cPOS': prev2_pos_conn_pos, 'prevWord2': prev2}

    return feat


def generate_pdtb_features(pdtb, parses):
    features = []
    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc = relation['DocID']
        connective = relation['Connective']['TokenList']
        connective_raw = relation['Connective']['RawText']
        leaf_indices = [token[4] for token in connective]
        ptree = parses[doc]['sentences'][connective[0][3]]['parsetree']
        ptree = nltk.ParentedTree.fromstring(ptree)

        if not ptree.leaves():
            continue

        arg1 = sorted({i[3] for i in relation['Arg1']['TokenList']})
        arg2 = sorted({i[3] for i in relation['Arg2']['TokenList']})
        if arg1[-1] < arg2[0]:
            features.append((get_features(ptree, connective_raw, leaf_indices), 'PS'))
        elif len(arg1) == 1 and len(arg2) == 1 and arg1[0] == arg2[0]:
            features.append((get_features(ptree, connective_raw, leaf_indices), 'SS'))
    return features


class ArgumentPositionClassifier:
    def __init__(self):
        self.pos_model = None

    def load(self, path):
        self.pos_model = pickle.load(open(os.path.join(path, 'position_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.pos_model, open(os.path.join(path, 'position_clf.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        pos_features = generate_pdtb_features(pdtb, parses)
        self.fit_on_features(pos_features, max_iter=max_iter)

    def fit_on_features(self, pos_features, max_iter=5):
        self.pos_model = nltk.MaxentClassifier.train(pos_features, max_iter=max_iter)

    def predict(self, X):
        pass

    def get_argument_position(self, parse, connective: str, leaf_index):
        features = get_features(parse, connective, leaf_index)
        arg_position = self.pos_model.classify(features)
        return arg_position


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    trainfeatureSet = generate_pdtb_features(trainpdtb, trainparses)
    devfeatureSet = generate_pdtb_features(devpdtb, devparses)
    clf = ArgumentPositionClassifier()
    clf.fit_on_features(trainfeatureSet, max_iter=5)
    print('......................................ON TRAINING DATA..................')
    print('Accuracy = ', nltk.classify.accuracy(clf.pos_model, trainfeatureSet))

    print('......................................ON DEVELOPMENT DATA..................')
    print('Accuracy = ', nltk.classify.accuracy(clf.pos_model, devfeatureSet))
