import json
import os
import pickle

import nltk

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.features import get_connective_sentence_position, lca


def findHead(pdtb):
    chm = ConnHeadMapper()
    for number, relation in enumerate(pdtb):
        if relation['Type'] == 'Explicit':
            head, indices = chm.map_raw_connective(relation['Connective']['RawText'])
            pdtb[number]['ConnectiveHead'] = head
    return pdtb


def get_features(relation, ptree):
    connective = relation['Connective']['RawText']
    chm = ConnHeadMapper()
    head, connective_head_index = chm.map_raw_connective(connective)
    leaf_index = [index[4] for index in relation['Connective']['TokenList']]
    connective_head_index = [leaf_index[i] for i in connective_head_index]

    lca_loc = lca(ptree, leaf_index)
    conn_tag = ptree[lca_loc].label()

    if leaf_index[0] == 0:
        prev = "NONE"
    else:
        prev = ptree.leaves()[leaf_index[0] - 1][0]

    conn_pos_relative = get_connective_sentence_position(connective_head_index, ptree)

    feat = {'Connective': connective, 'connectiveHead': head,
            'ConnectivePOS': conn_tag,
            'ConnectivePrev': prev, 'connectivePosition': conn_pos_relative}
    return feat


def generate_pdtb_features(pdtb, parses):
    findHead(pdtb)
    features = []
    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        sentenceOffSet = relation['Connective']['TokenList'][0][3]
        doc = relation['DocID']
        sense = relation['Sense'][0]
        ptree = nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentenceOffSet]['parsetree'])
        if not ptree.leaves():
            continue
        features.append((get_features(relation, ptree), sense))
    return features


class ExplicitSenseClassifier:
    def __init__(self):
        self.model = None

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        features = generate_pdtb_features(pdtb, parses)
        self.fit_on_features(features, max_iter=max_iter)

    def fit_on_features(self, features, max_iter=5):
        self.model = nltk.MaxentClassifier.train(features, max_iter=max_iter)

    def predict(self, X):
        pass

    def get_sense(self, relation, ptree):
        features = get_features(relation, ptree)
        sense = self.model.prob_classify(features)
        return sense.max(), sense.prob(sense.max())


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    train_data = generate_pdtb_features(trainpdtb, trainparses)
    clf = ExplicitSenseClassifier()
    clf.fit_on_features(train_data)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, train_data)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    val_data = generate_pdtb_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, val_data)))
