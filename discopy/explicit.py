import json
import os
import pickle

import nltk

from discopy.conn_head_mapper import ConnHeadMapper


def lca(ptree, leaf_index):
    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)[:-1]
    if not lca_loc:
        lca_loc = (0,)
    return lca_loc


def findHead(pdtb):
    chm = ConnHeadMapper()
    for number, relation in enumerate(pdtb):
        if relation['Type'] == 'Explicit':
            head, indices = chm.map_raw_connective(relation['Connective']['RawText'])
            pdtb[number]['ConnectiveHead'] = head
    return pdtb


def extract_features(relation, sent):
    connectiveString = relation['Connective']['RawText']
    chm = ConnHeadMapper()
    connHead, _ = chm.map_raw_connective(relation['Connective']['RawText'])
    indices = [index[4] for index in relation['Connective']['TokenList']]
    ptree = nltk.ParentedTree.fromstring(sent['parsetree'])

    lca_loc = lca(ptree, indices)
    connectivePOS = ptree[lca_loc].label()

    if indices[0] == 0:
        connectivePrev = None
    else:
        connectivePrev = sent['words'][indices[0] - 1][0]

    leaves = ptree.leaves()
    sentenceLen = len(leaves)
    m1 = sentenceLen * (1 / 3)
    m2 = sentenceLen * (2 / 3)
    if indices[len(indices) // 2] < m1:
        cPosition = 'START'
    elif indices[len(indices) // 2] >= m1 and indices[len(indices) // 2] < m2:
        cPosition = 'MIDDLE'
    else:
        cPosition = 'END'

    featureVector = {'Connective': connectiveString, 'connectiveHead': connHead,
                     'ConnectivePOS': connectivePOS,
                     'ConnectivePrev': connectivePrev, 'connectivePosition': cPosition}

    return featureVector


def get_features(pdtb, parses):
    findHead(pdtb)
    feature_set = []
    sense_selection = ['Temporal.Asynchronous.Precedence',
                       'Temporal.Asynchronous.Succession',
                       'Temporal.Synchrony',
                       'Contingency.Cause.Reason',
                       'Contingency.Cause.Result',
                       'Contingency.Condition',
                       'Comparison.Contrast',
                       'Comparison.Concession',
                       'Expansion.Conjunction',
                       'Expansion.Instantiation',
                       'Expansion.Restatement',
                       'Expansion.Alternative',
                       'Expansion.Alternative.Chosen alternative',
                       'Expansion.Exception',
                       'EntRel',
                       ]

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        sense = relation['Sense'][0]
        if sense in sense_selection:
            connectiveString = relation['Connective']['RawText']
            connHead = relation['ConnectiveHead']
            indices = [index[4] for index in relation['Connective']['TokenList']]
            sentenceOffSet = relation['Connective']['TokenList'][0][3]
            doc = relation['DocID']

            tree = parses[doc]['sentences'][sentenceOffSet]['parsetree']
            ptree = nltk.ParentedTree.fromstring(tree)
            if not ptree.leaves():
                continue

            lca_loc = lca(ptree, indices)
            connectivePOS = ptree[lca_loc].label()

            if indices[0] == 0:
                connectivePrev = None
            else:
                connectivePrev = parses[doc]['sentences'][sentenceOffSet]['words'][indices[0] - 1][0]

            leaves = ptree.leaves()
            sentenceLen = len(leaves)
            m1 = sentenceLen * (1 / 3)
            m2 = sentenceLen * (2 / 3)
            if indices[len(indices) // 2] < m1:
                cPosition = 'START'
            elif indices[len(indices) // 2] >= m1 and indices[len(indices) // 2] < m2:
                cPosition = 'MIDDLE'
            else:
                cPosition = 'END'

            featureVector = {'Connective': connectiveString, 'connectiveHead': connHead,
                             'ConnectivePOS': connectivePOS,
                             'ConnectivePrev': connectivePrev, 'connectivePosition': cPosition}

            sense = relation['Sense'][0]
            feature_set.append((featureVector, sense))
    return feature_set


class ExplicitSenseClassifier:
    def __init__(self):
        self.model = None

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        features = get_features(pdtb, parses)
        self.fit_on_features(features, max_iter=max_iter)

    def fit_on_features(self, features, max_iter=5):
        self.model = nltk.MaxentClassifier.train(features, max_iter=max_iter)

    def predict(self, X):
        pass

    def get_sense(self, relation, sent):
        features = extract_features(relation, sent)
        return [self.model.classify(features)]


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    train_data = get_features(trainpdtb, trainparses)
    clf = ExplicitSenseClassifier()
    clf.fit_on_features(train_data)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, train_data)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    val_data = get_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, val_data)))

    print(train_data[:5], len(train_data))
