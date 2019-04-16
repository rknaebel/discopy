import json
import os
import pickle

import nltk

import discopy.conn_head_mapper



def get_features(ptree, c, leaf_index):
    leaves = ptree.leaves()
    chm = discopy.conn_head_mapper.ConnHeadMapper()
    head, connectiveHead_index = chm.map_raw_connective(c)
    connectiveHead_index = [leaf_index[i] for i in connectiveHead_index]

    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)[:-1]
    if not lca_loc:
        lca_loc = (0,)

    cPOS = ptree[lca_loc].label()

    sentenceLen = len(leaves)
    m1 = sentenceLen * (1 / 3)
    m2 = sentenceLen * (2 / 3)

    if connectiveHead_index[len(connectiveHead_index) // 2] < m1:
        cPosition = 'START'
    elif connectiveHead_index[len(connectiveHead_index) // 2] >= m1 and connectiveHead_index[
        len(connectiveHead_index) // 2] < m2:
        cPosition = 'MIDDLE'
    else:
        cPosition = 'END'

    prev = leaf_index[0] - 1
    prev2 = prev - 1
    pl = ptree.pos()

    if prev >= 0:
        prevC = [pl[prev][0], head]
        prevC = ', '.join(prevC)
        prevPOS = pl[prev][1]
        prevPOScPOS = [pl[prev][1], cPOS]
        prevPOScPOS = ', '.join(prevPOScPOS)
        prev = pl[prev][0]
    else:
        prevC = ['NONE', head]
        prevC = ', '.join(prevC)
        prevPOS = 'NONE'
        prevPOScPOS = ['NONE', cPOS]
        prevPOScPOS = ', '.join(prevPOScPOS)
        prev = 'NONE'

    if prev2 >= 0:
        prev2C = [pl[prev2][0], head]
        prev2C = ', '.join(prev2C)
        prev2POS = pl[prev2][1]
        prev2POScPOS = [pl[prev2][1], cPOS]
        prev2POScPOS = ', '.join(prev2POScPOS)
        prev2 = pl[prev2][0]

    else:
        prev2C = ['NONE', head]
        prev2C = ', '.join(prev2C)
        prev2POS = 'NONE'
        prev2POScPOS = ['NONE', cPOS]
        prev2POScPOS = ', '.join(prev2POScPOS)
        prev2 = 'NONE'

    feat = {'connective': head, 'connectivePOS': cPOS, 'cPosition': cPosition, 'prevWord+c': prevC,
            'prevPOSTag': prevPOS, 'prevPOS+cPOS': prevPOScPOS, 'prevWord': prev, 'prev2Word+c': prev2C,
            'prev2POSTag': prev2POS, 'prev2POS+cPOS': prev2POScPOS, 'prevWord2': prev2}

    return feat


def generate_pdtb_features(pdtb, parses):
    featureSet = []
    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc = relation['DocID']
        connective = relation['Connective']['TokenList']
        connectiveRawText = relation['Connective']['RawText']
        sentenceOffset = connective[0][3]
        connectiveTokens = [token[4] for token in connective]
        ptree = parses[doc]['sentences'][sentenceOffset]['parsetree']
        ptree = nltk.ParentedTree.fromstring(ptree)

        if not ptree.leaves():
            continue
        arg1Set = set()
        arg2Set = set()
        for i in relation['Arg1']['TokenList']:
            arg1Set.add(i[3])
        for i in relation['Arg2']['TokenList']:
            arg2Set.add(i[3])

        arg1 = list(arg1Set)
        arg2 = list(arg2Set)
        arg1.sort()
        arg2.sort()
        if arg1[-1] < arg2[0]:
            featureSet.append((get_features(ptree, connectiveRawText, connectiveTokens), 'PS'))

        if len(arg1Set) == 1 and len(arg2Set) == 1:
            if arg1Set.pop() == arg2Set.pop():
                featureSet.append((get_features(ptree, connectiveRawText, connectiveTokens), 'SS'))
    return featureSet


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
