import json
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import nltk

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.utils import single_connectives, multi_connectives_first, multi_connectives, distant_connectives


def match_connectives(sentence, word_idx) -> list:
    sentence = [w[0].lower() for w in sentence]
    word = sentence[word_idx]
    if word in single_connectives:
        return [word]
    elif word in multi_connectives_first:
        if word_idx == (len(sentence) - 1):
            if word in ['as', 'before', 'for', 'if', 'so']:
                return [word]
            else:
                return []

        for multi_conn in multi_connectives:
            if (word_idx + len(multi_conn)) < len(sentence) and all(
                    c == sentence[word_idx + i] for i, c in enumerate(multi_conn)):
                return multi_conn

        for conn in distant_connectives:
            if word == conn[0]:
                try:
                    i = sentence.index(conn[1], word_idx)
                    return sentence[word_idx:i - word_idx]
                except ValueError:
                    continue
    return []


def get_features(ptree, leaf_index):
    leave_list = ptree.leaves()
    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)[:-1]
    if not lca_loc:
        lca_loc = (0,)

    selfcategory = ptree[lca_loc].label()
    parentcategory = ptree[lca_loc].parent().label()

    r2l = [ptree[lca_loc[:i + 1]].label() for i in range(len(lca_loc))]

    labels = {n.label() for n in ptree.subtrees(lambda t: t.height() > 2)}

    leftSibling = ptree[lca_loc].left_sibling()
    if leftSibling:
        leftSibling = leftSibling.label()

    rightSibling = ptree[lca_loc].right_sibling()
    rightVP = 'VP' in labels
    rightTR = 'T' in labels
    if rightSibling:
        rightSibling = rightSibling.label()

    prev = leaf_index[0] - 1
    next = leaf_index[len(leaf_index) - 1] + 1

    pl = ptree.pos()
    cPOS = selfcategory
    c = ' '.join(leave_list[leaf_index[0]:leaf_index[-1] + 1])
    c = c.lower()

    if prev >= 0:
        prevC = ','.join([pl[prev][0], c])
        prevPOS = pl[prev][1]
        prevPOScPOS = ','.join([pl[prev][1], cPOS])
    else:
        prevC = ','.join(['NONE', c])
        prevPOS = 'NONE'
        prevPOScPOS = ','.join(['NONE', cPOS])

    if next < len(leave_list):
        nextC = [pl[next][0], c]
        nextC = ','.join(nextC)
        nextPOS = pl[next][1]
        nextPOScPOS = [pl[next][1], cPOS]
        nextPOScPOS = ','.join(nextPOScPOS)
    else:
        nextC = ['NONE', c]
        nextC = ','.join(nextC)
        nextPOS = 'NONE'
        nextPOScPOS = ['NONE', cPOS]
        nextPOScPOS = ','.join(nextPOScPOS)

    # remove repeating labels
    r2lcomp = r2l[:1]
    for i in r2l[1:]:
        if r2lcomp[-1] != i:
            r2lcomp.append(i)

    feat = {'connective': c, 'connectivePOS': cPOS, 'prevWord': prevC, 'prevPOSTag': prevPOS,
            'prevPOS+cPOS': prevPOScPOS, 'nextWord': nextC, 'nextPOSTag': nextPOS, 'cPOS+nextPOS': nextPOScPOS,
            'root2LeafCompressed': ','.join(r2lcomp), 'root2Leaf': ','.join(r2l), 'leftSibling': leftSibling,
            'rightSibling': rightSibling, 'parentCategory': parentcategory, 'boolVP': rightVP, 'boolTrace': rightTR}

    return feat


def group_by_doc_id(pdtb: list) -> Dict[str, list]:
    pdtb_by_doc = defaultdict(list)
    for r in filter(lambda r: r['Type'] == 'Explicit', pdtb):
        pdtb_by_doc[r['DocID']].append(r)
    return pdtb_by_doc


def generate_pdtb_features(pdtb, parses):
    chm = ConnHeadMapper()
    featureSets = []
    pdtb = group_by_doc_id(pdtb)
    for doc_id, doc in parses.items():
        for sentence in doc['sentences']:
            parsetree = nltk.ParentedTree.fromstring(sentence['parsetree'])
            if not parsetree.leaves():
                continue
            doc_relations = pdtb[doc_id]
            words = sentence['words']  # a word is a complex object containing information about offset etc
            for word_idx, (word, wordDictionary) in enumerate(words):
                connective_candidate = match_connectives(words, word_idx)

                if connective_candidate:
                    skip = len(connective_candidate) - 1
                    word = word.lower()
                    cOBWord = wordDictionary['CharacterOffsetBegin']
                    cOEWord = wordDictionary['CharacterOffsetEnd']

                    tokenNo = list(range(word_idx, word_idx + skip + 1))
                    # check all relations in pdtb for this document
                    for relation in doc_relations:
                        connective, _ = chm.map_raw_connective(relation['Connective']['RawText'])
                        cOBConnective = relation['Connective']['CharacterSpanList'][0][0]
                        cOEConnective = relation['Connective']['CharacterSpanList'][-1][1]
                        if cOBConnective <= cOBWord and cOEWord <= cOEConnective:
                            l = [string.lower() for string in connective.split()]
                            if (word in l):
                                featureSets.append((get_features(parsetree, tokenNo), 'Y'))
                                break
                    else:
                        # use connective candidate as negative example if there is no relation marked in PDTB
                        featureSets.append((get_features(parsetree, tokenNo), 'N'))
    return featureSets


class ConnectiveClassifier:
    def __init__(self):
        self.model = None

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'connective_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'connective_clf.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        features = generate_pdtb_features(pdtb, parses)
        self.fit_on_features(features, max_iter=max_iter)

    def fit_on_features(self, features, max_iter=5):
        self.model = nltk.MaxentClassifier.train(features, max_iter=max_iter)

    def predict(self, X):
        pass

    def get_connective(self, parsetree, sentence, word_idx) -> List[str]:
        # TODO prob_classify function for confidence of classifier
        candidate = match_connectives(sentence, word_idx)
        if not candidate:
            return []
        else:
            conn_label = self.model.classify(
                get_features(parsetree, list(range(word_idx, word_idx + len(candidate)))))
            if conn_label is 'N':
                return []
            else:
                return candidate


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = ConnectiveClassifier()
    train_data = generate_pdtb_features(trainpdtb, trainparses)
    clf.fit_on_features(train_data)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, train_data)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    val_data = generate_pdtb_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, val_data)))
