import logging
import os
import pickle
import ujson as json
from collections import defaultdict

import nltk
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
from sklearn.pipeline import Pipeline
from typing import Dict, List

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.utils import single_connectives, multi_connectives_first, multi_connectives, distant_connectives, \
    init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


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
    else:
        leftSibling = 'NONE'

    rightSibling = ptree[lca_loc].right_sibling()
    rightVP = 'VP' in labels
    rightTR = 'T' in labels
    if rightSibling:
        rightSibling = rightSibling.label()
    else:
        rightSibling = 'NONE'

    prev = leaf_index[0] - 1
    next = leaf_index[len(leaf_index) - 1] + 1

    pl = ptree.pos()
    cPOS = selfcategory
    c = ' '.join(leave_list[leaf_index[0]:leaf_index[-1] + 1])
    c = c.lower()

    if prev >= 0:
        prevC = ','.join([lemmatizer.lemmatize(pl[prev][0]), c])
        prevPOS = pl[prev][1]
        prevPOScPOS = ','.join([pl[prev][1], cPOS])
    else:
        prevC = ','.join(['NONE', c])
        prevPOS = 'NONE'
        prevPOScPOS = ','.join(['NONE', cPOS])

    if next < len(leave_list):
        nextC = ','.join([lemmatizer.lemmatize(pl[next][0]), c])
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
    features = []
    pdtb = group_by_doc_id(pdtb)
    for doc_id, doc in parses.items():
        for sentence in doc['sentences']:
            try:
                parsetree = nltk.ParentedTree.fromstring(sentence['parsetree'])
            except ValueError:
                continue
            if not parsetree.leaves() or doc_id not in pdtb:
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
                        cOBConnective = relation['Connective']['TokenList'][0][0]
                        cOEConnective = relation['Connective']['TokenList'][-1][1]
                        if cOBConnective <= cOBWord and cOEWord <= cOEConnective:
                            l = [string.lower() for string in connective.split()]
                            if (word in l):
                                features.append((get_features(parsetree, tokenNo), 1))
                                break
                    else:
                        try:
                            # use connective candidate as negative example if there is no relation marked in PDTB
                            features.append((get_features(parsetree, tokenNo), 0))
                        # TODO remove later after fixing data pre-processing
                        except IndexError:
                            print(doc_id, connective_candidate, skip, tokenNo, word_idx)
    return list(zip(*features))


class ConnectiveClassifier:
    def __init__(self, n_estimators=1):
        if n_estimators > 1:
            self.model = Pipeline([
                ('vectorizer', DictVectorizer()),
                ('variance', VarianceThreshold(threshold=0.001)),
                ('selector', SelectKBest(mutual_info_classif, k=100)),
                ('model', BaggingClassifier(
                    base_estimator=SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100,
                                                 n_jobs=-1, class_weight='balanced', random_state=0),
                    n_estimators=n_estimators, max_samples=0.75, n_jobs=-1))
            ])
        else:
            self.model = Pipeline([
                ('vectorizer', DictVectorizer()),
                ('variance', VarianceThreshold(threshold=0.001)),
                ('selector', SelectKBest(mutual_info_classif, k=100)),
                ('model',
                 SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                               class_weight='balanced', random_state=0))
            ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'connective_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'connective_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)

    def score_on_features(self, X, y):
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Connective")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.score_on_features(X, y)

    def get_connective(self, parsetree, sentence, word_idx) -> (List[str], float):
        candidate = match_connectives(sentence, word_idx)
        if not candidate:
            return [], 1.0
        else:
            x = get_features(parsetree, list(range(word_idx, word_idx + len(candidate))))
            probs = self.model.predict_proba([x])[0]
            if probs.argmax() == 0:
                return [], probs[0]
            else:
                return candidate, probs[1]


if __name__ == "__main__":
    logger = init_logger()

    pdtb_train = [json.loads(s) for s in
                  open('/data/discourse/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('/data/discourse/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('/data/discourse/conll2016/en.test/relations.json', 'r').readlines()]
    parses_val = json.loads(open('/data/discourse/conll2016/en.test/parses.json').read())

    clf = ConnectiveClassifier()
    logger.info('Train model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
    clf.save('../tmp')
