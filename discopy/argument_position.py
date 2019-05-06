import logging
import os
import pickle
import ujson as json

import nltk
import sklearn
import sklearn.pipeline
from nltk.tree import ParentedTree

import discopy.conn_head_mapper
from discopy.features import get_connective_sentence_position, lca, get_pos_features

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_features(ptree: ParentedTree, connective: str, leaf_index: list):
    chm = discopy.conn_head_mapper.ConnHeadMapper()
    head, connective_head_index = chm.map_raw_connective(connective)
    connective_head_index = [leaf_index[i] for i in connective_head_index]

    lca_loc = lca(ptree, leaf_index)
    conn_tag = ptree[lca_loc].label()
    conn_pos_relative = get_connective_sentence_position(connective_head_index, ptree)

    prev, prev_conn, prev_pos, prev_pos_conn_pos = get_pos_features(ptree, leaf_index, head, -1)
    prev2, prev2_conn, prev2_pos, prev2_pos_conn_pos = get_pos_features(ptree, leaf_index, head, -2)

    prev = lemmatizer.lemmatize(prev)
    prev2 = lemmatizer.lemmatize(prev2)

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
        try:
            ptree = nltk.ParentedTree.fromstring(ptree)
        except ValueError:
            continue
        if not ptree.leaves():
            continue

        arg1 = sorted({i[3] for i in relation['Arg1']['TokenList']})
        arg2 = sorted({i[3] for i in relation['Arg2']['TokenList']})
        if not arg1 or not arg2:
            continue
        if arg1[-1] < arg2[0]:
            features.append((get_features(ptree, connective_raw, leaf_indices), 'PS'))
        elif len(arg1) == 1 and len(arg2) == 1 and arg1[0] == arg2[0]:
            features.append((get_features(ptree, connective_raw, leaf_indices), 'SS'))
    return list(zip(*features))


class ArgumentPositionClassifier:
    def __init__(self):
        self.model = sklearn.pipeline.Pipeline([
            ('vectorizer', sklearn.feature_extraction.DictVectorizer()),
            ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=100)),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=200, n_jobs=-1))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'position_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'position_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)
        logger.info("Acc: {}".format(self.model.score(X, y)))

    def get_argument_position(self, parse, connective: str, leaf_index):
        x = get_features(parse, connective, leaf_index)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


if __name__ == "__main__":
    pdtb_train = [json.loads(s) for s in
                  open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    parses_val = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = ArgumentPositionClassifier()
    X, y = generate_pdtb_features(pdtb_train, parses_train)
    clf.model.fit(X, y)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(clf.model.score(X, y)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    X_val, y_val = generate_pdtb_features(pdtb_val, parses_val)
    print('ACCURACY {}'.format(clf.model.score(X_val, y_val)))
    clf.save('../tmp')
