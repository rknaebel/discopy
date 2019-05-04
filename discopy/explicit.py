import json
import os
import pickle

import nltk
import sklearn
import sklearn.pipeline

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.features import get_connective_sentence_position, lca

lemmatizer = nltk.stem.WordNetLemmatizer()


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
        prev = lemmatizer.lemmatize(prev)

    conn_pos_relative = get_connective_sentence_position(connective_head_index, ptree)

    feat = {'Connective': connective, 'connectiveHead': head,
            'ConnectivePOS': conn_tag,
            'ConnectivePrev': prev, 'connectivePosition': conn_pos_relative}
    return feat


def generate_pdtb_features(pdtb, parses):
    features = []
    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        sentenceOffSet = relation['Connective']['TokenList'][0][3]
        doc = relation['DocID']
        sense = relation['Sense'][0]
        try:
            ptree = nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentenceOffSet]['parsetree'])
        except ValueError:
            continue
        if not ptree.leaves():
            continue
        features.append((get_features(relation, ptree), sense))
    return list(zip(*features))


class ExplicitSenseClassifier:
    def __init__(self):
        self.model = sklearn.pipeline.Pipeline([
            ('vectorizer', sklearn.feature_extraction.DictVectorizer()),
            ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=100)),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=-1,
                                                              max_iter=200))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)
        print("Acc:", self.model.score(X, y))

    def get_sense(self, relation, ptree):
        x = get_features(relation, ptree)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


if __name__ == "__main__":
    pdtb_train = [json.loads(s) for s in
                  open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    parses_val = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = ExplicitSenseClassifier()
    X, y = generate_pdtb_features(pdtb_train, parses_train)
    clf.model.fit(X, y)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(clf.model.score(X, y)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    X_val, y_val = generate_pdtb_features(pdtb_val, parses_val)
    print('ACCURACY {}'.format(clf.model.score(X_val, y_val)))
    clf.save('../tmp')
