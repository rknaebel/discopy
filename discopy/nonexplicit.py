import os
import pickle
import ujson as json
from collections import Counter

import nltk
import sklearn
import sklearn.pipeline


def get_production_rules(ptree):
    return [",".join([t.label()] + [tt.label() for tt in t]) for t in ptree.subtrees(lambda t: t.height() > 2)]


def extract_productions(ptrees):
    return [production for ptree in ptrees for production in get_production_rules(ptree)]


def get_features(parse_trees):
    productions = set(extract_productions(parse_trees))
    return productions


def generate_production_rules(pdtb, parses, min_occurrence=5):
    extracted_productions = []
    for relation in filter(lambda i: i['Type'] != 'Explicit', pdtb):
        sentence_ids = {t[3] for a in ['Arg1', 'Arg2'] for t in relation[a]['TokenList']}
        doc = relation['DocID']
        try:
            parse_trees = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for
                           sentence_id
                           in sentence_ids]
        except ValueError:
            continue
        extracted_productions.extend(extract_productions(parse_trees))

    prod_counter = Counter(extracted_productions)
    all_productions = list(p for p, c in prod_counter.items() if c > min_occurrence)
    return all_productions


def generate_pdtb_features(pdtb, parses):
    features = []
    for relation in filter(lambda i: i['Type'] != 'Explicit', pdtb):
        sentence_ids = {t[3] for a in ['Arg1', 'Arg2'] for t in relation[a]['TokenList']}
        doc = relation['DocID']
        try:
            parse_trees = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for
                           sentence_id
                           in sentence_ids]
        except ValueError:
            continue
        sense = relation['Sense'][0]
        features.append((get_features(parse_trees), sense))

    return list(zip(*features))


class NonExplicitSenseClassifier:
    def __init__(self):
        self.model = sklearn.pipeline.Pipeline([
            ('vectorizer', sklearn.feature_extraction.text.CountVectorizer(analyzer=set)),
            ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=100)),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=-1))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)

    def get_sense(self, sents):
        x = get_features(sents)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


if __name__ == "__main__":
    pdtb_train = [json.loads(s) for s in
                  open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    parses_val = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    print('....................................................................TRAINING..................')
    clf = NonExplicitSenseClassifier()
    X, y = generate_pdtb_features(pdtb_train, parses_train)
    clf.model.fit(X, y)
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(clf.model.score(X, y)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    X_val, y_val = generate_pdtb_features(pdtb_val, parses_val)
    print('ACCURACY {}'.format(clf.model.score(X_val, y_val)))
    clf.save('../tmp')
