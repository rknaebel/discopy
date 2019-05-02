import os
import pickle
import ujson as json
from collections import defaultdict

import nltk
import sklearn
import sklearn.pipeline


def get_production_rules(ptree):
    return ["{} <- {}".format(t.label(), ' '.join([tt.label() for tt in t]))
            for t in ptree.subtrees(lambda t: t.height() > 2)]


def extract_productions(ptrees):
    return [production for ptree in ptrees for production in get_production_rules(ptree)]


def get_dependencies(dtree):
    feat = defaultdict(list)
    for (d, n1, n2) in dtree:
        feat[n1[:n1.rfind('-')]].append(d)
    return ["{} <- {}".format(word, ' '.join(deps)) for word, deps in feat.items()]


def extract_dependencies(dtrees):
    return [deps for dtree in dtrees for deps in get_dependencies(dtree)]


def get_features(ptrees_prev, ptrees, dtrees_prev, dtrees):
    productions_prev = set(extract_productions(ptrees_prev))
    productions = set(extract_productions(ptrees))

    features = {}
    for r in productions | productions_prev:
        if r in productions_prev and r in productions:
            features[r] = 3
        elif r in productions_prev:
            features[r] = 1
        elif r in productions:
            features[r] = 2

    deps_prev = set(extract_dependencies(dtrees_prev))
    deps = set(extract_dependencies(dtrees))

    for d in deps | deps_prev:
        if d in deps_prev and d in deps:
            features[d] = 3
        elif d in deps_prev:
            features[d] = 1
        elif d in deps:
            features[d] = 2

    return features


def generate_pdtb_features(pdtb, parses):
    features = []
    for relation in filter(lambda i: i['Type'] != 'Explicit', pdtb):
        arg1_sentence_ids = {t[3] for t in relation['Arg1']['TokenList']}
        arg2_sentence_ids = {t[3] for t in relation['Arg2']['TokenList']}
        doc = relation['DocID']
        try:
            arg1_parse_trees = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for
                                sentence_id in arg1_sentence_ids]
            arg2_parse_trees = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for
                                sentence_id in arg2_sentence_ids]

            arg1_dep_trees = [parses[doc]['sentences'][s_id]['dependencies'] for s_id in arg1_sentence_ids]
            arg2_dep_trees = [parses[doc]['sentences'][s_id]['dependencies'] for s_id in arg2_sentence_ids]
        except ValueError:
            continue
        sense = relation['Sense'][0]
        features.append((get_features(arg1_parse_trees, arg2_parse_trees, arg1_dep_trees, arg2_dep_trees), sense))

    return list(zip(*features))


class NonExplicitSenseClassifier:
    def __init__(self):
        self.model = sklearn.pipeline.Pipeline([
            ('vectorizer', sklearn.feature_extraction.DictVectorizer()),
            ('selector', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=200)),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=-1))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)
        print("Acc:", self.model.score(X, y))

    def get_sense(self, sents_prev, sents, dtree_prev, dtree):
        x = get_features([sents_prev], [sents], [dtree_prev], [dtree])
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
