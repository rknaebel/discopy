import json
import os
import pickle
from collections import Counter

import nltk


def get_production_rules(ptree):
    return [",".join([t.label()] + [tt.label() for tt in t]) for t in ptree.subtrees(lambda t: t.height() > 2)]


def extract_productions(ptrees):
    prod_rules = {production for ptree in ptrees for production in get_production_rules(ptree)}
    return prod_rules


def get_features(parse_trees, all_productions):
    productions = extract_productions(parse_trees)

    feat = {}
    for p in all_productions:
        feat[p] = str(p in productions)
    return feat


# {'AltLex': {'Comparison.Concession',
#             'Comparison.Contrast',
#             'Contingency.Cause.Reason',
#             'Contingency.Cause.Result',
#             'Contingency.Condition',
#             'Expansion',
#             'Expansion.Conjunction',
#             'Expansion.Exception',
#             'Expansion.Instantiation',
#             'Expansion.Restatement',
#             'Temporal.Asynchronous.Precedence',
#             'Temporal.Asynchronous.Succession',
#             'Temporal.Synchrony'},
#  'EntRel': {'EntRel'},
#  'Implicit': {'Comparison',
#               'Comparison.Concession',
#               'Comparison.Contrast',
#               'Contingency',
#               'Contingency.Cause',
#               'Contingency.Cause.Reason',
#               'Contingency.Cause.Result',
#               'Contingency.Condition',
#               'Expansion',
#               'Expansion.Alternative',
#               'Expansion.Alternative.Chosen alternative',
#               'Expansion.Conjunction',
#               'Expansion.Exception',
#               'Expansion.Instantiation',
#               'Expansion.Restatement',
#               'Temporal',
#               'Temporal.Asynchronous.Precedence',
#               'Temporal.Asynchronous.Succession',
#               'Temporal.Synchrony'}}
def generate_pdtb_features(pdtb, parses):
    extracted_productions = []
    for relation in filter(lambda i: i['Type'] != 'Explicit', pdtb):
        sentence_ids = {t[3] for a in ['Arg1', 'Arg2'] for t in relation[a]['TokenList']}
        doc = relation['DocID']

        parse_trees = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for sentence_id
                       in sentence_ids]
        sense = relation['Sense'][0]
        extracted_productions.append((extract_productions(parse_trees), sense))

    prod_counter = Counter(p for feature in extracted_productions for p in feature[0])
    all_productions = list(p for p, c in prod_counter.items() if c > 5)[:100]

    feature_set = []
    for (productions, sense) in extracted_productions:
        feat = {}
        for p in all_productions:
            feat[p] = str(p in productions)
        feature_set.append((feat, sense))

    return all_productions, feature_set


class NonExplicitSenseClassifier:
    def __init__(self):
        self.model = None
        self.prod_rules = set()

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))
        self.prod_rules = pickle.load(open(os.path.join(path, 'non_explicit_prod_rules.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))
        pickle.dump(self.prod_rules, open(os.path.join(path, 'non_explicit_prod_rules.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        self.prod_rules, features = generate_pdtb_features(pdtb, parses)
        self.fit_on_features(features, max_iter=max_iter)

    def fit_on_features(self, features, max_iter=5):
        self.model = nltk.MaxentClassifier.train(features, max_iter=max_iter)

    def predict(self, X):
        pass

    def get_sense(self, sents):
        features = get_features(sents, self.prod_rules)
        sense = self.model.prob_classify(features)
        return sense.max(), sense.prob(sense.max())


if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('../../discourse/data/conll2016/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('../../discourse/data/conll2016/en.dev/parses.json').read())

    all_productions, train_data = generate_pdtb_features(trainpdtb, trainparses)
    clf = NonExplicitSenseClassifier()
    clf.prod_rules = all_productions
    clf.fit_on_features(train_data)
    clf.save('tmp')
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, train_data)))
    print('....................................................................ON DEVELOPMENT DATA..................')
    val_data = generate_pdtb_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, val_data)))
