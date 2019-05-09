import logging
import os
import pickle
import ujson as json
from collections import defaultdict

import nltk
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.pipeline import Pipeline, FeatureUnion

from discopy.utils import ItemSelector, preprocess_relations

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')


def get_production_rules(ptree):
    return ["{} <- {}".format(t.label(), ' '.join([tt.label() for tt in t]))
            for t in ptree.subtrees(lambda t: t.height() > 2)]


def extract_productions(ptrees):
    return [production for ptree in ptrees for production in get_production_rules(ptree)]


def get_dependencies(dtree):
    feat = defaultdict(list)
    for (d, n1, n2) in dtree:
        feat[n1[:n1.rfind('-')]].append(d)
    return ["{} <- {}".format(lemmatizer.lemmatize(word).lower(), ' '.join(deps)) for word, deps in feat.items()]


def extract_dependencies(dtrees):
    return [deps for dtree in dtrees for deps in get_dependencies(dtree)]


def get_features(ptrees_prev, ptrees, dtrees_prev, dtrees, arg1, arg2):
    productions_prev = set(extract_productions(ptrees_prev))
    productions = set(extract_productions(ptrees))

    features = {
        'prod': {},
        'deps': {},
        'word_pairs': {}
    }
    for r in productions | productions_prev:
        if r in productions_prev and r in productions:
            features['prod'][r] = 'both'
        elif r in productions_prev:
            features['prod'][r] = 'arg1'
        elif r in productions:
            features['prod'][r] = 'arg2'

    deps_prev = set(extract_dependencies(dtrees_prev))
    deps = set(extract_dependencies(dtrees))

    for d in deps | deps_prev:
        if d in deps_prev and d in deps:
            features['deps'][d] = 'both'
        elif d in deps_prev:
            features['deps'][d] = 'arg1'
        elif d in deps:
            features['deps'][d] = 'arg2'

    for w1 in arg1:
        for w2 in arg2:
            features['word_pairs']["({},{})".format(stemmer.stem(w1), stemmer.stem(w2))] = 1

    return features


def generate_pdtb_features(pdtb, parses):
    features = []
    pdtb = preprocess_relations(list(filter(lambda i: i['Type'] != 'Explicit', pdtb)))
    for relation in pdtb:
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
        arg1 = [parses[doc]['sentences'][t[3]]['words'][t[4]][0] for t in relation['Arg1']['TokenList']]
        arg2 = [parses[doc]['sentences'][t[3]]['words'][t[4]][0] for t in relation['Arg2']['TokenList']]
        sense = relation['Sense'][0]
        features.append(
            (get_features(arg1_parse_trees, arg2_parse_trees, arg1_dep_trees, arg2_dep_trees, arg1, arg2), sense))

    return list(zip(*features))


class NonExplicitSenseClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('union', FeatureUnion([
                ('productions', Pipeline([
                    ('selector', ItemSelector(key='prod')),
                    ('vectorizer', DictVectorizer()),
                    ('variance', VarianceThreshold(threshold=0.001)),
                    ('reduce', SelectKBest(mutual_info_classif, k=100))
                ])),
                ('dependecies', Pipeline([
                    ('selector', ItemSelector(key='deps')),
                    ('vectorizer', DictVectorizer()),
                    ('variance', VarianceThreshold(threshold=0.001)),
                    ('reduce', SelectKBest(mutual_info_classif, k=100))
                ])),
                ('word_pairs', Pipeline([
                    ('selector', ItemSelector(key='word_pairs')),
                    ('vectorizer', DictVectorizer()),
                    ('variance', VarianceThreshold(threshold=0.005)),
                    ('reduce', SelectKBest(mutual_info_classif, k=500))
                ]))
            ])),
            ('model', sklearn.linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=-1,
                                                              max_iter=200))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)
        logger.info("Acc: {}".format(self.model.score(X, y)))

    def get_sense(self, sents_prev, sents, dtree_prev, dtree, arg1, arg2):
        x = get_features([sents_prev], [sents], [dtree_prev], [dtree], arg1, arg2)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


if __name__ == "__main__":
    pdtb_train = [json.loads(s) for s in
                  open('../../discourse/data/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('../../discourse/data/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('../../discourse/data/conll2016/en.test/relations.json', 'r').readlines()]
    parses_val = json.loads(open('../../discourse/data/conll2016/en.test/parses.json').read())

    clf = NonExplicitSenseClassifier()
    print("generate features")
    X, y = generate_pdtb_features(pdtb_train, parses_train)
    print('train model')
    clf.model.fit(X, y)
    print('Evaluation on TRAIN')
    print('ACCURACY {}'.format(clf.model.score(X, y)))
    print('Evaluation on TEST')
    X_val, y_val = generate_pdtb_features(pdtb_val, parses_val)
    print('ACCURACY {}'.format(clf.model.score(X_val, y_val)))
    clf.save('../tmp')
