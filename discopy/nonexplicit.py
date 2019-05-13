import logging
import os
import pickle
import ujson as json
from collections import defaultdict

import nltk
from discopy.utils import ItemSelector, preprocess_relations
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline, FeatureUnion

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


def generate_pdtb_features(pdtb, parses, filters=True):
    features = []
    pdtb = preprocess_relations(list(filter(lambda i: i['Type'] != 'Explicit', pdtb)), filters=filters)
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
        sense = "{}--{}".format(relation['Type'], relation['Sense'][0])
        features.append(
            (get_features(arg1_parse_trees, arg2_parse_trees, arg1_dep_trees, arg2_dep_trees, arg1, arg2), sense))

    return list(zip(*features))


class NonExplicitSenseClassifier:
    def __init__(self, n_estimators=1):
        if n_estimators > 1:
            self.model = Pipeline([
                ('union', FeatureUnion([
                    ('productions', Pipeline([
                        ('selector', ItemSelector(key='prod')),
                        ('vectorizer', DictVectorizer()),
                        ('variance', VarianceThreshold(threshold=0.0005)),
                        ('reduce', SelectKBest(mutual_info_classif, k=100))
                    ])),
                    ('dependecies', Pipeline([
                        ('selector', ItemSelector(key='deps')),
                        ('vectorizer', DictVectorizer()),
                        ('variance', VarianceThreshold(threshold=0.0005)),
                        ('reduce', SelectKBest(mutual_info_classif, k=100))
                    ])),
                    ('word_pairs', Pipeline([
                        ('selector', ItemSelector(key='word_pairs')),
                        ('vectorizer', DictVectorizer()),
                        ('variance', VarianceThreshold(threshold=0.0001)),
                        ('reduce', SelectKBest(mutual_info_classif, k=500))
                    ]))
                ])),
                ('bagging', BaggingClassifier(base_estimator=Pipeline([
                    ('model', SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                                            class_weight='balanced', random_state=0))
                ]), n_estimators=n_estimators, max_samples=0.75, n_jobs=-1))
            ])
        else:
            self.model = Pipeline([
                ('union', FeatureUnion([
                    ('productions', Pipeline([
                        ('selector', ItemSelector(key='prod')),
                        ('vectorizer', DictVectorizer()),
                        ('variance', VarianceThreshold(threshold=0.0005)),
                        ('reduce', SelectKBest(mutual_info_classif, k=100))
                    ])),
                    ('dependecies', Pipeline([
                        ('selector', ItemSelector(key='deps')),
                        ('vectorizer', DictVectorizer()),
                        ('variance', VarianceThreshold(threshold=0.0005)),
                        ('reduce', SelectKBest(mutual_info_classif, k=100))
                    ])),
                    ('word_pairs', Pipeline([
                        ('selector', ItemSelector(key='word_pairs')),
                        ('vectorizer', DictVectorizer()),
                        ('variance', VarianceThreshold(threshold=0.0001)),
                        ('reduce', SelectKBest(mutual_info_classif, k=500))
                    ]))
                ])),
                ('model',
                 SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                               class_weight='balanced', random_state=0))
            ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses, filters=False)
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Sense(non-explicit)")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def get_sense(self, sents_prev, sents, dtree_prev, dtree, arg1, arg2):
        x = get_features([sents_prev], [sents], [dtree_prev], [dtree], arg1, arg2)
        probs = self.model.predict_proba([x])[0]
        r_type, r_sense = self.model.classes_[probs.argmax()].split('--')
        return r_sense, probs.max()


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
