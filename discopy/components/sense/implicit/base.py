import logging
import os
import pickle
from collections import defaultdict
from typing import List

import click
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline, FeatureUnion

from discopy.components.component import Component
from discopy.data.doc import ParsedDocument
from discopy.data.sentence import DepRel
from discopy.data.relation import Relation
from discopy.data.loaders.conll import load_parsed_conll_dataset
from discopy.utils import ItemSelector, init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')


def get_production_rules(ptree: nltk.ParentedTree):
    return ["{} <- {}".format(t.label(), ' '.join([tt.label() for tt in t]))
            for t in ptree.subtrees(lambda t: t.height() > 2)]


def extract_productions(ptrees: List[nltk.ParentedTree]):
    return [production for ptree in ptrees if ptree for production in get_production_rules(ptree)]


def get_dependencies(dtree: List[DepRel]):
    feat = defaultdict(list)
    for deprel in dtree:
        feat[deprel.head.surface if deprel.head else "ROOT"].append(deprel.rel)
    return ["{} <- {}".format(lemmatizer.lemmatize(word).lower(), ' '.join(deps)) for word, deps in feat.items()]


def extract_dependencies(dtrees: List[List[DepRel]]):
    return [deps for dtree in dtrees if dtree for deps in get_dependencies(dtree)]


def get_features(ptrees_prev: List[nltk.ParentedTree], ptrees: List[nltk.ParentedTree],
                 dtrees_prev: List[List[DepRel]], dtrees: List[List[DepRel]],
                 arg1: List[str], arg2: List[str]):
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


def generate_pdtb_features(docs: List[ParsedDocument]):
    features = []
    for doc in docs:
        for relation in filter(lambda r: r.type != "Explicit", doc.relations):
            arg1_sentence_ids = relation.arg1.get_sentence_idxs()
            arg2_sentence_ids = relation.arg2.get_sentence_idxs()
            arg1_parse_trees = [doc.sentences[i].get_ptree() for i in arg1_sentence_ids]
            arg2_parse_trees = [doc.sentences[i].get_ptree() for i in arg2_sentence_ids]
            arg1_dep_trees = [doc.sentences[i].dependencies for i in arg1_sentence_ids]
            arg2_dep_trees = [doc.sentences[i].dependencies for i in arg2_sentence_ids]
            arg1 = [t.surface for t in relation.arg1.tokens]
            arg2 = [t.surface for t in relation.arg2.tokens]
            features.append((
                get_features(arg1_parse_trees, arg2_parse_trees, arg1_dep_trees, arg2_dep_trees, arg1, arg2),
                relation.senses[0])
            )

    return list(zip(*features))


class NonExplicitSenseClassifier(Component):
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
            ('model',
             SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                           class_weight='balanced', random_state=0))
        ])

    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))

    def fit(self, docs_train: List[ParsedDocument], docs_val: List[ParsedDocument] = None):
        x, y = generate_pdtb_features(docs_train)
        self.model.fit(x, y)

    def score_on_features(self, x, y):
        y_pred = self.model.predict_proba(x)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Sense(non-explicit)")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, docs: List[ParsedDocument]):
        logger.debug('Extract features')
        x, y = generate_pdtb_features(docs)
        self.score_on_features(x, y)

    def get_sense(self, arg1_parse_trees, arg2_parse_trees, arg1_dep_trees, arg2_dep_trees, arg1, arg2):
        x = get_features(arg1_parse_trees, arg2_parse_trees, arg1_dep_trees, arg2_dep_trees, arg1, arg2)
        probs = self.model.predict_proba([x])[0]
        r_sense = self.model.classes_[probs.argmax()]
        return r_sense, probs.max()

    def parse(self, doc: ParsedDocument, relations: List[Relation] = None, **kwargs):
        if relations is None:
            raise ValueError('Component needs implicit arguments extracted.')
        for relation in filter(lambda r: r.type != "Explicit", relations):
            arg1_sentence_ids = relation.arg1.get_sentence_idxs()
            arg2_sentence_ids = relation.arg2.get_sentence_idxs()
            arg1_ptrees = [doc.sentences[i].get_ptree() for i in arg1_sentence_ids]
            arg2_ptrees = [doc.sentences[i].get_ptree() for i in arg2_sentence_ids]
            arg1_dtrees = [doc.sentences[i].dependencies for i in arg1_sentence_ids]
            arg2_dtrees = [doc.sentences[i].dependencies for i in arg2_sentence_ids]
            arg1 = [t.surface for t in relation.arg1.tokens]
            arg2 = [t.surface for t in relation.arg2.tokens]
            if all(ptree is None for ptree in arg1_ptrees) or all(ptree is None for ptree in arg2_ptrees):
                continue
            sense, sense_c = self.get_sense(arg1_ptrees, arg2_ptrees, arg1_dtrees, arg2_dtrees, arg1, arg2)
            relation.senses = [sense]
            if relation.senses[0] in ('EntRel', 'NoRel'):
                relation.type = relation.senses[0]
            else:
                relation.type = 'Implicit'
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_parsed_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = NonExplicitSenseClassifier()
    logger.info('Train model')
    clf.fit(docs_train)
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    print(clf.parse(docs_val[0], docs_val[0].relations, ))


if __name__ == "__main__":
    main()
