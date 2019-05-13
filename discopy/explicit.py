import json
import logging
import os
import pickle

import nltk
from discopy.conn_head_mapper import ConnHeadMapper
from discopy.features import get_connective_sentence_position, lca
from discopy.utils import preprocess_relations
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger('discopy')

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


def generate_pdtb_features(pdtb, parses, filters=True):
    features = []
    pdtb = preprocess_relations(list(filter(lambda i: i['Type'] == 'Explicit', pdtb)), filters=filters)
    for relation in pdtb:
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
    def __init__(self, n_estimators=1):
        if n_estimators > 1:
            self.model = Pipeline([
                ('vectorizer', DictVectorizer()),
                ('bagging', BaggingClassifier(base_estimator=Pipeline([
                    ('variance', VarianceThreshold(threshold=0.001)),
                    ('selector', SelectKBest(mutual_info_classif, k=100)),
                    ('model', SGDClassifier(loss='log', penalty='l2', average=32, tol=1e-3, max_iter=100, n_jobs=-1,
                                            class_weight='balanced', random_state=0))
                ]), n_estimators=n_estimators, max_samples=0.75, n_jobs=-1))
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
        self.model = pickle.load(open(os.path.join(path, 'explicit_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'explicit_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses)
        self.model.fit(X, y)

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses, filters=False)
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Sense(explicit)")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

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
