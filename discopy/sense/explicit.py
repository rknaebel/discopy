import logging
import os
import pickle
import sys

import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.data.conll16 import get_conll_dataset
from discopy.features import get_connective_sentence_position, lca
from discopy.utils import preprocess_relations, init_logger

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
        ptree = parses[doc]['sentences'][sentenceOffSet]['parsetree']
        if not ptree:
            continue
        features.append((get_features(relation, ptree), sense))
    return list(zip(*features))


class ExplicitSenseClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('variance', VarianceThreshold(threshold=0.0001)),
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

    def score_on_features(self, X, y):
        y_pred = self.model.predict_proba(X)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Sense(explicit)")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses, filters=False)
        self.score_on_features(X, y)

    def get_sense(self, relation, ptree):
        x = get_features(relation, ptree)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()


if __name__ == "__main__":
    logger = init_logger()

    data_path = sys.argv[1]
    parses_train, pdtb_train = get_conll_dataset(data_path, 'en.train', load_trees=True, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(data_path, 'en.dev', load_trees=True, connective_mapping=True)

    clf = ExplicitSenseClassifier()
    logger.info('Train model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
