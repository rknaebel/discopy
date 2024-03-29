import logging
import os
import pickle
from typing import List

import click
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.pipeline import Pipeline

from discopy.components.component import Component
from discopy.features import get_connective_sentence_position, lca
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_parsed_conll_dataset
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_features(relation: Relation, ptree: nltk.ParentedTree):
    conn_raw = ' '.join(t.surface for t in relation.conn.tokens)
    conn_idxs = [t.local_idx for t in relation.conn.tokens]

    lca_loc = lca(ptree, conn_idxs)
    conn_tag = ptree[lca_loc].label()

    if conn_idxs[0] == 0:
        prev = "NONE"
    else:
        prev = ptree.leaves()[conn_idxs[0] - 1][0]
        prev = lemmatizer.lemmatize(prev)

    conn_pos_relative = get_connective_sentence_position(conn_idxs, ptree)

    feat = {'Connective': conn_raw,
            'ConnectivePOS': conn_tag,
            'ConnectivePrev': prev, 'connectivePosition': conn_pos_relative}
    return feat


def generate_pdtb_features(docs: List[Document]):
    features = []
    for doc in docs:
        for relation in filter(lambda r: r.type == 'Explicit', doc.relations):
            sent_i = relation.conn.get_sentence_idxs()[0]
            sense = relation.senses[0]
            ptree = doc.sentences[sent_i].get_ptree()
            if not ptree:
                continue
            features.append((get_features(relation, ptree), sense))
    return list(zip(*features))


class ExplicitSenseClassifier(Component):
    model_name = 'explicit_sense_base_classifier'
    used_features = ['ptree']

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

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        x, y = generate_pdtb_features(docs_train)
        self.model.fit(x, y)

    def score_on_features(self, x, y):
        y_pred = self.model.predict_proba(x)
        y_pred_c = self.model.classes_[y_pred.argmax(axis=1)]
        logger.info("Evaluation: Sense(explicit)")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred_c)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred_c, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred_c)))

    def score(self, docs):
        x, y = generate_pdtb_features(docs)
        self.score_on_features(x, y)

    def get_sense(self, relation: Relation, ptree: nltk.ParentedTree):
        x = get_features(relation, ptree)
        probs = self.model.predict_proba([x])[0]
        return self.model.classes_[probs.argmax()], probs.max()

    def parse(self, doc: Document, relations: List[Relation] = None, **kwargs):
        if relations is None:
            raise ValueError('Component needs connectives already classified.')
        for relation in filter(lambda r: r.type == "Explicit", relations):
            sent_id = relation.conn.get_sentence_idxs()[0]
            ptree = doc.sentences[sent_id].get_ptree()
            if ptree is None:
                logger.warning('Failed on empty tree')
                continue
            sense, sense_c = self.get_sense(relation, ptree)
            relation.senses.append(sense)
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_parsed_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = ExplicitSenseClassifier()
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
