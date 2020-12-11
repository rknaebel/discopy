import logging
import os
import pickle
from typing import List

import click
import nltk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn_crfsuite import CRF

from discopy.components.argument.position import ArgumentPositionClassifier
from discopy.components.component import Component
from discopy.data.doc import Document
from discopy.data.relation import Relation
from discopy.data.loaders.conll import load_conll_dataset
from discopy.utils import init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')


def get_features(ptree: nltk.ParentedTree, conn_idxs: List[int]):
    features = []
    for i, (word, tag) in enumerate(ptree.pos()):
        features.append({
            'BOS': i == 0,
            'word': word,
            'pos': tag,
            'lemma': lemmatizer.lemmatize(word),
            'stem': stemmer.stem(word),
            'conn': i in conn_idxs
        })
    return features


def generate_pdtb_features(docs: List[Document]):
    ss_features = []
    ps_features = []
    for doc in docs:
        for relation in filter(lambda r: r.type == 'Explicit', doc.relations):
            arg1_sentence_id = relation.arg1.get_sentence_idxs()[0]
            arg2_sentence_id = relation.arg2.get_sentence_idxs()[0]
            ptree = doc.sentences[arg2_sentence_id].get_ptree()
            if ptree is None:
                continue
            arg1 = set(t.local_idx for t in relation.arg1.tokens)
            arg2 = set(t.local_idx for t in relation.arg2.tokens)
            conn = [t.local_idx for t in relation.conn.tokens]
            labels = []
            sent_features = get_features(ptree, conn)
            # Arg1 is in the same sentence (SS)
            if arg1_sentence_id == arg2_sentence_id:
                for i, _ in enumerate(sent_features):
                    if i in arg1:
                        labels.append("Arg1")
                    elif i in arg2:
                        labels.append("Arg2")
                    else:
                        labels.append("NULL")
                ss_features.append((sent_features, labels))
            # Arg1 is in the previous sentence (PS)
            elif (arg2_sentence_id - arg1_sentence_id) == 1:
                for i, _ in enumerate(sent_features):
                    if i in arg2:
                        labels.append("Arg2")
                    else:
                        labels.append("NULL")
                ps_features.append((sent_features, labels))
    return list(zip(*ss_features)), list(zip(*ps_features))


class CRFArgumentExtractor(Component):
    def __init__(self):
        self.id = "crf_arg_extract"
        self.arg_pos_clf = ArgumentPositionClassifier()
        self.ss_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
        self.ps_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

    def load(self, path):
        self.arg_pos_clf.load(path)
        self.ss_model = pickle.load(open(os.path.join(path, "{}.ss.p".format(self.id)), 'rb'))
        self.ps_model = pickle.load(open(os.path.join(path, "{}.ps.p".format(self.id)), 'rb'))

    def save(self, path):
        self.arg_pos_clf.save(path)
        pickle.dump(self.ss_model, open(os.path.join(path, "{}.ss.p".format(self.id)), 'wb'))
        pickle.dump(self.ps_model, open(os.path.join(path, "{}.ps.p".format(self.id)), 'wb'))

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        self.arg_pos_clf.fit(docs_train)
        (x_ss, y_ss), (x_ps, y_ps) = generate_pdtb_features(docs_train)
        self.ss_model.fit(x_ss, y_ss)
        self.ps_model.fit(x_ps, y_ps)

    def score_on_features(self, x_ss, y_ss, x_ps, y_ps):
        y_pred = np.concatenate(self.ss_model.predict(x_ss))
        y_ss = np.concatenate(y_ss)
        logger.info("Evaluation: SS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ss, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ss, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ss, y_pred)))

        y_pred = np.concatenate(self.ps_model.predict(x_ps))
        y_ps = np.concatenate(y_ps)
        logger.info("Evaluation: PS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ps, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ps, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ps, y_pred)))

    def score(self, docs: List[Document]):
        self.arg_pos_clf.score(docs)
        (x_ss, y_ss), (x_ps, y_ps) = generate_pdtb_features(docs)
        self.score_on_features(x_ss, y_ss, x_ps, y_ps)

    def extract_arguments(self, ptree: nltk.ParentedTree, relation: Relation, arg_pos: str):
        indices = [token.local_idx for token in relation.conn.tokens]
        ptree._label = 'S'
        x = get_features(ptree, indices)
        if arg_pos == 'SS':
            probs = np.array(
                [[p[c] for c in self.ss_model.classes_] for p in self.ss_model.predict_marginals_single(x)])
            probs_max = probs.max(1)
            labels = np.array(self.ss_model.classes_)[probs.argmax(axis=1)]
            arg1 = np.where(labels == 'Arg1')[0]
            arg2 = np.where(labels == 'Arg2')[0]
            arg1_prob = probs_max[arg1].mean() if len(arg1) else 0.0
            arg2_prob = probs_max[arg2].mean() if len(arg2) else 0.0
            arg1, arg2 = arg1.tolist(), arg2.tolist()
            if not arg1:
                logger.warning("Empty Arg1")
            if not arg2:
                logger.warning("Empty Arg2")
        elif arg_pos == 'PS':
            probs = np.array(
                [[p[c] for c in self.ps_model.classes_] for p in self.ps_model.predict_marginals_single(x)])
            probs_max = probs.max(1)
            labels = np.array(self.ps_model.classes_)[probs.argmax(axis=1)]
            arg1 = []
            arg1_prob = 1.0
            arg2 = np.where(labels == 'Arg2')[0]
            arg2_prob = probs_max[arg2].mean() if len(arg2) else 0.0
            arg2 = arg2.tolist()
            if not arg2:
                logger.warning("Empty Arg2")
        else:
            raise NotImplementedError('Unknown argument position')

        return arg1, arg2, arg1_prob, arg2_prob

    def parse(self, doc: Document, relations: List[Relation] = None):
        if relations is None:
            raise ValueError('Component needs connectives already classified.')
        for relation in filter(lambda r: r.type == "Explicit", relations):
            sent_id = relation.conn.get_sentence_idxs()[0]
            sent = doc.sentences[sent_id]
            ptree = sent.get_ptree()
            if ptree is None or len(relation.conn.tokens) == 0:
                continue
            # ARGUMENT POSITION
            conn_raw = ' '.join(t.surface for t in relation.conn.tokens)
            conn_idxs = [t.local_idx for t in relation.conn.tokens]
            arg_pos, arg_pos_confidence = self.arg_pos_clf.get_argument_position(ptree, conn_raw, conn_idxs)
            # If position poorly classified as PS, go to the next relation
            if arg_pos == 'PS' and sent_id == 0:
                continue
            # ARGUMENT EXTRACTION
            arg1, arg2, arg1_c, arg2_c = self.extract_arguments(ptree, relation, arg_pos)
            if arg_pos == 'PS':
                prev_sent = doc.sentences[sent_id]
                relation.arg1.tokens = prev_sent.tokens
                relation.arg2.tokens = [sent.tokens[i] for i in arg2]
            elif arg_pos == 'SS':
                relation.arg1.tokens = [sent.tokens[i] for i in arg1]
                relation.arg2.tokens = [sent.tokens[i] for i in arg2]
            else:
                logger.error('Unknown Argument Position: ' + arg_pos)
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_train = load_conll_dataset(os.path.join(conll_path, 'en.train'))
    docs_val = load_conll_dataset(os.path.join(conll_path, 'en.dev'))

    clf = CRFArgumentExtractor()
    logger.info('Train model')
    clf.fit(docs_train)
    logger.info('Evaluation on TRAIN')
    clf.score(docs_train)
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    print(clf.parse(docs_val[0], []))


if __name__ == "__main__":
    main()
