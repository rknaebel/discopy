import copy
import logging
import os
import pickle
import ujson as json

import nltk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn_crfsuite import CRF

from discopy.utils import init_logger

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')


def get_features(ptree, indices):
    features = []
    for i, (word, tag) in enumerate(ptree.pos()):
        features.append({
            'BOS': i == 0,
            'word': word,
            'pos': tag,
            'lemma': lemmatizer.lemmatize(word),
            'stem': stemmer.stem(word),
            'conn': i in indices
        })
    return features


def generate_pdtb_features_crf(pdtb, parses):
    ss_features = []
    ps_features = []

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc_id = relation['DocID']
        try:
            arg1_sentence_id = relation['Arg1']['TokenList'][0][3]
            arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
            s = parses[doc_id]['sentences'][arg2_sentence_id]['parsetree']
            ptree = nltk.ParentedTree.fromstring(s)
        except ValueError:
            continue
        except IndexError:
            continue

        if not ptree.leaves():
            continue

        arg1 = set([i[4] for i in relation['Arg1']['TokenList']])
        arg2 = set([i[4] for i in relation['Arg2']['TokenList']])

        conn = [token[4] for token in relation['Connective']['TokenList']]

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


def generate_pdtb_features_gosh(pdtb, parses):
    arg1_features = []
    arg2_features = []

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc_id = relation['DocID']
        try:
            arg1_sentence_id = relation['Arg1']['TokenList'][0][3]
            arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
            s = parses[doc_id]['sentences'][arg2_sentence_id]['parsetree']
            ptree = nltk.ParentedTree.fromstring(s)
        except ValueError:
            continue
        except IndexError:
            continue

        if not ptree.leaves():
            continue

        arg1 = set([i[4] for i in relation['Arg1']['TokenList']])
        arg2 = set([i[4] for i in relation['Arg2']['TokenList']])
        conn = [token[4] for token in relation['Connective']['TokenList']]

        sent_features = get_features(ptree, conn)

        # Arg1 is in the same sentence (SS)
        if arg1_sentence_id == arg2_sentence_id:
            X_arg1 = copy.deepcopy(sent_features)
            X_arg2 = copy.deepcopy(sent_features)
            labels_arg1 = []
            labels_arg2 = []
            for i, word in enumerate(X_arg1):
                if i in arg1:
                    labels_arg1.append("Arg1")
                    labels_arg2.append("NULL")
                    word['is_arg2'] = False
                elif i in arg2:
                    labels_arg1.append("NULL")
                    labels_arg2.append("Arg2")
                    word['is_arg2'] = True
                else:
                    labels_arg1.append("NULL")
                    labels_arg2.append("NULL")
                    word['is_arg2'] = False
            arg1_features.append((X_arg1, labels_arg1))
            arg2_features.append((X_arg2, labels_arg2))
        # Arg1 is in the previous sentence (PS)
        elif (arg2_sentence_id - arg1_sentence_id) == 1:
            X_arg2 = copy.deepcopy(sent_features)
            labels_arg2 = []
            for i, word in enumerate(X_arg2):
                if i in arg2:
                    labels_arg2.append("Arg2")
                    word['is_arg2'] = True
                else:
                    labels_arg2.append("NULL")
                    word['is_arg2'] = False
            arg2_features.append((X_arg2, labels_arg2))
    return list(zip(*arg1_features)), list(zip(*arg2_features))


class ArgumentExtractCRF:
    def __init__(self):
        self.ss_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
        self.ps_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

    def load(self, path):
        self.ss_model = pickle.load(open(os.path.join(path, 'ss_extract_clf.p'), 'rb'))
        self.ps_model = pickle.load(open(os.path.join(path, 'ps_extract_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.ss_model, open(os.path.join(path, 'ss_extract_clf.p'), 'wb'))
        pickle.dump(self.ps_model, open(os.path.join(path, 'ps_extract_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        (X_ss, y_ss), (X_ps, y_ps) = generate_pdtb_features_crf(pdtb, parses)
        self.ss_model.fit(X_ss, y_ss)
        self.ps_model.fit(X_ps, y_ps)

    def score_on_features(self, X_ss, y_ss, X_ps, y_ps):
        y_pred = np.concatenate(self.ss_model.predict(X_ss))
        y_ss = np.concatenate(y_ss)
        logger.info("Evaluation: SS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ss, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ss, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ss, y_pred)))

        y_pred = np.concatenate(self.ps_model.predict(X_ps))
        y_ps = np.concatenate(y_ps)
        logger.info("Evaluation: PS Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_ps, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_ps, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_ps, y_pred)))

    def score(self, pdtb, parses):
        (X_ss, y_ss), (X_ps, y_ps) = generate_pdtb_features_crf(pdtb, parses)
        self.score_on_features(X_ss, y_ss, X_ps, y_ps)

    def extract_arguments(self, ptree, relation):
        indices = [token[4] for token in relation['Connective']['TokenList']]
        ptree._label = 'S'

        X = get_features(ptree, indices)
        if relation['ArgPos'] == 'SS':
            probs = np.array(
                [[p[c] for c in self.ss_model.classes_] for p in self.ss_model.predict_marginals_single(X)])
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
        elif relation['ArgPos'] == 'PS':
            probs = np.array(
                [[p[c] for c in self.ps_model.classes_] for p in self.ps_model.predict_marginals_single(X)])
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


class GoshArgumentExtract:
    def __init__(self):
        self.arg1_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
        self.arg2_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

    def load(self, path):
        self.arg1_model = pickle.load(open(os.path.join(path, 'arg1_extract_clf.p'), 'rb'))
        self.arg2_model = pickle.load(open(os.path.join(path, 'arg2_extract_clf.p'), 'rb'))

    def save(self, path):
        pickle.dump(self.arg1_model, open(os.path.join(path, 'arg1_extract_clf.p'), 'wb'))
        pickle.dump(self.arg2_model, open(os.path.join(path, 'arg2_extract_clf.p'), 'wb'))

    def fit(self, pdtb, parses):
        (X_arg1, y_arg1), (X_arg2, y_arg2) = generate_pdtb_features_gosh(pdtb, parses)
        self.arg1_model.fit(X_arg1, y_arg1)
        self.arg2_model.fit(X_arg2, y_arg2)

    def score_on_features(self, X_arg1, y_arg1, X_arg2, y_arg2):
        y_pred = np.concatenate(self.arg1_model.predict(X_arg1))
        y_arg1 = np.concatenate(y_arg1)
        logger.info("Evaluation: Arg1 Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_arg1, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_arg1, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_arg1, y_pred)))

        y_pred = np.concatenate(self.arg2_model.predict(X_arg2))
        y_arg2 = np.concatenate(y_arg2)
        logger.info("Evaluation: Arg2 Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_arg2, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_arg2, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_arg2, y_pred)))

    def score(self, pdtb, parses):
        (X_arg1, y_arg1), (X_arg2, y_arg2) = generate_pdtb_features_gosh(pdtb, parses)
        self.score_on_features(X_arg1, y_arg1, X_arg2, y_arg2)

    def extract_arguments(self, ptree, relation):
        indices = [token[4] for token in relation['Connective']['TokenList']]
        ptree._label = 'S'

        X = get_features(ptree, indices)
        if relation['ArgPos'] == 'SS':
            arg2_probs = np.array(
                [[p[c] for c in self.arg2_model.classes_] for p in self.arg2_model.predict_marginals_single(X)])
            arg2_probs_max = arg2_probs.max(1)
            arg2_labels = np.array(self.arg2_model.classes_)[arg2_probs.argmax(axis=1)]
            arg2 = np.where(arg2_labels == 'Arg2')[0]
            arg2_prob = arg2_probs_max[arg2].mean() if len(arg2) else 0.0
            arg2 = arg2.tolist()
            for i, w in enumerate(X):
                w['is_arg2'] = i in arg2
            arg1_probs = np.array(
                [[p[c] for c in self.arg1_model.classes_] for p in self.arg1_model.predict_marginals_single(X)])
            arg1_probs_max = arg1_probs.max(1)
            arg1_labels = np.array(self.arg1_model.classes_)[arg1_probs.argmax(axis=1)]
            arg1 = np.where(arg1_labels == 'Arg2')[0]
            arg1_prob = arg1_probs_max[arg1].mean() if len(arg1) else 0.0
            arg1 = arg1.tolist()
            if not arg1:
                logger.warning("Empty Arg1")
            if not arg2:
                logger.warning("Empty Arg2")
        elif relation['ArgPos'] == 'PS':
            arg2_probs = np.array(
                [[p[c] for c in self.arg2_model.classes_] for p in self.arg2_model.predict_marginals_single(X)])
            arg2_probs_max = arg2_probs.max(1)
            arg2_labels = np.array(self.arg2_model.classes_)[arg2_probs.argmax(axis=1)]
            arg2 = np.where(arg2_labels == 'Arg2')[0]
            arg2_prob = arg2_probs_max[arg2].mean() if len(arg2) else 0.0
            arg2 = arg2.tolist()
            arg1 = []
            arg1_prob = 1.0
            if not arg2:
                logger.warning("Empty Arg2")
        else:
            raise NotImplementedError('Unknown argument position')

        return arg1, arg2, arg1_prob, arg2_prob


if __name__ == "__main__":
    logger = init_logger()

    pdtb_train = [json.loads(s) for s in
                  open('/data/discourse/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('/data/discourse/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('/data/discourse/conll2016/en.test/relations.json', 'r').readlines()]
    parses_val = json.loads(open('/data/discourse/conll2016/en.test/parses.json').read())

    clf = ArgumentExtractCRF()
    logger.info('Train CRF model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
    clf.save('../tmp')

    clf = GoshArgumentExtract()
    logger.info('Train Gosh model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
    clf.save('../tmp')
