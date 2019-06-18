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


class ArgumentExtractCRF:
    def __init__(self):
        self.id = "crf_arg_extract"
        self.ss_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
        self.ps_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

    def load(self, path):
        self.ss_model = pickle.load(open(os.path.join(path, "{}.ss.p".format(self.id)), 'rb'))
        self.ps_model = pickle.load(open(os.path.join(path, "{}.ps.p".format(self.id)), 'rb'))

    def save(self, path):
        pickle.dump(self.ss_model, open(os.path.join(path, "{}.ss.p".format(self.id)), 'wb'))
        pickle.dump(self.ps_model, open(os.path.join(path, "{}.ps.p".format(self.id)), 'wb'))

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
    # clf.save('../tmp')
