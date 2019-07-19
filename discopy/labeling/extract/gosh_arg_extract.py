import copy
import logging
import os
import pickle
import ujson as json

import nltk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn_crfsuite import CRF

from discopy.utils import init_logger, encode_iob, decode_iob

logger = logging.getLogger('discopy')

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')


def get_gosh_features(ptree, indices, sense, offset):
    features = []
    for i, (word, tag) in enumerate(ptree.pos()):
        features.append({
            'idx': offset + i,
            'BOS': i == 0,
            'word': word,
            'pos': tag,
            'lemma': lemmatizer.lemmatize(word),
            'stem': stemmer.stem(word),
            'conn': sense if offset + i in indices else ""
        })
    return features


def generate_pdtb_features_gosh(pdtb, parses):
    arg1_features = []
    arg2_features = []

    for relation in filter(lambda i: i['Type'] == 'Explicit', pdtb):
        doc_id = relation['DocID']
        arg1 = set([i[2] for i in relation['Arg1']['TokenList']])
        arg2 = set([i[2] for i in relation['Arg2']['TokenList']])
        conn = [token[2] for token in relation['Connective']['TokenList']]

        try:
            arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
            sent_features = []

            offsets = [0]
            for s in parses[doc_id]['sentences']:
                offsets.append(offsets[-1] + len(s['words']))

            for i in range(-2, 3):
                sent_i = arg2_sentence_id + i
                if sent_i < 0 or sent_i >= len(parses[doc_id]['sentences']):
                    continue
                s_i = parses[doc_id]['sentences'][sent_i]['parsetree']
                ptree_i = nltk.ParentedTree.fromstring(s_i)
                if not ptree_i.leaves():
                    continue
                sent_features.extend(get_gosh_features(ptree_i, conn, relation['Sense'][0], offsets[sent_i]))
        except ValueError:
            continue
        except IndexError:
            continue

        X_arg1 = copy.deepcopy(sent_features)
        X_arg2 = copy.deepcopy(sent_features)
        labels_arg1 = []
        labels_arg2 = []
        for i, w_arg1 in enumerate(X_arg1):
            if w_arg1['idx'] in arg1:
                labels_arg1.append("Arg1")
                labels_arg2.append("NULL")
                w_arg1['is_arg2'] = False
            elif w_arg1['idx'] in arg2:
                labels_arg1.append("NULL")
                labels_arg2.append("Arg2")
                w_arg1['is_arg2'] = True
            else:
                labels_arg1.append("NULL")
                labels_arg2.append("NULL")
                w_arg1['is_arg2'] = False
            del X_arg1[i]['idx']
            del X_arg2[i]['idx']
        labels_arg1 = encode_iob(labels_arg1)
        labels_arg2 = encode_iob(labels_arg2)
        arg1_features.append((X_arg1, labels_arg1))
        arg2_features.append((X_arg2, labels_arg2))
    return list(zip(*arg1_features)), list(zip(*arg2_features))


class GoshArgumentExtract:
    def __init__(self):
        self.id = 'gosh_arg_extract'
        self.arg1_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
        self.arg2_model = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

    def load(self, path):
        self.arg1_model = pickle.load(open(os.path.join(path, "{}.arg1.p".format(self.id)), 'rb'))
        self.arg2_model = pickle.load(open(os.path.join(path, "{}.arg2.p".format(self.id)), 'rb'))

    def save(self, path):
        pickle.dump(self.arg1_model, open(os.path.join(path, "{}.arg1.p".format(self.id)), 'wb'))
        pickle.dump(self.arg2_model, open(os.path.join(path, "{}.arg2.p".format(self.id)), 'wb'))

    def fit(self, pdtb, parses):
        (X_arg1, y_arg1), (X_arg2, y_arg2) = generate_pdtb_features_gosh(pdtb, parses)
        self.arg1_model.fit(X_arg1, y_arg1)
        self.arg2_model.fit(X_arg2, y_arg2)

    def score_on_features(self, X_arg1, y_arg1, X_arg2, y_arg2):
        y_pred = np.concatenate([decode_iob(s) for s in self.arg1_model.predict(X_arg1)])
        y_arg1 = np.concatenate([decode_iob(s) for s in y_arg1])
        logger.info("Evaluation: Arg1 Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_arg1, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_arg1, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_arg1, y_pred)))

        y_pred = np.concatenate([decode_iob(s) for s in self.arg2_model.predict(X_arg2)])
        y_arg2 = np.concatenate([decode_iob(s) for s in y_arg2])
        logger.info("Evaluation: Arg2 Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y_arg2, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y_arg2, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y_arg2, y_pred)))

    def score(self, pdtb, parses):
        (X_arg1, y_arg1), (X_arg2, y_arg2) = generate_pdtb_features_gosh(pdtb, parses)
        self.score_on_features(X_arg1, y_arg1, X_arg2, y_arg2)

    def extract_arguments(self, doc, relation):
        conn = [token[2] for token in relation['Connective']['TokenList']]

        arg2_sentence_id = relation['Arg2']['TokenList'][0][3]
        X = []

        offsets = [0]
        for s in doc['sentences']:
            offsets.append(offsets[-1] + len(s['words']))

        for i in range(-2, 3):
            sent_i = arg2_sentence_id + i
            if sent_i < 0 or sent_i >= len(doc['sentences']):
                continue
            s_i = doc['sentences'][sent_i]['parsetree']
            ptree_i = nltk.ParentedTree.fromstring(s_i)
            ptree_i._label = 'S'
            if not ptree_i.leaves():
                continue
            X.extend(get_gosh_features(ptree_i, conn, relation['Sense'][0], offsets[sent_i]))

        indices = []
        for i in X:
            indices.append(i['idx'])
            del i['idx']
        indices = np.array(indices)

        arg2_probs = np.array(
            [[p[c] for c in self.arg2_model.classes_] for p in self.arg2_model.predict_marginals_single(X)])
        arg2_probs_max = arg2_probs.max(1)
        arg2_labels = np.array(self.arg2_model.classes_)[arg2_probs.argmax(axis=1)]

        for i, arg2_label in zip(X, arg2_labels):
            i['is_arg2'] = arg2_label == 'Arg2'

        arg1_probs = np.array(
            [[p[c] for c in self.arg1_model.classes_] for p in self.arg1_model.predict_marginals_single(X)])
        arg1_probs_max = arg1_probs.max(1)
        arg1_labels = np.array(self.arg1_model.classes_)[arg1_probs.argmax(axis=1)]

        arg1 = indices[np.where(arg1_labels == 'Arg1')[0]]
        arg2 = indices[np.where(arg2_labels == 'Arg2')[0]]

        arg1_prob = arg1_probs_max[arg1].mean() if len(arg1) else 0.0
        arg2_prob = arg2_probs_max[arg2].mean() if len(arg2) else 0.0

        arg1, arg2 = arg1.tolist(), arg2.tolist()

        return arg1, arg2, arg1_prob, arg2_prob


if __name__ == "__main__":
    logger = init_logger()

    pdtb_train = [json.loads(s) for s in
                  open('/data/discourse/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('/data/discourse/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('/data/discourse/conll2016/en.test/relations.json', 'r').readlines()]
    parses_val = json.loads(open('/data/discourse/conll2016/en.test/parses.json').read())

    clf = GoshArgumentExtract()
    logger.info('Train Gosh model')
    clf.fit(pdtb_train, parses_train)
    logger.info('Evaluation on TRAIN')
    clf.score(pdtb_train, parses_train)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_val, parses_val)
    # clf.save('../tmp')
