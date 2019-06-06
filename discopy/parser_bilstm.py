import logging
import os
import ujson as json
from collections import Counter

import joblib
import nltk

from discopy.labeling.connective import ConnectiveClassifier
from discopy.labeling.neural.bilstm import ArgumentExtractBiLSTMCRF, ArgumentExtractBiLSTMCRFwithConn
from discopy.sense.explicit import ExplicitSenseClassifier
from discopy.sense.nonexplicit import NonExplicitSenseClassifier
from discopy.utils import init_logger

logger = logging.getLogger('discopy')


def get_token_list2(doc_words, tokens):
    return [[doc_words[i][0][1]['CharacterOffsetBegin'],
             doc_words[i][0][1]['CharacterOffsetEnd'],
             i, doc_words[i][1], doc_words[i][2]] for i in sorted(tokens)]


def get_raw_tokens(doc_words, idxs):
    # TODO fix whitespaces, depend on the character offsets
    return " ".join([doc_words[i][0][0] for i in sorted(idxs)])


class BiLSTMDiscourseParser3(object):

    def __init__(self, n_estimators=1, no_crf=False):
        self.arg_labeler = ArgumentExtractBiLSTMCRF(window_length=150, hidden_dim=128, rnn_dim=256, no_rnn=False,
                                                    no_dense=False, no_crf=no_crf, explicits_only=False)
        self.explicit_clf = ExplicitSenseClassifier(n_estimators=n_estimators)
        self.non_explicit_clf = NonExplicitSenseClassifier(n_estimators=n_estimators)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=10)
        logger.info('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses)
        logger.info('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses)

    def score(self, pdtb, parses):
        self.arg_labeler.score(pdtb, parses)
        self.explicit_clf.score(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.arg_labeler.save(path)
        self.explicit_clf.save(path)
        self.non_explicit_clf.save(path)

    def load(self, path, parses):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
        self.arg_labeler.init_model(parses)
        self.arg_labeler.load(path)
        self.explicit_clf.load(path)
        self.non_explicit_clf.load(path)

    @staticmethod
    def from_path(path):
        return joblib.load(os.path.join(path, 'parser.joblib'))

    def parse_documents(self, documents):
        relations = []
        for idx, (doc_id, doc) in enumerate(documents.items()):
            parsed_relations = self.parse_doc(doc)
            for p in parsed_relations:
                p['DocID'] = doc_id
            relations.extend(parsed_relations)

        return relations

    def parse_doc(self, doc):
        relations = []
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

        # ARGUMENT EXTRACTION
        # TODO adjust max distance
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words], strides=1, max_distance=0.5)
        # print('ArgPred', arguments_pred)
        # print('Length', len(arguments_pred))
        for r in arguments_pred:
            relation = {
                'Connective': {
                    'TokenList': [],
                    'RawText': ''
                },
                'Arg1': {
                    'TokenList': [],
                    'RawText': ''
                },
                'Arg2': {
                    'TokenList': [],
                    'RawText': ''
                },
                'Type': '',
                'Sense': []
            }
            relation['Arg1']['TokenList'] = get_token_list2(doc_words, r.arg1)
            relation['Arg1']['RawText'] = get_raw_tokens(doc_words, r.arg1)
            relation['Arg2']['TokenList'] = get_token_list2(doc_words, r.arg2)
            relation['Arg2']['RawText'] = get_raw_tokens(doc_words, r.arg2)

            if r.conn:
                relation['Type'] = 'Explicit'
                sent_id = Counter(i[3] for i in get_token_list2(doc_words, r.conn)).most_common(1)[0][0]

                conn = [i[2] for i in get_token_list2(doc_words, r.conn) if i[3] == sent_id]

                relation['Connective']['TokenList'] = get_token_list2(doc_words, conn)
                relation['Connective']['RawText'] = get_raw_tokens(doc_words, conn)

                # EXPLICIT SENSE
                sent_id = min([i[3] for i in relation['Connective']['TokenList']])
                try:
                    ptree = nltk.ParentedTree.fromstring(doc['sentences'][sent_id]['parsetree'])
                except ValueError:
                    logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                    continue
                if not ptree.leaves():
                    logger.warning('Failed on empty tree')
                    continue

                explicit, explicit_c = self.explicit_clf.get_sense(relation, ptree)
                relation['Sense'] = [explicit]
                relations.append(relation)
            else:
                relation['Type'] = 'Implicit'

                sent_id_prev = Counter(i[3] for i in get_token_list2(doc_words, r.arg1)).most_common(1)[0][0]
                sent_id = Counter(i[3] for i in get_token_list2(doc_words, r.arg2)).most_common(1)[0][0]

                if sent_id_prev + 1 != sent_id:
                    continue

                sent = doc['sentences'][sent_id]
                try:
                    ptree = nltk.ParentedTree.fromstring(sent['parsetree'])
                    ptree_prev = nltk.ParentedTree.fromstring(doc['sentences'][sent_id - 1]['parsetree'])
                    dtree = sent['dependencies']
                    dtree_prev = doc['sentences'][sent_id - 1]['dependencies']
                except ValueError:
                    logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                    continue

                if not ptree.leaves() or not ptree_prev.leaves():
                    continue
                arg1 = [doc_words[i][0][0] for i in r.arg1]
                arg2 = [doc_words[i][0][0] for i in r.arg2]
                sense, sense_c = self.non_explicit_clf.get_sense(ptree_prev, ptree, dtree_prev, dtree, arg1, arg2)
                relation['Sense'] = [sense]
                relations.append(relation)

        return relations


class BiLSTMDiscourseParser2(object):

    def __init__(self, n_estimators=1, no_crf=False):
        self.arg_labeler = ArgumentExtractBiLSTMCRF(window_length=100, hidden_dim=128, rnn_dim=512, no_rnn=False,
                                                    no_dense=False, no_crf=no_crf, explicits_only=True)
        self.explicit_clf = ExplicitSenseClassifier(n_estimators=n_estimators)
        self.non_explicit_clf = NonExplicitSenseClassifier(n_estimators=n_estimators)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=10)
        logger.info('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses)
        logger.info('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses)

    def score(self, pdtb, parses):
        self.arg_labeler.score(pdtb, parses)
        self.explicit_clf.score(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.arg_labeler.save(path)
        self.explicit_clf.save(path)
        self.non_explicit_clf.save(path)

    def load(self, path, parses):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
        self.arg_labeler.init_model(parses)
        self.arg_labeler.load(path)
        self.explicit_clf.load(path)
        self.non_explicit_clf.load(path)

    @staticmethod
    def from_path(path):
        return joblib.load(os.path.join(path, 'parser.joblib'))

    def parse_documents(self, documents):
        relations = []
        for idx, (doc_id, doc) in enumerate(documents.items()):
            parsed_relations = self.parse_doc(doc)
            for p in parsed_relations:
                p['DocID'] = doc_id
            relations.extend(parsed_relations)

        return relations

    def parse_doc(self, doc):
        relations = []
        inter_relations = set()
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

        # ARGUMENT EXTRACTION
        # TODO adjust max distance
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words], strides=1, max_distance=0.5)
        # print('ArgPred', arguments_pred)
        # print('Length', len(arguments_pred))
        for r in arguments_pred:
            if not r.conn:
                continue
            relation = {
                'Connective': {},
                'Arg1': {},
                'Arg2': {},
                'Type': 'Explicit',
            }
            sent_id = Counter(i[3] for i in get_token_list2(doc_words, r.conn)).most_common(1)[0][0]

            conn = [i[2] for i in get_token_list2(doc_words, r.conn) if i[3] == sent_id]

            relation['Connective']['TokenList'] = get_token_list2(doc_words, conn)
            relation['Connective']['RawText'] = get_raw_tokens(doc_words, conn)
            relation['Arg1']['TokenList'] = get_token_list2(doc_words, r.arg1)
            relation['Arg1']['RawText'] = get_raw_tokens(doc_words, r.arg1)
            relation['Arg2']['TokenList'] = get_token_list2(doc_words, r.arg2)
            relation['Arg2']['RawText'] = get_raw_tokens(doc_words, r.arg2)
            relations.append(relation)
            # print(relation)

        # EXPLICIT SENSE
        for relation in relations:
            sent_id = min([i[3] for i in relation['Connective']['TokenList']])
            try:
                ptree = nltk.ParentedTree.fromstring(doc['sentences'][sent_id]['parsetree'])
            except ValueError:
                logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                continue
            if not ptree.leaves():
                logger.warning('Failed on empty tree')
                continue

            explicit, explicit_c = self.explicit_clf.get_sense(relation, ptree)
            relation['Sense'] = [explicit]
            if sent_id - max(i[3] for i in relation['Arg1']['TokenList']) == 1:
                inter_relations.add(sent_id)

        token_id = 0
        sent_lengths = [0]
        for sent_id, sent in enumerate(doc['sentences']):
            sent_lengths.append(len(sent['words']))
            if sent_id == 0 or sent_id in inter_relations:
                token_id += len(sent['words'])
                continue

            try:
                ptree = nltk.ParentedTree.fromstring(sent['parsetree'])
                ptree_prev = nltk.ParentedTree.fromstring(doc['sentences'][sent_id - 1]['parsetree'])
                dtree = sent['dependencies']
                dtree_prev = doc['sentences'][sent_id - 1]['dependencies']
            except ValueError:
                logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                continue

            if not ptree.leaves() or not ptree_prev.leaves():
                continue
            arg1_idxs = list(range(sum(sent_lengths[:-2]), sum(sent_lengths[:-1])))
            arg2_idxs = list(range(sum(sent_lengths[:-1]), sum(sent_lengths)))
            relation = {
                'Connective': {
                    'TokenList': []
                },
                'Arg1': {
                    'TokenList': get_token_list2(doc_words, arg1_idxs)
                    # 'TokenList': get_token_list2(doc_words, list(range(len(ptree_prev.leaves()))), sent_id - 1,
                    #                             sum(sent_lengths[:-2]))
                },
                'Arg2': {
                    'TokenList': get_token_list2(doc_words, arg2_idxs)
                    # 'TokenList': get_token_list2(doc_words, list(range(len(ptree.leaves()))), sent_id,
                    #                             sum(sent_lengths[:-1]))
                },
                'Type': 'Implicit',
                'Sense': [],
            }
            relation['Arg1']['RawText'] = get_raw_tokens(doc_words, arg1_idxs)
            relation['Arg2']['RawText'] = get_raw_tokens(doc_words, arg2_idxs)
            arg1 = [doc_words[i][0][0] for i in arg1_idxs]
            arg2 = [doc_words[i][0][0] for i in arg2_idxs]
            sense, sense_c = self.non_explicit_clf.get_sense(ptree_prev, ptree, dtree_prev, dtree, arg1, arg2)
            relation['Sense'] = [sense]
            relations.append(relation)

            token_id += len(sent['words'])

        # for r in relations:
        #     r['Confidence'] = np.mean(list(r['Confidences'].values()))

        return relations


class BiLSTMDiscourseParser1(object):
    """
    Extracts explicit arguments based on the connective prediction
    """

    def __init__(self, n_estimators=1, no_crf=False):
        self.connective_clf = ConnectiveClassifier(n_estimators=n_estimators)
        self.arg_labeler = ArgumentExtractBiLSTMCRFwithConn(window_length=100, hidden_dim=128, rnn_dim=512,
                                                            no_rnn=False,
                                                            no_dense=False, no_crf=no_crf)
        self.explicit_clf = ExplicitSenseClassifier(n_estimators=n_estimators)
        self.non_explicit_clf = NonExplicitSenseClassifier(n_estimators=n_estimators)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses)
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=5)
        logger.info('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses)
        logger.info('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses)

    def score(self, pdtb, parses):
        self.connective_clf.score(pdtb, parses)
        self.arg_labeler.score(pdtb, parses)
        self.explicit_clf.score(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.connective_clf.save(path)
        self.arg_labeler.save(path)
        self.explicit_clf.save(path)
        self.non_explicit_clf.save(path)

    def load(self, path, parses):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
        self.connective_clf.load(path)
        self.arg_labeler.init_model(parses)
        self.arg_labeler.load(path)
        self.explicit_clf.load(path)
        self.non_explicit_clf.load(path)

    @staticmethod
    def from_path(path):
        return joblib.load(os.path.join(path, 'parser.joblib'))

    def parse_documents(self, documents):
        relations = []
        for idx, (doc_id, doc) in enumerate(documents.items()):
            parsed_relations = self.parse_doc(doc)
            for p in parsed_relations:
                p['DocID'] = doc_id
            relations.extend(parsed_relations)

        return relations

    def parse_doc(self, doc):
        relations = []
        token_id = 0
        sent_offset = 0
        inter_relations = set()
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        doc_sents = []

        for sent_id, sent in enumerate(doc['sentences']):
            sent_len = len(sent['words'])
            try:
                ptree = nltk.ParentedTree.fromstring(sent['parsetree'])
            except ValueError:
                logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                token_id += sent_len
                continue
            if not ptree.leaves():
                logger.warning('Failed on empty tree')
                token_id += sent_len
                continue

            current_token = 0
            while current_token < sent_len:
                relation = {
                    'Connective': {},
                    'Arg1': {},
                    'Arg2': {},
                    'Type': 'Explicit',
                    # 'Confidences': {},
                    'ptree': ptree,
                }

                # CONNECTIVE CLASSIFIER
                connective, connective_confidence = self.connective_clf.get_connective(ptree, sent['words'],
                                                                                       current_token)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    current_token += 1
                    continue
                conn_idxs = [token_id + i for i in range(len(connective))]
                relation['Connective']['TokenList'] = get_token_list2(doc_words, conn_idxs)
                relation['Connective']['RawText'] = get_raw_tokens(doc_words, conn_idxs)

                relations.append(relation)
                token_id += len(connective)
                current_token += len(connective)
            sent_offset += sent_len
        # print(relations)
        # print(len(relations))

        # ARGUMENT EXTRACTION
        if relations:
            arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words], relations,
                                                                max_distance=0.5)
            # print('ArgPred', arguments_pred)
            # print('Length', len(arguments_pred))
            for relation, r in zip(relations, arguments_pred):
                relation['Arg1']['TokenList'] = get_token_list2(doc_words, r.arg1)
                relation['Arg1']['RawText'] = get_raw_tokens(doc_words, r.arg1)
                relation['Arg2']['TokenList'] = get_token_list2(doc_words, r.arg2)
                relation['Arg2']['RawText'] = get_raw_tokens(doc_words, r.arg2)

        # EXPLICIT SENSE
        for relation in relations:
            explicit, explicit_c = self.explicit_clf.get_sense(relation, relation['ptree'])
            relation['Sense'] = [explicit]
            del relation['ptree']

        # TODO add intersentence relations
        # inter_relations.add(sent_id)

        token_id = 0
        sent_lengths = [0]
        for sent_id, sent in enumerate(doc['sentences']):
            sent_lengths.append(len(sent['words']))
            if sent_id == 0 or sent_id in inter_relations:
                token_id += len(sent['words'])
                continue

            try:
                ptree = nltk.ParentedTree.fromstring(sent['parsetree'])
                ptree_prev = nltk.ParentedTree.fromstring(doc['sentences'][sent_id - 1]['parsetree'])
                dtree = sent['dependencies']
                dtree_prev = doc['sentences'][sent_id - 1]['dependencies']
            except ValueError:
                logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                continue

            if not ptree.leaves() or not ptree_prev.leaves():
                continue
            arg1_idxs = list(range(sum(sent_lengths[:-2]), sum(sent_lengths[:-1])))
            arg2_idxs = list(range(sum(sent_lengths[:-1]), sum(sent_lengths)))
            relation = {
                'Connective': {
                    'TokenList': []
                },
                'Arg1': {
                    'TokenList': get_token_list2(doc_words, arg1_idxs)
                    # 'TokenList': get_token_list2(doc_words, list(range(len(ptree_prev.leaves()))), sent_id - 1,
                    #                             sum(sent_lengths[:-2]))
                },
                'Arg2': {
                    'TokenList': get_token_list2(doc_words, arg2_idxs)
                    # 'TokenList': get_token_list2(doc_words, list(range(len(ptree.leaves()))), sent_id,
                    #                             sum(sent_lengths[:-1]))
                },
                'Type': 'Implicit',
                'Sense': [],
            }
            relation['Arg1']['RawText'] = get_raw_tokens(doc_words, arg1_idxs)
            relation['Arg2']['RawText'] = get_raw_tokens(doc_words, arg2_idxs)
            arg1 = [doc_words[i][0][0] for i in arg1_idxs]
            arg2 = [doc_words[i][0][0] for i in arg2_idxs]
            sense, sense_c = self.non_explicit_clf.get_sense(ptree_prev, ptree, dtree_prev, dtree, arg1, arg2)
            relation['Sense'] = [sense]
            relations.append(relation)

            token_id += len(sent['words'])

        # for r in relations:
        #     r['Confidence'] = np.mean(list(r['Confidences'].values()))

        return relations


if __name__ == "__main__":
    logger = init_logger()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    pdtb_train = [json.loads(s) for s in
                  open('/data/discourse/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('/data/discourse/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('/data/discourse/conll2016/en.dev/relations.json', 'r').readlines()]
    parses_val = json.loads(open('/data/discourse/conll2016/en.dev/parses.json').read())
    pdtb_test = [json.loads(s) for s in open('/data/discourse/conll2016/en.test/relations.json', 'r').readlines()]
    parses_test = json.loads(open('/data/discourse/conll2016/en.test/parses.json').read())

    parser = BiLSTMDiscourseParser1()
    logger.info('Train Parser')
    parser.train(pdtb_train, parses_train, pdtb_val, parses_val)
    # parser.save('bilstm-tmp')
    parser.parse_doc(parses_train['wsj_1843'])
    # logger.info('Evaluation on VAL')
    # clf.score(pdtb_val, parses_val)
    # logger.info('Evaluation on TEST')
    # clf.score(pdtb_test, parses_test)
