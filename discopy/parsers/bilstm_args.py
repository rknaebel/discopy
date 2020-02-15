import copy
import logging
import os
import sys
from collections import Counter

import joblib
import numpy as np
from tqdm import tqdm

from discopy.data.conll16 import get_conll_dataset
from discopy.labeling.connective import ConnectiveClassifier
from discopy.labeling.neural.arg_extract import ArgumentExtractBiLSTM, BiLSTMConnectiveArgumentExtractor
from discopy.labeling.neural.elmo_arg_extract import ElmoConnectiveArgumentExtractor, ElmoArgumentExtract
from discopy.parsers.parser import AbstractBaseParser
from discopy.parsers.utils import get_token_list2, get_raw_tokens2
from discopy.utils import init_logger, ParsedRelation, bootstrap_dataset

logger = logging.getLogger('discopy')


# TODO use this class in other parsers as well!
class Relation:
    class Span:
        def __init__(self):
            self.TokenList = []
            self.RawText = ''

        def get_sentence_ids(self):
            return sorted(set(i[3] for i in self.TokenList))

        def get_global_ids(self):
            return sorted(set(i[2] for i in self.TokenList))

        def get_local_ids(self):
            return sorted(set(i[4] for i in self.TokenList))

        def is_valid(self):
            return len(self.TokenList) > 0

    def __init__(self):
        self.Connective = Relation.Span()
        self.Arg1 = Relation.Span()
        self.Arg2 = Relation.Span()

    def to_dict(self):
        return {
            'Connective': {
                'TokenList': self.Connective.TokenList,
                'RawText': self.Connective.RawText
            },
            'Arg1': {
                'TokenList': self.Arg1.TokenList,
                'RawText': self.Arg1.RawText
            },
            'Arg2': {
                'TokenList': self.Arg2.TokenList,
                'RawText': self.Arg2.RawText
            },
        }

    def is_valid(self):
        return self.Arg1.is_valid() and self.Arg2.is_valid()

    def to_conll(self):
        r = self.to_dict()
        return r

    def is_explicit(self):
        return self.Connective.RawText != ''

    def __str__(self):
        return "Relation(arg1:<{}> arg2:<{}> conn:<{}>)".format(
            ",".join(map(str, self.Arg1.get_global_ids())),
            ",".join(map(str, self.Arg2.get_global_ids())),
            ",".join(map(str, self.Connective.get_global_ids()))
        )

    @staticmethod
    def from_dict(d):
        r = Relation()
        r.Connective.TokenList = d['Connective']['TokenList']
        r.Connective.RawText = d['Connective']['RawText']
        r.Arg1.TokenList = d['Arg1']['TokenList']
        r.Arg1.RawText = d['Arg1']['RawText']
        r.Arg2.TokenList = d['Arg2']['TokenList']
        r.Arg2.RawText = d['Arg2']['RawText']
        return r


class AbstractBiLSTMDiscourseParser(AbstractBaseParser):
    def __init__(self, hidden_size, rnn_size, window_length):
        self.hidden_size = hidden_size
        self.rnn_size = rnn_size
        self.window_length = window_length

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')

    @staticmethod
    def from_path(path):
        return joblib.load(os.path.join(path, 'parser.joblib'))


class NeuralExplicitArgumentExtractor(AbstractBiLSTMDiscourseParser):

    def __init__(self, hidden_size=256, rnn_size=256, window_length=100, elmo=False, no_crf=False):
        super().__init__(hidden_size, rnn_size, window_length)
        if elmo:
            self.arg_labeler = ElmoArgumentExtract(window_length=self.window_length,
                                                   hidden_dim=self.hidden_size,
                                                   rnn_dim=self.rnn_size,
                                                   no_rnn=False,
                                                   no_dense=False,
                                                   no_crf=no_crf,
                                                   explicits_only=True)
        else:
            self.arg_labeler = ArgumentExtractBiLSTM(window_length=self.window_length,
                                                     hidden_dim=self.hidden_size,
                                                     rnn_dim=self.rnn_size,
                                                     no_rnn=False,
                                                     no_dense=False,
                                                     no_crf=no_crf,
                                                     explicits_only=True)

    def fit(self, pdtb, parses, pdtb_val, parses_val, path='', epochs=10):
        logger.info('Train Argument Extractor...')
        if not os.path.exists(path):
            os.makedirs(path)
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, save_path=path, epochs=epochs)

    def fit_noisy(self, pdtb, parses, pdtb_val, parses_val, pdtb_noisy, parses_noisy, path='', epochs=10):
        logger.info('Train Argument Extractor...')
        if not os.path.exists(path):
            os.makedirs(path)
        self.arg_labeler.fit_noisy(pdtb, parses, pdtb_val, parses_val, pdtb_noisy, parses_noisy, save_path=path,
                                   epochs=epochs)

    def score(self, pdtb, parses):
        self.arg_labeler.score(pdtb, parses)

    def save(self, path):
        super().save(path)
        self.arg_labeler.save(path)

    def load(self, path):
        super().load(path)
        self.arg_labeler.load(path)

    def parse_explicit_arguments(self, doc):
        relations = []
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                            strides=1, max_distance=0.5)
        for r in arguments_pred:
            if not r.conn:
                continue
            relation = ParsedRelation()
            sent_id = Counter(i[3] for i in get_token_list2(doc_words, r.conn)).most_common(1)[0][0]
            conn = [i[2] for i in get_token_list2(doc_words, r.conn) if i[3] == sent_id]
            relation.Connective.TokenList = get_token_list2(doc_words, conn)
            relation.Connective.RawText = get_raw_tokens2(doc_words, conn)
            relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
            relation.Arg1.RawText = get_raw_tokens2(doc_words, r.arg1)
            relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
            relation.Arg2.RawText = get_raw_tokens2(doc_words, r.arg2)
            relations.append(relation)
        return relations

    def parse_doc(self, doc):
        relations = self.parse_explicit_arguments(doc)
        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_conll() for r in relations if r.is_valid()]
        return relations


class NeuralConnectiveArgumentExtractor(AbstractBiLSTMDiscourseParser):
    """
    Extracts explicit arguments based on the connective prediction
    """

    def __init__(self, hidden_size=256, rnn_size=256, window_length=100, elmo=False, no_crf=False):
        super().__init__(hidden_size, rnn_size, window_length)
        self.connective_clf = ConnectiveClassifier()
        if elmo:
            self.arg_labeler = ElmoConnectiveArgumentExtractor(window_length=self.window_length,
                                                               hidden_dim=self.hidden_size,
                                                               rnn_dim=self.rnn_size,
                                                               no_rnn=False,
                                                               no_dense=False, no_crf=no_crf)
        else:
            self.arg_labeler = BiLSTMConnectiveArgumentExtractor(window_length=self.window_length,
                                                                 hidden_dim=self.hidden_size,
                                                                 rnn_dim=self.rnn_size,
                                                                 no_rnn=False,
                                                                 no_dense=False, no_crf=no_crf)

    def fit(self, pdtb, parses, pdtb_val, parses_val, epochs=25):
        logger.info('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses)
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=epochs)

    def score(self, pdtb, parses):
        self.connective_clf.score(pdtb, parses)
        self.arg_labeler.score(pdtb, parses)

    def save(self, path):
        super().save(path)
        self.connective_clf.save(path)
        self.arg_labeler.save(path)

    def load(self, path):
        super().load(path)
        self.connective_clf.load(path)
        self.arg_labeler.load(path)

    def parse_connectives(self, doc):
        relations = []
        token_id = 0
        sent_offset = 0
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

        for sent_id, sent in enumerate(doc['sentences']):
            sent_len = len(sent['words'])
            ptree = sent['parsetree']
            if not ptree.leaves():
                logger.warning('Failed on empty tree')
                token_id += sent_len
                sent_offset += sent_len
                continue

            current_token = 0
            while current_token < sent_len:
                relation = ParsedRelation()
                relation.Type = 'Explicit'
                connective, connective_confidence = self.connective_clf.get_connective(ptree, sent['words'],
                                                                                       current_token)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    current_token += 1
                    continue
                conn_idxs = [sent_offset + i for i, c in connective]
                relation.Connective.TokenList = get_token_list2(doc_words, conn_idxs)
                relation.Connective.RawText = get_raw_tokens2(doc_words, conn_idxs)

                relations.append(relation)
                token_id += 1
                current_token += 1
            sent_offset += sent_len
        return relations

    def parse_explicit_arguments(self, doc, relations):
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                            [r.to_dict() for r in relations])
        for relation, r in zip(relations, arguments_pred):
            relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
            relation.Arg1.RawText = get_raw_tokens2(doc_words, r.arg1)
            relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
            relation.Arg2.RawText = get_raw_tokens2(doc_words, r.arg2)
        return relations

    def parse_doc(self, doc):
        relations = self.parse_connectives(doc)
        relations = self.parse_explicit_arguments(doc, relations)
        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_conll() for r in relations if r.is_valid()]
        return relations


class NeuralTriConnectiveArgumentExtractor(NeuralConnectiveArgumentExtractor):
    """
    Extracts explicit arguments based on the connective prediction
    """

    def __init__(self, hidden_sizes=(64, 128, 256), rnn_sizes=(128, 256, 512), window_length=150, no_crf=False):
        self.connective_clf = ConnectiveClassifier()

        self.arg_labelers = []
        for hidden_size, rnn_size in zip(hidden_sizes, rnn_sizes):
            self.arg_labelers.append(BiLSTMConnectiveArgumentExtractor(window_length=window_length,
                                                                       hidden_dim=hidden_size,
                                                                       rnn_dim=rnn_size,
                                                                       no_rnn=False,
                                                                       no_dense=False, no_crf=no_crf))

    def fit(self, pdtb, parses, pdtb_val, parses_val, epochs=25, bootstrap=False, init_model=True):
        if init_model:
            logger.info('Train Connective Classifier...')
            self.connective_clf.fit(pdtb, parses)
        logger.info('Train Argument Extractors...')
        if bootstrap:
            logger.info('Bootstrap data:')
            straps = bootstrap_dataset(pdtb, parses, n_straps=len(self.arg_labelers), ratio=0.7)
            for arg_labeler, (pdtb_i, parses_i) in zip(self.arg_labelers, straps):
                logger.info('Dataset sizes: Relations({}) Documents({})'.format(len(pdtb_i), len(parses_i)))
                arg_labeler.fit(pdtb_i, parses_i, pdtb_val, parses_val, epochs=epochs, init_model=init_model)
        else:
            logger.info('Use all data')
            for arg_labeler in self.arg_labelers:
                arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=epochs, init_model=init_model)

    def score(self, pdtb, parses):
        self.connective_clf.score(pdtb, parses)
        for arg_labeler in self.arg_labelers:
            arg_labeler.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.connective_clf.save(path)
        for idx, arg_labeler in enumerate(self.arg_labelers):
            if not os.path.exists(os.path.join(path, str(idx))):
                os.makedirs(os.path.join(path, str(idx)))
            arg_labeler.save(os.path.join(path, str(idx)))

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
        self.connective_clf.load(path)
        for idx, arg_labeler in enumerate(self.arg_labelers):
            arg_labeler.load(os.path.join(path, str(idx)))

    def parse_connectives(self, doc):
        relations = []
        token_id = 0
        sent_offset = 0
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

        for sent_id, sent in enumerate(doc['sentences']):
            sent_len = len(sent['words'])
            ptree = sent['parsetree']
            if not ptree.leaves():
                logger.warning('Failed on empty tree')
                token_id += sent_len
                sent_offset += sent_len
                continue

            current_token = 0
            while current_token < sent_len:
                relation = ParsedRelation()
                relation.Type = 'Explicit'
                connective, connective_confidence = self.connective_clf.get_connective(ptree, sent['words'],
                                                                                       current_token)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    current_token += 1
                    continue
                conn_idxs = [sent_offset + i for i, c in connective]
                relation.Connective.TokenList = get_token_list2(doc_words, conn_idxs)
                relation.Connective.RawText = get_raw_tokens2(doc_words, conn_idxs)

                relations.append(relation)
                token_id += 1
                current_token += 1
            sent_offset += sent_len
        return relations

    def parse_explicit_arguments(self, doc, relations, mode='majority'):
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        arguments_preds = []
        result = []
        if mode == 'majority':
            for arg_labeler in self.arg_labelers:
                arguments_preds.append(arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                                     [r.to_dict() for r in relations]))
            for relation, r0, r1, r2 in zip(relations, arguments_preds[0], arguments_preds[1], arguments_preds[2]):
                if r0 == r1 or r0 == r2:
                    r = r0
                elif r1 == r2:
                    r = r1
                else:
                    continue
                relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
                relation.Arg1.RawText = get_raw_tokens2(doc_words, r.arg1)
                relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
                relation.Arg2.RawText = get_raw_tokens2(doc_words, r.arg2)
                if relation.is_valid():
                    result.append(relation)
        elif mode == 'all':
            for arg_labeler in self.arg_labelers:
                arguments_preds.append(arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                                     [r.to_dict() for r in relations]))
            for relation, r0, r1, r2 in zip(relations, arguments_preds[0], arguments_preds[1], arguments_preds[2]):
                for r in [r0, r1, r2]:
                    rel = copy.deepcopy(relation)
                    rel.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
                    rel.Arg1.RawText = get_raw_tokens2(doc_words, r.arg1)
                    rel.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
                    rel.Arg2.RawText = get_raw_tokens2(doc_words, r.arg2)
                    result.append(rel)

        else:
            for arg_labeler in self.arg_labelers:
                arguments_preds.append(arg_labeler.get_window_probs([w[0][0] for w in doc_words]))
            arguments_preds = np.stack(arguments_preds).mean(axis=0)
            arguments_pred, _ = self.arg_labelers[0].get_relations_for_window_probs(arguments_preds,
                                                                                    [w[0][0] for w in
                                                                                     doc_words],
                                                                                    [r.to_dict() for r in
                                                                                     relations])
            for relation, r in zip(relations, arguments_pred):
                relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
                relation.Arg1.RawText = get_raw_tokens2(doc_words, r.arg1)
                relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
                relation.Arg2.RawText = get_raw_tokens2(doc_words, r.arg2)
                result.append(relation)

        return result

    def parse_doc(self, doc, mode='average'):
        relations = self.parse_connectives(doc)
        relations = self.parse_explicit_arguments(doc, relations, mode=mode)
        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_conll() for r in relations if r.is_valid()]
        return relations

    def extract_document_relations(self, documents, mode='majority', limit=0):
        doc_relations = []
        for idx, (doc_id, doc) in tqdm(enumerate(documents.items()), total=len(documents)):
            if limit and idx > limit:
                break
            parsed_relations = self.parse_doc(doc, mode=mode)
            for p in parsed_relations:
                p['DocID'] = doc_id
                # p['ID'] = hash(p['Connective']['RawText'] + p['Arg1']['RawText'] + p['Arg2']['RawText'])
                doc_relations.append(p)
        return doc_relations


if __name__ == "__main__":
    logger = init_logger()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    data_path = sys.argv[1]
    parses_train, pdtb_train = get_conll_dataset(data_path, 'en.train', load_trees=False, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(data_path, 'en.dev', load_trees=False, connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset(data_path, 'en.test', load_trees=False, connective_mapping=True)

    parser = NeuralConnectiveArgumentExtractor()
    logger.info('Train Parser')
    parser.load('NCA-tmp')
    parser.fit(pdtb_train, parses_train, pdtb_val, parses_val)
    # parser.save('bilstm-tmp')
    logger.info('Evaluation on VAL')
    parser.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    parser.score(pdtb_test, parses_test)

    parser = NeuralExplicitArgumentExtractor()
    logger.info('Train Parser')
    parser.load('NEA-tmp')
    parser.fit(pdtb_train, parses_train, pdtb_val, parses_val)
    # parser.save('bilstm-tmp')
    logger.info('Evaluation on VAL')
    parser.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    parser.score(pdtb_test, parses_test)