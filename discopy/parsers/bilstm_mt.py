import logging
import os
import sys
from collections import Counter

import joblib
import numpy as np

from discopy.data.conll16 import get_conll_dataset
from discopy.labeling.connective import ConnectiveClassifier
from discopy.labeling.neural.arg_extract_mt import BiLSTMConnectiveArgumentExtractor, ArgumentExtractBiLSTM
from discopy.labeling.neural.arg_extract_mt import BertConnectiveArgumentExtractor, BertArgumentExtractor
from discopy.parsers.parser import AbstractBaseParser
from discopy.parsers.utils import get_token_list2, get_raw_tokens2
from discopy.utils import init_logger, ParsedRelation

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

    def __init__(self, hidden_size=256, rnn_size=256, window_length=100, no_crf=False, use_bert=False):
        super().__init__(hidden_size, rnn_size, window_length)
        self.use_bert = use_bert
        if use_bert:
            self.arg_labeler = BertArgumentExtractor(window_length=self.window_length,
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

    def fit(self, pdtb, parses, pdtb_val, parses_val, path=None, epochs=10):
        logger.info('Train Argument Extractor...')
        if path and not os.path.exists(path):
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
        if self.use_bert:
            embds = np.concatenate([s['bert'] for s in doc['sentences']])
            arguments_pred = self.arg_labeler.extract_arguments(embds, strides=1, max_distance=0.5)
        else:
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

    def __init__(self, hidden_size=256, rnn_size=256, window_length=100, no_crf=False, use_bert=False):
        super().__init__(hidden_size, rnn_size, window_length)
        self.connective_clf = ConnectiveClassifier()
        self.use_bert = use_bert
        if use_bert:
            self.arg_labeler = BertConnectiveArgumentExtractor(window_length=self.window_length,
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
            if not ptree or not ptree.leaves():
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
        if self.use_bert:
            embds = np.concatenate([s['bert'] for s in doc['sentences']])
            arguments_pred = self.arg_labeler.extract_arguments(embds, [r.to_dict() for r in relations])
        else:
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
