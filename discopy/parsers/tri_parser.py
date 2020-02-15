import copy
import logging
import os

import joblib
import numpy as np
from discopy.parser import DiscourseParser
from tqdm import tqdm

from discopy.labeling.connective import ConnectiveClassifier
from discopy.labeling.neural.arg_extract import BiLSTMConnectiveArgumentExtractor
from discopy.parsers.bilstm_args import NeuralConnectiveArgumentExtractor
from discopy.parsers.utils import get_token_list2, get_raw_tokens2
from discopy.utils import bootstrap_dataset, ParsedRelation

logger = logging.getLogger('discopy')


class TriDiscourseParser(object):

    def __init__(self):
        self.models = [DiscourseParser() for _ in range(3)]
        self.training_data = []

    def train(self, pdtb, parses, bs_ratio=0.75):
        self.training_data = bootstrap_dataset(pdtb, parses, n_straps=3, ratio=bs_ratio)
        for p_i, p in enumerate(self.models):
            p.train(*self.training_data[p_i])

    def train_more(self, pdtbs, parsed_docs):
        for p_i, p in enumerate(self.models):
            strap_pdtb, strap_parses = self.training_data[p_i]
            for doc_id, rels in pdtbs[p_i].items():
                strap_pdtb.extend(rels)
                strap_parses[doc_id] = parsed_docs[doc_id]
            p.train(strap_pdtb, strap_parses)

    def score(self, pdtb, parses):
        for p_i, p in enumerate(self.models):
            logger.info('Evaluation Parser {}'.format(p_i))
            p.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for p_i, p in enumerate(self.models):
            os.makedirs(os.path.join(path, str(p_i)), exist_ok=True)
            joblib.dump(p, os.path.join(path, str(p_i), 'parser.joblib'))
        joblib.dump(self.training_data, os.path.join(path, "training.data"))

    @staticmethod
    def from_path(path):
        parser = TriDiscourseParser()
        for p_i, p in enumerate(parser.models):
            os.makedirs(os.path.join(path, str(p_i)), exist_ok=True)
            parser.models[p_i] = joblib.load(os.path.join(path, str(p_i), 'parser.joblib'))
        parser.training_data = joblib.load(os.path.join(path, "training.data"))
        return parser

    def parse_documents(self, documents):
        relations = {}
        for p_i, p in enumerate(self.models):
            relations[p_i] = p.parse_documents(documents)

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
