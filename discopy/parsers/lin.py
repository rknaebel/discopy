import logging
import os

import joblib

from discopy.labeling.argument_position import ArgumentPositionClassifier
from discopy.labeling.connective import ConnectiveClassifier
from discopy.labeling.extract.lin_arg_extract import LinArgumentExtractClassifier
from discopy.parsers.parser import AbstractBaseParser
from discopy.parsers.utils import get_token_list2, get_raw_tokens2
from discopy.sense.explicit import ExplicitSenseClassifier
from discopy.sense.nonexplicit import NonExplicitSenseClassifier
from discopy.utils import ParsedRelation

logger = logging.getLogger('discopy')


class LinParser(AbstractBaseParser):

    def __init__(self):
        self.connective_clf = ConnectiveClassifier()
        self.arg_pos_clf = ArgumentPositionClassifier()
        self.arg_extract_clf = LinArgumentExtractClassifier()
        self.explicit_clf = ExplicitSenseClassifier()
        self.non_explicit_clf = NonExplicitSenseClassifier()

    def fit(self, pdtb, parses, pdtb_val=None, parses_val=None):
        logger.info('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses)
        logger.info('Train ArgPosition Classifier...')
        self.arg_pos_clf.fit(pdtb, parses)
        logger.info('Train Argument Extractor...')
        self.arg_extract_clf.fit(pdtb, parses)
        logger.info('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses)
        logger.info('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses)

    def score(self, pdtb, parses):
        self.connective_clf.score(pdtb, parses)
        self.arg_pos_clf.score(pdtb, parses)
        self.arg_extract_clf.score(pdtb, parses)
        self.explicit_clf.score(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        self.connective_clf.save(path)
        self.arg_pos_clf.save(path)
        self.arg_extract_clf.save(path)
        self.explicit_clf.save(path)
        self.non_explicit_clf.save(path)
        joblib.dump(self, os.path.join(path, 'parser.joblib'))

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
        if os.path.exists(os.path.join(path, 'parser.joblib')):
            parser = joblib.load(os.path.join(path, 'parser.joblib'))
            self.connective_clf = parser.connective_clf
            self.arg_pos_clf = parser.arg_pos_clf
            self.arg_extract_clf = parser.arg_extract_clf
            self.explicit_clf = parser.explicit_clf
            self.non_explicit_clf = parser.non_explicit_clf
        else:
            self.connective_clf.load(path)
            self.arg_pos_clf.load(path)
            self.arg_extract_clf.load(path)
            self.explicit_clf.load(path)
            self.non_explicit_clf.load(path)

    @staticmethod
    def from_path(path):
        return joblib.load(os.path.join(path, 'parser.joblib'))

    def parse_connectives(self, doc):
        relations = []
        token_id = 0
        sent_offset = 0
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

        for sent_id, sent in enumerate(doc['sentences']):
            sent_len = len(sent['words'])
            ptree = sent['parsetree']
            if ptree is None or not ptree.leaves():
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

        for relation in relations:
            sent_id = relation.Connective.get_sentence_ids()[0]
            ptree = doc['sentences'][sent_id]['parsetree']
            if ptree is None or not ptree.leaves():
                logger.warning('Failed on empty tree')
                continue

            # ARGUMENT POSITION
            leaf_index = relation.Connective.get_local_ids()
            arg_pos, arg_pos_confidence = self.arg_pos_clf.get_argument_position(ptree, relation.Connective.RawText,
                                                                                 leaf_index)
            # If position poorly classified as PS, go to the next token
            if arg_pos == 'PS' and sent_id == 0:
                continue

            # ARGUMENT EXTRACTION
            arg1, arg2, arg1_c, arg2_c = self.arg_extract_clf.extract_arguments(ptree, relation.to_dict(), arg_pos)
            if arg_pos == 'PS':
                arg1 = [i for i, (w, s_i, w_i) in enumerate(doc_words) if s_i == (sent_id - 1)]

                relation.Arg1.TokenList = get_token_list2(doc_words, arg1)
                relation.Arg1.RawText = get_raw_tokens2(doc_words, arg1)
                relation.Arg2.TokenList = get_token_list2(doc_words, arg2, sent_id)
                relation.Arg2.RawText = get_raw_tokens2(doc_words, arg2, sent_id)
            elif arg_pos == 'SS':
                relation.Arg1.TokenList = get_token_list2(doc_words, arg1, sent_id)
                relation.Arg1.RawText = get_raw_tokens2(doc_words, arg1, sent_id)
                relation.Arg2.TokenList = get_token_list2(doc_words, arg2, sent_id)
                relation.Arg2.RawText = get_raw_tokens2(doc_words, arg2, sent_id)
            else:
                logger.error('Unknown Argument Position: ' + arg_pos)
                # raise ValueError('Unknown Argument Position')

        return [r for r in relations if r.is_valid()]

    def parse_explicit_sense(self, doc, relations):
        for relation in relations:
            sent_id = relation.Connective.get_sentence_ids()[0]
            ptree = doc['sentences'][sent_id]['parsetree']
            if ptree is None or not ptree.leaves():
                logger.warning('Failed on empty tree')
                continue
            explicit, explicit_c = self.explicit_clf.get_sense(relation.to_dict(), ptree)
            relation.Sense = explicit
        return relations

    def parse_implicit_arguments(self, doc, relations):
        inter_relations = set()
        for relation in relations:
            arg1_idxs = relation.Arg1.get_sentence_ids()
            arg2_idxs = relation.Arg2.get_sentence_ids()
            if not arg1_idxs or not arg2_idxs:
                continue
            elif max(arg1_idxs) == min(arg2_idxs) - 1:
                inter_relations.add(min(arg2_idxs) - 1)
            elif max(arg2_idxs) == min(arg1_idxs) - 1:
                inter_relations.add(min(arg1_idxs) - 1)

        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        token_id = 0
        sent_lengths = [0]
        for sent_id, sent in enumerate(doc['sentences']):
            sent_lengths.append(len(sent['words']))
            if sent_id == 0 or sent_id in inter_relations:
                token_id += len(sent['words'])
                continue
            arg1_idxs = list(range(sum(sent_lengths[:-2]), sum(sent_lengths[:-1])))
            arg2_idxs = list(range(sum(sent_lengths[:-1]), sum(sent_lengths)))
            relation = ParsedRelation()
            relation.Type = 'Implicit'
            relation.Arg1.TokenList = get_token_list2(doc_words, arg1_idxs)
            relation.Arg2.TokenList = get_token_list2(doc_words, arg2_idxs)
            relation.Arg1.RawText = get_raw_tokens2(doc_words, arg1_idxs)
            relation.Arg2.RawText = get_raw_tokens2(doc_words, arg2_idxs)
            relations.append(relation)

            token_id += len(sent['words'])
        return relations

    def parse_implicit_senses(self, doc, relations):
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        for relation in filter(lambda r: not r.is_explicit(), relations):
            sent_id = relation.Arg2.get_sentence_ids()[0]
            sent = doc['sentences'][sent_id]
            ptree = sent['parsetree']
            ptree_prev = doc['sentences'][sent_id - 1]['parsetree']
            dtree = sent['dependencies']
            dtree_prev = doc['sentences'][sent_id - 1]['dependencies']

            if ptree is None or not ptree.leaves() or ptree_prev is None or not ptree_prev.leaves():
                continue

            arg1_idxs = [t[2] for t in relation.Arg1.TokenList]
            arg2_idxs = [t[2] for t in relation.Arg2.TokenList]

            arg1 = [doc_words[i][0][0] for i in arg1_idxs]
            arg2 = [doc_words[i][0][0] for i in arg2_idxs]
            sense, sense_c = self.non_explicit_clf.get_sense(ptree_prev, ptree, dtree_prev, dtree, arg1, arg2)
            relation.Sense = sense
            if relation.Sense in ('EntRel', 'NoRel'):
                relation.Type = relation.Sense
            else:
                relation.Type = 'Implicit'
        return relations

    def parse_doc(self, doc):
        relations = self.parse_connectives(doc)
        relations = self.parse_explicit_arguments(doc, relations)
        relations = self.parse_explicit_sense(doc, relations)
        relations = self.parse_implicit_arguments(doc, relations)
        relations = self.parse_implicit_senses(doc, relations)
        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_conll() for r in relations]
        return relations


class LinArgumentParser(LinParser):

    def __init__(self):
        self.connective_clf = ConnectiveClassifier()
        self.arg_pos_clf = ArgumentPositionClassifier()
        self.arg_extract_clf = LinArgumentExtractClassifier()

    def fit(self, pdtb, parses, pdtb_val=None, parses_val=None):
        logger.info('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses)
        logger.info('Train ArgPosition Classifier...')
        self.arg_pos_clf.fit(pdtb, parses)
        logger.info('Train Argument Extractor...')
        self.arg_extract_clf.fit(pdtb, parses)

    def score(self, pdtb, parses):
        self.connective_clf.score(pdtb, parses)
        self.arg_pos_clf.score(pdtb, parses)
        self.arg_extract_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        self.connective_clf.save(path)
        self.arg_pos_clf.save(path)
        self.arg_extract_clf.save(path)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
        self.connective_clf.load(path)
        self.arg_pos_clf.load(path)
        self.arg_extract_clf.load(path)

    def parse_doc(self, doc):
        relations = self.parse_connectives(doc)
        relations = self.parse_explicit_arguments(doc, relations)
        relations = [r.to_conll() for r in relations]
        return relations
