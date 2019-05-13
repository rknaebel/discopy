import logging
import os

import joblib
import nltk
import numpy as np
from discopy.argument_extract import ArgumentExtractClassifier
from discopy.argument_position import ArgumentPositionClassifier
from discopy.connective import ConnectiveClassifier
from discopy.explicit import ExplicitSenseClassifier
from discopy.nonexplicit import NonExplicitSenseClassifier

logger = logging.getLogger('discopy')


def get_token_list(doc_words, tokens, sent_id, sent_off):
    return [[doc_words[sent_off + t][1]['CharacterOffsetBegin'],
             doc_words[sent_off + t][1]['CharacterOffsetEnd'],
             sent_off + t, sent_id, t] for t in tokens]


def get_raw_tokens(doc_words, idxs):
    return " ".join([doc_words[i[2]][0] for i in idxs])


class DiscourseParser(object):

    def __init__(self, n_estimators=1):
        self.connective_clf = ConnectiveClassifier(n_estimators=n_estimators)
        self.arg_pos_clf = ArgumentPositionClassifier(n_estimators=n_estimators)
        self.arg_extract_clf = ArgumentExtractClassifier(n_estimators=n_estimators)
        self.explicit_clf = ExplicitSenseClassifier(n_estimators=n_estimators)
        self.non_explicit_clf = NonExplicitSenseClassifier(n_estimators=n_estimators)

    def train(self, pdtb, parses):
        logger.info('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses)
        self.connective_clf.score(pdtb, parses)
        logger.info('Train ArgPosition Classifier...')
        self.arg_pos_clf.fit(pdtb, parses)
        self.arg_pos_clf.score(pdtb, parses)
        logger.info('Train Argument Extractor...')
        self.arg_extract_clf.fit(pdtb, parses)
        self.arg_extract_clf.score(pdtb, parses)
        logger.info('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses)
        self.explicit_clf.score(pdtb, parses)
        logger.info('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def score(self, pdtb, parses):
        self.connective_clf.score(pdtb, parses)
        self.arg_pos_clf.score(pdtb, parses)
        self.arg_extract_clf.score(pdtb, parses)
        self.explicit_clf.score(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self, os.path.join(path, 'parser.joblib'))

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
        output = []
        token_id = 0
        sent_offset = 0
        inter_relations = set()
        doc_words = [w for s in doc['sentences'] for w in s['words']]
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
                    'Confidences': {}
                }

                # CONNECTIVE CLASSIFIER
                connective, connective_confidence = self.connective_clf.get_connective(ptree, sent['words'],
                                                                                       current_token)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    current_token += 1
                    continue

                relation['Connective']['TokenList'] = get_token_list(doc_words, [current_token + t_sent for t_sent in
                                                                                 range(len(connective))], sent_id,
                                                                     sent_offset)
                relation['Connective']['RawText'] = get_raw_tokens(doc_words, relation['Connective']['TokenList'])
                relation['Confidences']['Connective'] = connective_confidence

                # ARGUMENT POSITION
                leaf_index = [i[4] for i in relation['Connective']['TokenList']]
                arg_pos, arg_pos_confidence = self.arg_pos_clf.get_argument_position(ptree, ' '.join(connective),
                                                                                     leaf_index)
                relation['ArgPos'] = arg_pos
                relation['Confidences']['ArgPos'] = arg_pos_confidence

                # If position poorly classified as PS, go to the next token
                if arg_pos == 'PS' and sent_id == 0:
                    token_id += len(connective)
                    current_token += len(connective)
                    continue

                # ARGUMENT EXTRACTION
                if arg_pos == 'PS':
                    sent_prev = doc['sentences'][sent_id - 1]
                    _, arg2, arg1_c, arg2_c = self.arg_extract_clf.extract_arguments(ptree, relation)
                    len_prev = len(sent_prev['words'])
                    relation['Arg1']['TokenList'] = get_token_list(doc_words, list(range(len_prev)), sent_id - 1,
                                                                   sent_offset - len_prev)
                    relation['Arg2']['TokenList'] = get_token_list(doc_words, arg2, sent_id, sent_offset)
                    relation['Arg1']['RawText'] = get_raw_tokens(doc_words, relation['Arg1']['TokenList'])
                    relation['Arg2']['RawText'] = get_raw_tokens(doc_words, relation['Arg2']['TokenList'])
                    relation['Confidences']['Arg2'] = arg2_c
                    inter_relations.add(sent_id)
                elif arg_pos == 'SS':
                    arg1, arg2, arg1_c, arg2_c = self.arg_extract_clf.extract_arguments(ptree, relation)
                    relation['Arg1']['TokenList'] = get_token_list(doc_words, arg1, sent_id, sent_offset)
                    relation['Arg2']['TokenList'] = get_token_list(doc_words, arg2, sent_id, sent_offset)
                    relation['Arg1']['RawText'] = get_raw_tokens(doc_words, relation['Arg1']['TokenList'])
                    relation['Arg2']['RawText'] = get_raw_tokens(doc_words, relation['Arg2']['TokenList'])
                    relation['Confidences']['Arg1'] = arg1_c
                    relation['Confidences']['Arg2'] = arg2_c
                else:
                    logger.error('Unknown Argument Position: ' + arg_pos)
                    raise ValueError('Unknown Argument Position')

                # EXPLICIT SENSE
                explicit, explicit_c = self.explicit_clf.get_sense(relation, ptree)
                relation['Sense'] = [explicit]
                relation['Confidences']['Sense'] = explicit_c
                output.append(relation)
                token_id += len(connective)
                current_token += len(connective)
            sent_offset += sent_len

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

            relation = {
                'Connective': {
                    'TokenList': []
                },
                'Arg1': {
                    'TokenList': get_token_list(doc_words, list(range(len(ptree_prev.leaves()))), sent_id - 1,
                                                sum(sent_lengths[:-2]))
                },
                'Arg2': {
                    'TokenList': get_token_list(doc_words, list(range(len(ptree.leaves()))), sent_id,
                                                sum(sent_lengths[:-1]))
                },
                'Type': 'Implicit',
                'Sense': [],
                'Confidences': {}
            }
            relation['Arg1']['RawText'] = get_raw_tokens(doc_words, relation['Arg1']['TokenList'])
            relation['Arg2']['RawText'] = get_raw_tokens(doc_words, relation['Arg2']['TokenList'])
            arg1 = [doc_words[i[2]][0] for i in relation['Arg1']['TokenList']]
            arg2 = [doc_words[i[2]][0] for i in relation['Arg2']['TokenList']]
            sense, sense_c = self.non_explicit_clf.get_sense(ptree_prev, ptree, dtree_prev, dtree, arg1, arg2)
            relation['Sense'] = [sense]
            relation['Confidences']['Sense'] = sense_c
            output.append(relation)

            token_id += len(sent['words'])

        for r in output:
            r['Confidence'] = np.mean(list(r['Confidences'].values()))
        return output
