import codecs
import json

import nltk

from discopy.argument_extract import ArgumentExtractClassifier
from discopy.argument_position import ArgumentPositionClassifier
from discopy.connective import ConnectiveClassifier
from discopy.explicit import ExplicitSenseClassifier


class DiscourseParser(object):

    def __init__(self):
        self.connClassifier = ConnectiveClassifier()
        self.argPosClassifier = ArgumentPositionClassifier()
        self.argExtractClassifier = ArgumentExtractClassifier()
        self.explicitClassifier = ExplicitSenseClassifier()

    def train(self, pdtb_dir, parses_dir):
        print('Load PDTB and WSJ')
        pdtb_train = [json.loads(s) for s in open(pdtb_dir, 'r').readlines()]
        parses_train = json.loads(open(parses_dir).read())
        print('Train Connective Classifier...')
        self.connClassifier.fit(pdtb_train, parses_train)
        print('Train ArgPosition Classifier...')
        self.argPosClassifier.fit(pdtb_train, parses_train)
        print('Train Argument Extractor...')
        self.argExtractClassifier.fit(pdtb_train, parses_train)
        print('Train Explicit Sense Classifier...')
        self.explicitClassifier.fit(pdtb_train, parses_train)

    def save(self, path):
        self.connClassifier.save(path)
        self.argPosClassifier.save(path)
        self.argExtractClassifier.save(path)
        self.explicitClassifier.save(path)

    def load(self, path):
        self.connClassifier.load(path)
        self.argPosClassifier.load(path)
        self.argExtractClassifier.load(path)
        self.explicitClassifier.load(path)

    def parse_file(self, input_file):
        documents = json.loads(codecs.open(input_file, mode='rb', encoding='utf-8').read())
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
        for i, sent in enumerate(doc['sentences']):
            sent_len = len(sent['words'])
            sent_parse = nltk.ParentedTree.fromstring(sent['parsetree'])
            if not sent_parse.leaves():
                continue
            j = 0
            while j < sent_len:
                relation = {
                    'Connective': {},
                    'Arg1': {},
                    'Arg2': {},
                    'Type': 'Explicit',
                    'Sent1': 0,
                    'Sent2': 0,
                }

                # CONNECTIVE CLASSIFIER
                connective = self.connClassifier.get_connective(sent_parse, sent['words'], j)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    j += 1
                    continue

                relation['Connective']['TokenList'] = [(0, 0, t_doc, i, t_sent) for t_doc, t_sent in
                                                       zip(range(token_id, token_id + len(connective)),
                                                           range(j, j + len(connective)))]
                relation['Connective']['RawText'] = ' '.join(connective)

                # ARGUMENT POSITION
                leaf_index = list(range(j, j + len(connective)))
                arg_pos = self.argPosClassifier.get_argument_position(sent_parse, ' '.join(connective),
                                                                      leaf_index)
                relation['ArgPos'] = arg_pos
                # If position poorly classified as PS, go to the next token
                if arg_pos == 'PS' and i == 0:
                    token_id += len(connective)
                    j += len(connective)
                    continue

                # ARGUMENT EXTRACTION
                if arg_pos == 'PS':
                    sent_prev = doc['sentences'][i - 1]
                    len_prev = len(sent_prev['words'])
                    relation['Arg1']['TokenList'] = list(range((sent_offset - len_prev), sent_offset - 1))
                    relation['Arg2']['TokenList'] = list(range(sent_offset, (sent_offset + sent_len) - 1))
                    inter_relations.add(i)
                elif arg_pos == 'SS':
                    arg1, arg2 = self.argExtractClassifier.extract_arguments(sent_parse, relation)
                    relation['Arg1']['TokenList'] = [i + token_id - j for i in arg1]
                    relation['Arg2']['TokenList'] = [i + token_id - j for i in arg2]

                # EXPLICIT SENSE
                relation['Sense'] = self.explicitClassifier.get_explicit_sense(relation, sent)
                output.append(relation)
                token_id += len(connective)
                j += len(connective)
            sent_offset += sent_len

        token_id = 0
        for i, sent in enumerate(doc['sentences']):
            if i == 0 or i in inter_relations:
                token_id += len(sent['words'])
                continue

            sent_prev = doc['sentences'][i - 1]

            relation = {
                'Connective': {
                    'TokenList': []
                },
                'Arg1': {
                    'TokenList': list(range((token_id - len(sent_prev['words'])), token_id - 1))
                },
                'Arg2': {
                    'TokenList': list(range(token_id, (token_id + len(sent['words']) - 1)))
                },
                'Type': 'Implicit'
            }
            # TODO add implicit sense classification
            relation['Sense'] = [None]
            output.append(relation)

            token_id += len(sent['words'])
        return output
