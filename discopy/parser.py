import codecs
import json

import nltk

from discopy.argument_extract import ArgumentExtractClassifier
from discopy.argument_position import ArgumentPositionClassifier
from discopy.connective import ConnectiveClassifier
from discopy.explicit import ExplicitSenseClassifier
from discopy.nonexplicit import NonExplicitSenseClassifier


class DiscourseParser(object):

    def __init__(self):
        self.connective_clf = ConnectiveClassifier()
        self.arg_pos_clf = ArgumentPositionClassifier()
        self.arg_extract_clf = ArgumentExtractClassifier()
        self.explicit_clf = ExplicitSenseClassifier()
        self.non_explicit_clf = NonExplicitSenseClassifier()

    def train(self, pdtb, parses, epochs=10):
        print('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses, max_iter=epochs)
        print('Train ArgPosition Classifier...')
        self.arg_pos_clf.fit(pdtb, parses, max_iter=epochs)
        print('Train Argument Extractor...')
        self.arg_extract_clf.fit(pdtb, parses, max_iter=epochs)
        print('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses, max_iter=epochs)
        print('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses, max_iter=epochs)

    def save(self, path):
        self.connective_clf.save(path)
        self.arg_pos_clf.save(path)
        self.arg_extract_clf.save(path)
        self.explicit_clf.save(path)
        self.non_explicit_clf.save(path)

    def load(self, path):
        self.connective_clf.load(path)
        self.arg_pos_clf.load(path)
        self.arg_extract_clf.load(path)
        self.explicit_clf.load(path)
        self.non_explicit_clf.load(path)

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
            try:
                sent_parse = nltk.ParentedTree.fromstring(sent['parsetree'])
            except ValueError:
                print('Failed to parse doc {} idx {}'.format(doc['DocID'], i))
                token_id += sent_len
                continue
            if not sent_parse.leaves():
                token_id += sent_len
                continue
            j = 0
            while j < sent_len:
                relation = {
                    'Connective': {},
                    'Arg1': {},
                    'Arg2': {},
                    'Type': 'Explicit',
                    'Confidences': {}
                }

                # CONNECTIVE CLASSIFIER
                connective, connective_confidence = self.connective_clf.get_connective(sent_parse, sent['words'], j)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    j += 1
                    continue

                relation['Connective']['TokenList'] = [(0, 0, t_doc, i, t_sent) for t_doc, t_sent in
                                                       zip(range(token_id, token_id + len(connective)),
                                                           range(j, j + len(connective)))]
                relation['Connective']['RawText'] = ' '.join(connective)
                relation['Confidences']['Connective'] = connective_confidence

                # ARGUMENT POSITION
                leaf_index = list(range(j, j + len(connective)))
                arg_pos, arg_pos_confidence = self.arg_pos_clf.get_argument_position(sent_parse, ' '.join(connective),
                                                                                     leaf_index)
                relation['ArgPos'] = arg_pos
                relation['Confidences']['ArgPos'] = arg_pos_confidence

                # If position poorly classified as PS, go to the next token
                if arg_pos == 'PS' and i == 0:
                    token_id += len(connective)
                    j += len(connective)
                    continue

                # ARGUMENT EXTRACTION
                if arg_pos == 'PS':
                    sent_prev = doc['sentences'][i - 1]
                    _, arg2, arg1_c, arg2_c = self.arg_extract_clf.extract_arguments(sent_parse, relation)
                    len_prev = len(sent_prev['words'])
                    relation['Arg1']['TokenList'] = list(range((token_id - len_prev), token_id - 1))
                    relation['Arg2']['TokenList'] = [i + token_id - j for i in arg2]
                    relation['Confidences']['Arg1'] = 1.0
                    relation['Confidences']['Arg2'] = arg2_c
                    inter_relations.add(i)
                elif arg_pos == 'SS':
                    arg1, arg2, arg1_c, arg2_c = self.arg_extract_clf.extract_arguments(sent_parse, relation)
                    relation['Arg1']['TokenList'] = [i + token_id - j for i in arg1]
                    relation['Arg2']['TokenList'] = [i + token_id - j for i in arg2]
                    relation['Confidences']['Arg1'] = arg1_c
                    relation['Confidences']['Arg2'] = arg2_c

                # EXPLICIT SENSE
                explicit, explicit_c = self.explicit_clf.get_sense(relation, sent_parse)
                relation['Sense'] = [explicit]
                relation['Confidences']['Sense'] = explicit_c
                output.append(relation)
                token_id += len(connective)
                j += len(connective)
            sent_offset += sent_len

        token_id = 0
        for i, sent in enumerate(doc['sentences']):
            if i == 0 or i in inter_relations:
                token_id += len(sent['words'])
                continue

            sent_parse = nltk.ParentedTree.fromstring(sent['parsetree'])
            sent_prev_parse = nltk.ParentedTree.fromstring(doc['sentences'][i - 1]['parsetree'])

            if not sent_parse.leaves() or not sent_prev_parse.leaves():
                continue

            sense, sense_c = self.non_explicit_clf.get_sense([sent_prev_parse, sent_parse])
            relation = {
                'Connective': {
                    'TokenList': []
                },
                'Arg1': {
                    'TokenList': list(range((token_id - len(sent_prev_parse.leaves())), token_id - 1))
                },
                'Arg2': {
                    'TokenList': list(range(token_id, (token_id + len(sent_parse.leaves()) - 1)))
                },
                'Type': 'Implicit',
                'Sense': [sense],
                'Confidences': {
                    'Sense': sense_c
                }
            }
            output.append(relation)

            token_id += len(sent['words'])
        return output
