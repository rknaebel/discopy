import logging

import nltk

from discopy.labeling.extract.gosh_arg_extract import GoshArgumentExtract
from discopy.parsers.lin import LinParser
from discopy.parsers.utils import get_token_list2, get_raw_tokens2

logger = logging.getLogger('discopy')


def get_token_list(doc_words, tokens, sent_id, sent_off):
    return [[doc_words[t][1]['CharacterOffsetBegin'],
             doc_words[t][1]['CharacterOffsetEnd'],
             sent_off + t, sent_id, t] for t in tokens]


class GoshParser(LinParser):

    def __init__(self):
        super().__init__()
        self.arg_extract_clf = GoshArgumentExtract()

    def parse_explicit_arguments(self, doc, relations):
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

        for relation in relations:
            sent_id = relation.Connective.get_sentence_ids()[0]
            try:
                ptree = nltk.ParentedTree.fromstring(doc['sentences'][sent_id]['parsetree'])
            except ValueError:
                logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                continue
            if not ptree.leaves():
                logger.warning('Failed on empty tree')
                continue
            arg1, arg2, arg1_c, arg2_c = self.arg_extract_clf.extract_arguments(doc, relation.to_dict())

            relation.Arg1.TokenList = get_token_list2(doc_words, arg1)
            relation.Arg1.RawText = get_raw_tokens2(doc_words, arg1)
            relation.Arg2.TokenList = get_token_list2(doc_words, arg2)
            relation.Arg2.RawText = get_raw_tokens2(doc_words, arg2)
        return [r for r in relations if r.is_valid()]

    def parse_doc(self, doc):
        relations = self.parse_connectives(doc)
        relations = self.parse_explicit_sense(doc, relations)
        relations = self.parse_explicit_arguments(doc, relations)
        relations = self.parse_implicit_arguments(doc, relations)
        relations = self.parse_implicit_senses(doc, relations)
        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_conll() for r in relations]
        return relations
