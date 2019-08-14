import logging
import os
import ujson as json
from collections import Counter

import joblib
import nltk

from discopy.labeling.connective import ConnectiveClassifier
from discopy.labeling.neural.arg_extract import ArgumentExtractBiLSTM, ArgumentExtractBiLSTMwithConn
from discopy.sense.explicit import ExplicitSenseClassifier
from discopy.sense.nonexplicit import NonExplicitSenseClassifier
from discopy.utils import init_logger
from discopy.parsers.utils import get_raw_tokens, get_token_list2

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

    def __init__(self):
        self.Connective = Relation.Span()
        self.Arg1 = Relation.Span()
        self.Arg2 = Relation.Span()
        self.Type = ''
        self.Sense = ''
        self.ptree = None

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
            'Type': self.Type,
            'Sense': [self.Sense],
            'ptree': self.ptree
        }

    def to_conll(self):
        r = self.to_dict()
        del r['ptree']
        return r

    def is_explicit(self):
        return self.Connective.RawText != ''

    def __str__(self):
        return "Relation({}|{}|arg1:{}|arg2:{}|conn:{})".format(self.Type, self.Sense,
                                                                self.Arg1.get_global_ids(),
                                                                self.Arg2.get_global_ids(),
                                                                self.Connective.get_global_ids())


class AbstractBiLSTMDiscourseParser:
    def __init__(self, n_estimators=1):
        self.hidden_size = 128
        self.rnn_size = 512
        self.window_length = 100

        self.explicit_clf = ExplicitSenseClassifier(n_estimators=n_estimators)
        self.non_explicit_clf = NonExplicitSenseClassifier(n_estimators=n_estimators)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Explicit Sense Classifier...')
        self.explicit_clf.fit(pdtb, parses)
        logger.info('Train Non-Explicit Sense Classifier...')
        self.non_explicit_clf.fit(pdtb, parses)

    def score(self, pdtb, parses):
        self.explicit_clf.score(pdtb, parses)
        self.non_explicit_clf.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.explicit_clf.save(path)
        self.non_explicit_clf.save(path)

    def load(self, path, parses):
        if not os.path.exists(path):
            raise FileNotFoundError('Path not found')
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
        raise NotImplementedError("parse_doc not found")

    def parse_explicit_sense(self, doc, relations):
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
            explicit, explicit_c = self.explicit_clf.get_sense(relation.to_dict(), ptree)
            relation.Sense = explicit
            relation.ptree = None
        return relations

    def parse_implicit_senses(self, doc, relations):
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        for relation in filter(lambda r: not r.is_explicit(), relations):
            sent_id = relation.Arg2.get_sentence_ids()[0]
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

            arg1_idxs = [t[2] for t in relation.Arg1.TokenList]
            arg2_idxs = [t[2] for t in relation.Arg2.TokenList]

            arg1 = [doc_words[i][0][0] for i in arg1_idxs]
            arg2 = [doc_words[i][0][0] for i in arg2_idxs]
            sense, sense_c = self.non_explicit_clf.get_sense(ptree_prev, ptree, dtree_prev, dtree, arg1, arg2)
            relation.Sense = sense
        return relations


class BiLSTMDiscourseParser3(AbstractBiLSTMDiscourseParser):

    def __init__(self, n_estimators=1, no_crf=False):
        super().__init__(n_estimators)
        self.arg_labeler = ArgumentExtractBiLSTM(window_length=self.window_length,
                                                 hidden_dim=self.hidden_size,
                                                 rnn_dim=self.rnn_size, no_rnn=False,
                                                 no_dense=False, no_crf=no_crf, explicits_only=False)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=10)
        super().train(pdtb, parses, pdtb_val, parses_val)

    def score(self, pdtb, parses):
        super().score(pdtb, parses)
        self.arg_labeler.score(pdtb, parses)

    def save(self, path):
        super().save(path)
        self.arg_labeler.save(path)

    def load(self, path, parses):
        super().load(path, parses)
        self.arg_labeler.init_model(parses)
        self.arg_labeler.load(path)

    def parse_arguments(self, doc):
        relations = []
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                            strides=1, max_distance=0.5)
        for r in arguments_pred:
            relation = Relation()
            relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
            relation.Arg1.RawText = get_raw_tokens(doc_words, r.arg1)
            relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
            relation.Arg2.RawText = get_raw_tokens(doc_words, r.arg2)
            if r.conn:
                # TODO make more elegant... just some workaround
                sent_id = Counter(i[3] for i in get_token_list2(doc_words, r.conn)).most_common(1)[0][0]
                conn = [i[2] for i in get_token_list2(doc_words, r.conn) if i[3] == sent_id]
                relation.Connective.TokenList = get_token_list2(doc_words, conn)
                relation.Connective.RawText = get_raw_tokens(doc_words, conn)
            relations.append(relation)
        return relations

    def parse_doc(self, doc):
        relations = self.parse_arguments(doc)

        explicits = [r for r in relations if r.is_explicit()]
        non_explicits = [r for r in relations if not r.is_explicit()]

        explicits = self.parse_explicit_sense(doc, explicits)
        non_explicits = self.parse_implicit_senses(doc, non_explicits)
        relations = explicits + non_explicits

        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_dict() for r in relations]
        return relations


class BiLSTMDiscourseParser2(AbstractBiLSTMDiscourseParser):

    def __init__(self, n_estimators=1, no_crf=False):
        super().__init__(n_estimators)
        self.arg_labeler = ArgumentExtractBiLSTM(window_length=100, hidden_dim=128, rnn_dim=512, no_rnn=False,
                                                 no_dense=False, no_crf=no_crf, explicits_only=True)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=10)
        super().train(pdtb, parses, pdtb_val, parses_val)

    def score(self, pdtb, parses):
        super().score(pdtb, parses)
        self.arg_labeler.score(pdtb, parses)

    def save(self, path):
        super().save(path)
        self.arg_labeler.save(path)

    def load(self, path, parses):
        super().load(path, parses)
        self.arg_labeler.init_model(parses)
        self.arg_labeler.load(path)

    def parse_explicit_arguments(self, doc):
        relations = []
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                            strides=1, max_distance=0.5)
        for r in arguments_pred:
            if not r.conn:
                continue
            relation = Relation()
            sent_id = Counter(i[3] for i in get_token_list2(doc_words, r.conn)).most_common(1)[0][0]
            conn = [i[2] for i in get_token_list2(doc_words, r.conn) if i[3] == sent_id]
            relation.Connective.TokenList = get_token_list2(doc_words, conn)
            relation.Connective.RawText = get_raw_tokens(doc_words, conn)
            relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
            relation.Arg1.RawText = get_raw_tokens(doc_words, r.arg1)
            relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
            relation.Arg2.RawText = get_raw_tokens(doc_words, r.arg2)
            relations.append(relation)
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
            relation = Relation()
            relation.Type = 'Implicit'
            relation.Arg1.TokenList = get_token_list2(doc_words, arg1_idxs)
            relation.Arg2.TokenList = get_token_list2(doc_words, arg2_idxs)
            relation.Arg1.RawText = get_raw_tokens(doc_words, arg1_idxs)
            relation.Arg2.RawText = get_raw_tokens(doc_words, arg2_idxs)
            relations.append(relation)

            token_id += len(sent['words'])
        return relations

    def parse_doc(self, doc):
        relations = self.parse_explicit_arguments(doc)
        relations = self.parse_explicit_sense(doc, relations)
        relations = self.parse_implicit_arguments(doc, relations)
        relations = self.parse_implicit_senses(doc, relations)
        # TODO remove later
        # transforms the relation structure into dict format
        # just for now as long as the relation structure is used locally only
        relations = [r.to_dict() for r in relations]
        return relations


class BiLSTMDiscourseParser1(AbstractBiLSTMDiscourseParser):
    """
    Extracts explicit arguments based on the connective prediction
    """

    def __init__(self, n_estimators=1, no_crf=False):
        super().__init__(n_estimators)
        self.connective_clf = ConnectiveClassifier(n_estimators=n_estimators)
        self.arg_labeler = ArgumentExtractBiLSTMwithConn(window_length=self.window_length,
                                                         hidden_dim=self.hidden_size,
                                                         rnn_dim=self.rnn_size,
                                                         no_rnn=False,
                                                         no_dense=False, no_crf=no_crf)

    def train(self, pdtb, parses, pdtb_val, parses_val):
        logger.info('Train Connective Classifier...')
        self.connective_clf.fit(pdtb, parses)
        logger.info('Train Argument Extractor...')
        self.arg_labeler.fit(pdtb, parses, pdtb_val, parses_val, epochs=1)
        super().train(pdtb, parses, pdtb_val, parses_val)

    def score(self, pdtb, parses):
        super().score(pdtb, parses)
        self.connective_clf.score(pdtb, parses)
        self.arg_labeler.score(pdtb, parses)

    def save(self, path):
        super().save(path)
        self.connective_clf.save(path)
        self.arg_labeler.save(path)

    def load(self, path, parses):
        super().load(path, parses)
        self.connective_clf.load(path)
        self.arg_labeler.init_model(parses)
        self.arg_labeler.load(path)

    def parse_connectives(self, doc):
        relations = []
        token_id = 0
        sent_offset = 0
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]

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
                relation = Relation()
                relation.Type = 'Explicit'
                connective, connective_confidence = self.connective_clf.get_connective(ptree, sent['words'],
                                                                                       current_token)
                # whenever a position is not identified as connective, go to the next token
                if not connective:
                    token_id += 1
                    current_token += 1
                    continue
                conn_idxs = [token_id + i for i in range(len(connective))]
                relation.Connective.TokenList = get_token_list2(doc_words, conn_idxs)
                relation.Connective.RawText = get_raw_tokens(doc_words, conn_idxs)

                relations.append(relation)
                token_id += len(connective)
                current_token += len(connective)
            sent_offset += sent_len
        return relations

    def parse_explicit_arguments(self, doc, relations):
        doc_words = [(w, s_i, w_i) for s_i, s in enumerate(doc['sentences']) for w_i, w in enumerate(s['words'])]
        arguments_pred = self.arg_labeler.extract_arguments([w[0][0] for w in doc_words],
                                                            [r.to_dict() for r in relations],
                                                            max_distance=0.5)
        for relation, r in zip(relations, arguments_pred):
            relation.Arg1.TokenList = get_token_list2(doc_words, r.arg1)
            relation.Arg1.RawText = get_raw_tokens(doc_words, r.arg1)
            relation.Arg2.TokenList = get_token_list2(doc_words, r.arg2)
            relation.Arg2.RawText = get_raw_tokens(doc_words, r.arg2)
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
            relation = Relation()
            relation.Type = 'Implicit'
            relation.Arg1.TokenList = get_token_list2(doc_words, arg1_idxs)
            relation.Arg2.TokenList = get_token_list2(doc_words, arg2_idxs)
            relation.Arg1.RawText = get_raw_tokens(doc_words, arg1_idxs)
            relation.Arg2.RawText = get_raw_tokens(doc_words, arg2_idxs)
            relations.append(relation)

            token_id += len(sent['words'])
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
        relations = [r.to_dict() for r in relations]
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
    parser.load('bilstm-tmp', parses_train)
    # parser.train(pdtb_train, parses_train, pdtb_val, parses_val)
    # parser.save('bilstm-tmp')
    print(parser.parse_doc(parses_train['wsj_1843']))
    # logger.info('Evaluation on VAL')
    # parser.score(pdtb_val, parses_val)
    # logger.info('Evaluation on TEST')
    # parser.score(pdtb_test, parses_test)
