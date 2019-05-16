import copy
import logging
from collections import defaultdict, Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#
# argument extractor
#
logger = logging.getLogger('discopy')

discourse_adverbial = {'accordingly', 'additionally', 'afterwards', 'also', 'alternatively', 'as a result',
                       'as an alternative', 'as well', 'besides', 'by comparison', 'by contrast',
                       'by then', 'consequently', 'conversely', 'earlier', 'either..or', 'except', 'finally',
                       'for example', 'for instance', 'further', 'furthermore', 'hence', 'in addition',
                       'in contrast', 'in fact', 'in other words', 'in particular', 'in short', 'in sum',
                       'in the end', 'in turn', 'indeed', 'instead', 'later', 'likewise', 'meantime',
                       'meanwhile', 'moreover', 'nevertheless', 'next', 'nonetheless',
                       'on the contrary', 'on the other hand', 'otherwise', 'overall', 'previously',
                       'rather', 'regardless', 'separately', 'similarly', 'simultaneously', 'specifically',
                       'still', 'thereafter', 'thereby', 'therefore', 'thus', 'ultimately', 'whereas'
                       }

coordinating_connective = {'and', 'but', 'else', 'if then', 'neither nor', 'nor',
                           'on the one hand on the other hand', 'or', 'plus', 'then', 'yet'}

subordinating_connective = {'after', 'although', 'as', 'as if', 'as long as', 'as soon as', 'as though', 'because',
                            'before', 'before and after', 'for', 'however', 'if', 'if and when', 'insofar as',
                            'lest', 'much as', 'now that', 'once', 'since', 'so', 'so that', 'though', 'till', 'unless',
                            'until', 'when', 'when and if', 'while'}
#
# connective
#
single_connectives = {'accordingly', 'additionally', 'after', 'afterward', 'also', 'alternatively',
                      'although', 'and', 'because', 'besides', 'but', 'consequently', 'conversely', 'earlier',
                      'else', 'except', 'finally', 'further', 'furthermore', 'hence', 'however', 'indeed',
                      'instead', 'later', 'lest', 'likewise', 'meantime', 'meanwhile', 'moreover',
                      'nevertheless', 'next', 'nonetheless', 'nor', 'once', 'or', 'otherwise', 'overall',
                      'plus', 'previously', 'rather', 'regardless', 'separately', 'similarly',
                      'simultaneously', 'since', 'specifically', 'still', 'then', 'thereafter', 'thereby',
                      'therefore', 'though', 'thus', 'till', 'ultimately', 'unless', 'until', 'whereas',
                      'while', 'yet'}

multi_connectives = list(map(lambda s: s.split(' '), [
    'as a result',
    'as a result',
    'as an alternative',
    'as if',
    'as long as',
    'as soon as',
    'as though',
    'as well',
    'as',
    'before and after',
    'before',
    'by comparison',
    'by contrast',
    # 'by',
    'for example',
    'for instance',
    'for',
    'if then',
    'if and when',
    'if',
    'in addition',
    'in contrast',
    'in fact',
    'in other words',
    'in particular',
    'in short',
    'in sum',
    'in the end',
    'in turn',
    'insofar as',
    'much as',
    'now that',
    'on the contrary',
    'on the other hand',
    'so that',
    'so',
    'when and if',
    'when',
    'neither nor',
    'either or',
]))

distant_connectives = list(map(lambda s: s.split(' '), [
    'if then',
    'neither nor',
    'either or',
]))

multi_connectives_first = {'as', 'before', 'by', 'for', 'either', 'if', 'in', 'insofar', 'much', 'neither',
                           'now', 'on', 'so', 'when'}


def jaccard_index(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1
    else:
        return len(a & b) / len(a | b)


def jaccard_distance(a, b):
    return 1 - jaccard_index(a, b)


class Relation:
    def __init__(self, arg1=None, arg2=None, conn=None, senses=None):
        self.arg1 = set(arg1) if arg1 else set()
        self.arg2 = set(arg2) if arg2 else set()
        self.conn = set(conn) if conn else set()
        self.senses = senses or []

    def __eq__(self, other):
        return (self.arg1 == other.arg1) and (self.arg2 == other.arg2) and (self.conn == other.conn)

    def __and__(self, other):
        r = Relation()
        r.arg1 = self.arg1 & other.arg1
        r.arg2 = self.arg2 & other.arg2
        r.conn = self.conn & other.conn
        return r

    def __or__(self, other):
        r = Relation()
        r.conn = self.conn | other.conn
        r.arg2 = (self.arg2 | other.arg2) - r.conn
        r.arg1 = (self.arg1 | other.arg1) - (r.conn | r.arg2)
        return r

    def __bool__(self):
        return bool(self.arg1) and bool(self.arg2)

    def __str__(self):
        return "Rel(arg1: <{}>, arg2: <{}>, conn: <{}>)".format(
            ",".join(map(str, sorted(self.arg1))),
            ",".join(map(str, sorted(self.arg2))),
            ",".join(map(str, sorted(self.conn)))
        )

    def is_explicit(self):
        return bool(self.conn)

    def __repr__(self):
        return "({},{},{})".format(list(self.arg1), list(self.arg2), list(self.conn))

    def distance(self, other):
        d_arg1 = jaccard_distance(self.arg1, other.arg1)
        d_arg2 = jaccard_distance(self.arg2 | self.conn, other.arg2 | other.conn)
        return (d_arg1 + d_arg2) / 2

    @staticmethod
    def from_conll(r):
        conn = [(i[2] if type(i) == list else i) for i in r['Connective']['TokenList']]
        arg1 = [(i[2] if type(i) == list else i) for i in r['Arg1']['TokenList']]
        arg2 = [(i[2] if type(i) == list else i) for i in r['Arg2']['TokenList']]
        senses = r['Sense']
        return Relation(arg1, arg2, conn, senses)


def convert_to_conll(document):
    p = {
        'DocID': document['DocID'],
        'sentences': []
    }
    for sentence in document['sentences']:
        if not sentence:
            continue
        p['sentences'].append({
            'words': [(word, {
                'CharacterOffsetBegin': sentence['SentenceOffset'] + offset,
                'CharacterOffsetEnd': sentence['SentenceOffset'] + offset + len(word),
                'Linkers': [],
                'PartOfSpeech': pos,
            }) for word, offset, pos in zip(sentence['Tokens'], sentence['Offset'], sentence['POS'])],
            'parsetree': sentence['Parse'],
            'dependencies': [(dep, "{}-{}".format(*node1), "{}-{}".format(*node2)) for (dep, node1, node2) in
                             sentence['Dep']]
        })
    return p


def load_relations(relations_json):
    relations = defaultdict(list)
    for r in relations_json:
        relations[r['DocID']].append(Relation.from_conll(r))
    return dict(relations)


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_items):
        return [item[self.key] for item in data_items]


def preprocess_relations(pdtb, level=2, filters=True):
    pdtb = copy.deepcopy(pdtb)
    for r in pdtb:
        sense = []
        for s in r['Sense']:
            sense.append('.'.join(s.split('.')[:level]))
        r['Sense'] = sense
    if filters:
        n_senses = Counter(s for r in pdtb for s in r['Sense'])
        logger.debug(n_senses)
        limit = len(pdtb) // (len(n_senses) * 10)
        logger.debug("Limit: {}".format(limit))
        pdtb = [r for r in pdtb if n_senses.get(r['Sense'][0], 0) > limit]
        logger.info("Preprocessed PDTB relations left: {}".format(len(pdtb)))
        logger.debug("Remaining classes")
        logger.debug({r['Sense'][0] for r in pdtb})
        logger.debug("Removed classes")
        logger.debug([k for k, v in n_senses.items() if v < limit])
    return pdtb


def init_logger(path='', log_level='INFO'):
    logger = logging.getLogger('discopy')
    logger.setLevel(log_level)
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s (%(levelname)s) %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    if path:
        fh = logging.FileHandler(path, mode='a')
        # create file handler which logs even debug messages
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info('=' * 50)
    logger.info('@  NEW RUN')
    logger.info('=' * 50)
    return logger


def bootstrap_dataset(pdtb, parses, n_straps=3, ratio=0.7, replace=True):
    n_samples = int(len(parses) * ratio)
    doc_ids = list(parses.keys())
    straps = []
    for i in range(n_straps):
        strap_doc_ids = set(np.random.choice(doc_ids, size=n_samples, replace=replace))
        strap_pdtb = [r for r in pdtb if r['DocID'] in strap_doc_ids]
        strap_parses = {doc_id: doc for doc_id, doc in parses.items() if doc_id in strap_doc_ids}
        straps.append((strap_pdtb, strap_parses))
    return straps
