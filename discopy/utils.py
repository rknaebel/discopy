#
# argument extractor
#
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
