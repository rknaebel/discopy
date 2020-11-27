from collections import namedtuple
from typing import List

import nltk


class Document:
    def __init__(self, doc_id, sentences, relations):
        self.doc_id = doc_id
        self.sentences: List[Sentence] = sentences
        self.relations: List[Relation] = relations
        self.text = '\n'.join([s.get_text() for s in self.sentences])

    def to_json(self):
        return {
            'DocID': self.doc_id,
            'text': self.text,
            'sentences': [s.to_json() for s in self.sentences],
            'relations': [r.to_json(self.doc_id, rel_id=r_i) for r_i, r in enumerate(self.relations)]
        }


DepRel = namedtuple('DepRel', ['rel', 'head', 'dep'])


class Sentence:
    def __init__(self, tokens, dependencies=None, parsetree=None):
        self.tokens: List[Token] = tokens
        self.parsetree: str = parsetree or ""
        self.dependencies: List[DepRel] = dependencies or []
        self.__parsetree = None

    def get_text(self):
        return ''.join([self.tokens[0].surface] +
                       [('' if self.tokens[t_i].offset_end == t.offset_begin else ' ') + t.surface
                        for t_i, t in enumerate(self.tokens[1:])])

    def get_ptree(self):
        if not self.__parsetree:
            try:
                ptree = nltk.Tree.fromstring(self.parsetree.strip())
                ptree = nltk.ParentedTree.convert(ptree)
                if not ptree.label().strip():
                    ptree = list(ptree)[0]
            except:
                ptree = None
            self.__parsetree = ptree
        return self.__parsetree

    def to_json(self):
        return {
            'dependencies': [(
                d.rel,
                f"{d.head.surface if d.head else 'ROOT'}-{d.head.local_idx if d.head else -1 + 1}",
                f"{d.dep.surface}-{d.dep.local_idx + 1}"
            ) for d in self.dependencies],
            'parsetree': self.parsetree,
            'words': [t.to_json() for t in self.tokens]
        }


class Token:

    def __init__(self, idx, sent_idx, local_idx, offset_begin, offset_end, surface, tag=""):
        self.surface: str = surface
        self.tag: str = tag
        # global word index
        self.idx: int = idx
        # sentence index
        self.sent_idx: int = sent_idx
        # local word index regarding sentence
        self.local_idx: int = local_idx
        self.offset_begin: int = offset_begin
        self.offset_end: int = offset_end

    def __str__(self):
        return f"{self.idx}-{self.surface}:{self.tag}"

    __repr__ = __str__

    def to_json(self):
        return (self.surface,
                {'CharacterOffsetBegin': self.offset_begin,
                 'CharacterOffsetEnd': self.offset_end,
                 'Linkers': [],
                 'PartOfSpeech': self.tag})

    def to_json_indices(self):
        return self.offset_begin, self.offset_end, self.idx, self.sent_idx, self.local_idx


class TokenSpan:

    def __init__(self, tokens):
        self.tokens: List[Token] = tokens

    def get_character_spans(self):
        spans = []
        if not self.tokens:
            return []
        span_begin = self.tokens[0].offset_begin
        span_end = self.tokens[0].offset_end
        cur_tok_idx = self.tokens[0].idx
        for t in self.tokens[1:]:
            if t.idx != cur_tok_idx + 1:
                spans.append((span_begin, span_end))
                span_begin = t.offset_begin
            span_end = t.offset_end
            cur_tok_idx = t.idx
        spans.append((span_begin, span_end))
        return spans


class Relation:

    def __init__(self, arg1, arg2, conn, senses, type):
        self.conn: TokenSpan = TokenSpan(conn)
        self.arg1: TokenSpan = TokenSpan(arg1)
        self.arg2: TokenSpan = TokenSpan(arg2)
        self.senses: List[str] = senses
        self.type: str = type

    def __str__(self):
        return "Relation({} {} arg1:<{}> arg2:<{}> conn:<{}>)".format(
            self.type, self.senses,
            ",".join([str(t.idx) for t in self.arg1.tokens]),
            ",".join([str(t.idx) for t in self.arg2.tokens]),
            ",".join([str(t.idx) for t in self.conn.tokens])
        )

    __repr__ = __str__

    def to_json(self, doc_id, rel_id):
        return {
            'Arg1': {'CharacterSpanList': self.arg1.get_character_spans(),
                     'RawText': ' '.join(t.surface for t in self.arg1.tokens),
                     'TokenList': [t.to_json_indices() for t in self.arg1.tokens]},
            'Arg2': {'CharacterSpanList': self.arg2.get_character_spans(),
                     'RawText': ' '.join(t.surface for t in self.arg2.tokens),
                     'TokenList': [t.to_json_indices() for t in self.arg2.tokens]},
            'Connective': {'CharacterSpanList': self.conn.get_character_spans(),
                           'RawText': ' '.join(t.surface for t in self.conn.tokens),
                           'TokenList': [t.to_json_indices() for t in self.conn.tokens]},
            'DocID': doc_id,
            'ID': rel_id,
            'Sense': self.senses,
            'Type': self.type
        }


def get_character_span_lists(tokens: List[Token]):
    spans = []
    if not tokens:
        return []
    span_begin = tokens[0].offset_begin
    span_end = tokens[0].offset_end
    cur_tok_idx = tokens[0].idx
    for t in tokens[1:]:
        if t.idx != cur_tok_idx + 1:
            spans.append((span_begin, span_end))
            span_begin = t.offset_begin
        span_end = t.offset_end
        cur_tok_idx = t.idx
    spans.append((span_begin, span_end))
    return spans
