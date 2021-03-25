from typing import List

from discopy.data.token import TokenSpan
from discopy.utils import jaccard_distance


class Relation:

    def __init__(self, arg1=None, arg2=None, conn=None, senses=None, type=""):
        self.conn: TokenSpan = TokenSpan(conn or [])
        self.arg1: TokenSpan = TokenSpan(arg1 or [])
        self.arg2: TokenSpan = TokenSpan(arg2 or [])
        self.senses: List[str] = senses or []
        self.type: str = type

    def is_explicit(self):
        return self.type == "Explicit"

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

    def is_empty(self):
        return len(self.conn) == len(self.arg1) == len(self.arg2) == 0

    def distance(self, other):
        d_args = jaccard_distance(self.arg1 | self.arg2, other.arg1 | other.arg2)
        d_arg1 = jaccard_distance(self.arg1, other.arg1)
        d_conn = jaccard_distance(self.conn, other.conn)
        d_arg2 = jaccard_distance(self.arg2, other.arg2)
        return sum([d_args, d_arg1, d_arg2, d_conn]) / 4
