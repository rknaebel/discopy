from typing import List


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

    def __hash__(self):
        return hash(str(self))

    __repr__ = __str__

    def to_json(self):
        return (self.surface,
                {'CharacterOffsetBegin': self.offset_begin,
                 'CharacterOffsetEnd': self.offset_end,
                 'Linkers': [],
                 'PartOfSpeech': self.tag})

    def to_json_indices(self):
        return self.offset_begin, self.offset_end, self.idx, self.sent_idx, self.local_idx

    def __eq__(self, other: 'Token'):
        return all([
            self.idx == other.idx, self.surface == other.surface, self.tag == other.tag,
            self.sent_idx == other.sent_idx, self.local_idx == other.local_idx, self.offset_begin == other.offset_begin,
            self.offset_end == other.offset_end
        ])


class TokenSpan:

    def __init__(self, tokens):
        self.tokens: List[Token] = tokens

    def get_sentence_idxs(self):
        return sorted(set(t.sent_idx for t in self.tokens))

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

    def overlap(self, other: 'TokenSpan') -> int:
        return sum(int(i == j) for i in self.tokens for j in other.tokens)

    def add(self, token: Token):
        self.tokens.append(token)

    def __or__(self, other):
        tokens = sorted(set(self.tokens) | set(other.tokens), key=lambda t: t.idx)
        # TODO consistency check!
        return TokenSpan(tokens)

    def __and__(self, other):
        tokens = sorted(set(self.tokens) & set(other.tokens), key=lambda t: t.idx)
        return TokenSpan(tokens)

    def __len__(self):
        return len(self.tokens)
