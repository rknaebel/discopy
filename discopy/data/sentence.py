from collections import namedtuple
from typing import List, Optional

import numpy as np
import nltk

from discopy.data.token import Token


class Sentence:
    def __init__(self, tokens):
        self.tokens: List[Token] = tokens

    def get_text(self) -> str:
        return ''.join([self.tokens[0].surface] +
                       [('' if self.tokens[t_i].offset_end == t.offset_begin else ' ') + t.surface
                        for t_i, t in enumerate(self.tokens[1:])])


class ParsedSentence(Sentence):
    def __init__(self, tokens, dependencies=None, parsetree=None):
        super().__init__(tokens)
        self.parsetree: str = parsetree or ""
        self.dependencies: List[DepRel] = dependencies or []
        self.__parsetree = None

    def get_ptree(self) -> Optional[nltk.ParentedTree]:
        if not self.__parsetree:
            try:
                ptree = nltk.Tree.fromstring(self.parsetree.strip())
                ptree = nltk.ParentedTree.convert(ptree)
                if not ptree.label().strip():
                    ptree = list(ptree)[0]
                if not ptree.leaves():
                    return None
            except:
                ptree = None
            self.__parsetree = ptree
        return self.__parsetree

    def to_json(self) -> dict:
        return {
            'dependencies': [(
                d.rel,
                f"{d.head.surface if d.head else 'ROOT'}-{d.head.local_idx if d.head else -1 + 1}",
                f"{d.dep.surface}-{d.dep.local_idx + 1}"
            ) for d in self.dependencies],
            'parsetree': self.parsetree,
            'words': [t.to_json() for t in self.tokens]
        }


class BertSentence(Sentence):
    def __init__(self, tokens, embeddings: np.array):
        super().__init__(tokens)
        self.embeddings: np.array = embeddings

    def get_embeddings(self):
        return self.embeddings

    def to_json(self) -> dict:
        return {
            'dependencies': [],
            'parsetree': '',
            'words': [t.to_json() for t in self.tokens]
        }


DepRel = namedtuple('DepRel', ['rel', 'head', 'dep'])
