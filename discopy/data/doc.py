import json
from collections import namedtuple
from typing import List, Optional

import nltk

from discopy.data.relation import Relation
from discopy.data.token import Token


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

    def __str__(self):
        return json.dumps(self.to_json(), indent=2)

    def get_explicit_relations(self):
        return Document(doc_id=self.doc_id, sentences=self.sentences,
                        relations=[r for r in self.relations if r.is_explicit()])


DepRel = namedtuple('DepRel', ['rel', 'head', 'dep'])


class Sentence:
    def __init__(self, tokens, dependencies=None, parsetree=None):
        self.tokens: List[Token] = tokens
        self.parsetree: str = parsetree or ""
        self.dependencies: List[DepRel] = dependencies or []
        self.__parsetree = None

    def get_text(self) -> str:
        return ''.join([self.tokens[0].surface] +
                       [('' if self.tokens[t_i].offset_end == t.offset_begin else ' ') + t.surface
                        for t_i, t in enumerate(self.tokens[1:])])

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
