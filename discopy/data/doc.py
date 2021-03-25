import json
from typing import List

import numpy as np

from discopy.data.relation import Relation
from discopy.data.sentence import Sentence


class Document:
    def __init__(self, doc_id, sentences: List[Sentence], relations: List[Relation]):
        self.doc_id = doc_id
        self.meta = {}
        self.sentences: List[Sentence] = sentences
        self.relations: List[Relation] = relations
        self.text = '\n'.join([s.get_text() for s in self.sentences])

    def to_json(self):
        return {
            'docID': self.doc_id,
            'text': self.text,
            'sentences': [s.to_json() for s in self.sentences],
            'relations': [r.to_json(self.doc_id, rel_id=r_i) for r_i, r in enumerate(self.relations)]
        }

    def get_tokens(self):
        return [token for sent in self.sentences for token in sent.tokens]

    def get_embeddings(self) -> np.array:
        return np.concatenate([s.get_embeddings() for s in self.sentences], axis=0)

    def get_embedding_dim(self) -> int:
        return int(self.sentences[0].embeddings.shape[-1])

    def with_relations(self, relations):
        return Document(self.doc_id, self.sentences, relations)

    def __str__(self):
        return json.dumps(self.to_json(), indent=2)

    def get_explicit_relations(self):
        return [r for r in self.relations if r.is_explicit()]
