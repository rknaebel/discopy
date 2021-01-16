import json
from typing import List
import numpy as np
from transformers import AutoTokenizer, TFAutoModel

from discopy.data.relation import Relation
from discopy.data.sentence import ParsedSentence, BertSentence


class ParsedDocument:
    def __init__(self, doc_id, sentences, relations):
        self.doc_id = doc_id
        self.sentences: List[ParsedSentence] = sentences
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
        return ParsedDocument(doc_id=self.doc_id, sentences=self.sentences,
                              relations=[r for r in self.relations if r.is_explicit()])


class BertDocument:
    def __init__(self, doc_id, sentences: List[BertSentence], relations):
        self.doc_id = doc_id
        self.sentences: List[BertSentence] = sentences
        self.relations: List[Relation] = relations
        self.text = '\n'.join([s.get_text() for s in self.sentences])

    @classmethod
    def from_document(cls, doc: ParsedDocument, tokenizer=None, model=None, device='cpu'):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained('bert-base-cased')
        model = model or TFAutoModel.from_pretrained('bert-base-cased')
        sentences = [BertSentence(s.tokens, tokenizer, model, device) for s in doc.sentences]
        return BertDocument(doc.doc_id, sentences, doc.relations)

    def get_tokens(self):
        return [token for sent in self.sentences for token in sent.tokens]

    def get_embeddings(self) -> np.array:
        return np.concatenate([s.embeddings for s in self.sentences])

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
        return ParsedDocument(doc_id=self.doc_id, sentences=self.sentences,
                              relations=[r for r in self.relations if r.is_explicit()])
