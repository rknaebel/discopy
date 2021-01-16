from typing import List

from discopy.data.doc import Document
from discopy.data.relation import Relation


class Component:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        raise NotImplementedError()

    def score(self, docs: List[Document]):
        raise NotImplementedError()

    def parse(self, doc: Document, relations: List[Relation] = None, **kwargs):
        raise NotImplementedError()


class SubComponent:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs: List[Document]):
        raise NotImplementedError()

    def score(self, docs: List[Document]):
        raise NotImplementedError()
