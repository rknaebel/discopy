from typing import List, Union

from discopy.data.doc import ParsedDocument, BertDocument
from discopy.data.relation import Relation


class Component:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs_train: List[Union[ParsedDocument, BertDocument]],
            docs_val: List[Union[ParsedDocument, BertDocument]] = None):
        raise NotImplementedError()

    def score(self, docs: List[Union[ParsedDocument, BertDocument]]):
        raise NotImplementedError()

    def parse(self, doc: Union[ParsedDocument, BertDocument], relations: List[Relation] = None, **kwargs):
        raise NotImplementedError()


class SubComponent:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs: List[Union[ParsedDocument, BertDocument]]):
        raise NotImplementedError()

    def score(self, docs: List[Union[ParsedDocument, BertDocument]]):
        raise NotImplementedError()
