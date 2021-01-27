from typing import List, Union

from discopy.data.doc import ParsedDocument, BertDocument
from discopy.data.relation import Relation


class Component:

    def __init__(self, used_features: List[str] = None):
        self.used_features = used_features or []

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

    def get_used_features(self) -> List[str]:
        return self.used_features


class SubComponent:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs: List[Union[ParsedDocument, BertDocument]]):
        raise NotImplementedError()

    def score(self, docs: List[Union[ParsedDocument, BertDocument]]):
        raise NotImplementedError()
