from typing import List

from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation


class Component:
    model_name = ""
    used_features = []

    def get_config(self):
        return {
            'model_name': self.model_name,
        }

    @staticmethod
    def from_config(config: dict):
        raise NotImplementedError()

    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs_train: List[Document],
            docs_val: List[Document] = None):
        raise NotImplementedError()

    def score(self, docs: List[Document]):
        raise NotImplementedError()

    def parse(self, doc: Document,
              relations: List[Relation] = None, **kwargs):
        raise NotImplementedError()

    def get_used_features(self) -> List[str]:
        return self.used_features


class SubComponent:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs: List[Document]):
        raise NotImplementedError()

    def score(self, docs: List[Document]):
        raise NotImplementedError()
