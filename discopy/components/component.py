from typing import List, Union

import discopy.data.doc as ddoc
import discopy.data.relation as drelation


class Component:

    def __init__(self, used_features: List[str] = None):
        self.used_features = used_features or []

    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs_train: List[ddoc.Document],
            docs_val: List[ddoc.Document] = None):
        raise NotImplementedError()

    def score(self, docs: List[ddoc.Document]):
        raise NotImplementedError()

    def parse(self, doc: ddoc.Document,
              relations: List[drelation.Relation] = None, **kwargs):
        raise NotImplementedError()

    def get_used_features(self) -> List[str]:
        return self.used_features


class SubComponent:
    def load(self, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        raise NotImplementedError()

    def fit(self, docs: List[ddoc.Document]):
        raise NotImplementedError()

    def score(self, docs: List[ddoc.Document]):
        raise NotImplementedError()
