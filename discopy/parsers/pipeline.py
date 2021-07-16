import json
import os
from typing import List

from tqdm import tqdm

from discopy.components.component import Component
from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation


class ParserPipeline:

    def __init__(self, components: List[Component]):
        self.components: List[Component] = []
        for c in components:
            if isinstance(c, Component):
                self.components.append(c)
            else:
                raise TypeError('Components should consist of Component instances only.')

    def __call__(self, doc: Document):
        return self.parse_doc(doc)

    def parse_doc(self, doc: Document):
        relations: List[Relation] = []
        for c in self.components:
            relations = c.parse(doc, relations)
        return doc.with_relations(relations)

    def parse(self, docs: List[Document]):
        return [self(doc) for doc in tqdm(docs)]

    def fit(self, docs_train: List[Document],
            docs_val: List[Document] = None):
        for c in self.components:
            print("train component:", c.model_name)
            c.fit(docs_train, docs_val)

    def score(self, docs: List[Document]):
        for c in self.components:
            c.score(docs)

    def save(self, path):
        configs = []
        for c in self.components:
            c.save(path)
            configs.append(c.get_config())
        json.dump(configs, open(os.path.join(path, 'config.json'), 'w'), indent=2)

    @staticmethod
    def from_config(path):
        from discopy.parsers.utils import component_register
        configs = json.load(open(os.path.join(path, 'config.json'), 'r'))
        components = []
        for config in configs:
            model_name = config['model_name']
            components.append(component_register[model_name].from_config(config))
        return ParserPipeline(components)

    def load(self, path):
        for c in self.components:
            c.load(path)
