import os
from typing import List, Union

import click
from tqdm import tqdm

from discopy.components.argument.base import ExplicitArgumentExtractor, ImplicitArgumentExtractor
from discopy.components.component import Component
from discopy.data.doc import Document
from discopy.data.relation import Relation
from discopy.evaluate.conll import print_results, evaluate_docs, evaluate_docs_average
from discopy.utils import init_logger


class ParserPipeline:

    def __init__(self, components: List[Component]):
        self.components: List[Component] = []
        for c in components:
            if isinstance(c, Component):
                self.components.append(c)
            else:
                raise TypeError('Components should consist of Component instances only.')

    def __call__(self, doc: Document):
        relations: List[Relation] = []
        for c in self.components:
            relations = c.parse(doc, relations)
        return doc.with_relations(relations)

    def parse(self, docs: List[Document]):
        return [self(doc) for doc in tqdm(docs)]

    def fit(self, docs_train: List[Document],
            docs_val: List[Document] = None):
        for c in self.components:
            print("train component:", c)
            c.fit(docs_train, docs_val)

    def score(self, docs: List[Document]):
        for c in self.components:
            c.score(docs)

    def save(self, path):
        for c in self.components:
            c.save(path)

    def load(self, path):
        for c in self.components:
            c.load(path)


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    # docs_train = load_conll_dataset(os.path.join(conll_path, 'en.train'), simple_connectives=True)
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'), simple_connectives=True)

    parser = ParserPipeline([
        ConnectiveClassifier(),
        ExplicitArgumentExtractor(),
        ExplicitSenseClassifier(),
        ImplicitArgumentExtractor(),
        NonExplicitSenseClassifier()
    ])
    # logger.info('Train model')
    # parser.fit(docs_train)
    # logger.info('Evaluation on TRAIN')
    # parser.score(docs_train)
    # logger.info('Evaluation on TEST')
    # parser.score(docs_val)
    # logger.info('Save Parser')
    # parser.save('models/lin-new')
    parser.load('models/lin-new')
    logger.info('Parse one document')
    docs = [d.get_explicit_relations() for d in docs_val]
    preds = [parser(d) for d in docs]
    print_results(evaluate_docs(docs, preds))
    print_results(evaluate_docs_average(docs, preds))


if __name__ == "__main__":
    main()
