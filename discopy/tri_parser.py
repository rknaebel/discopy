import logging
import os

import joblib

from discopy.parser import DiscourseParser
from discopy.utils import bootstrap_dataset

logger = logging.getLogger('discopy')


class TriDiscourseParser(object):

    def __init__(self):
        self.models = [DiscourseParser() for _ in range(3)]
        self.training_data = []

    def train(self, pdtb, parses, bs_ratio=0.75):
        self.training_data = bootstrap_dataset(pdtb, parses, n_straps=3, ratio=bs_ratio)
        for p_i, p in enumerate(self.models):
            p.train(*self.training_data[p_i])

    def train_more(self, pdtbs, parsed_docs):
        for p_i, p in enumerate(self.models):
            strap_pdtb, strap_parses = self.training_data[p_i]
            for doc_id, rels in pdtbs[p_i].items():
                strap_pdtb.extend(rels)
                strap_parses[doc_id] = parsed_docs[doc_id]
            p.train(strap_pdtb, strap_parses)

    def score(self, pdtb, parses):
        for p_i, p in enumerate(self.models):
            logger.info('Evaluation Parser {}'.format(p_i))
            p.score(pdtb, parses)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for p_i, p in enumerate(self.models):
            os.makedirs(os.path.join(path, str(p_i)), exist_ok=True)
            joblib.dump(p, os.path.join(path, str(p_i), 'parser.joblib'))
        joblib.dump(self.training_data, os.path.join(path, "training.data"))

    @staticmethod
    def from_path(path):
        parser = TriDiscourseParser()
        for p_i, p in enumerate(parser.models):
            os.makedirs(os.path.join(path, str(p_i)), exist_ok=True)
            parser.models[p_i] = joblib.load(os.path.join(path, str(p_i), 'parser.joblib'))
        parser.training_data = joblib.load(os.path.join(path, "training.data"))
        return parser

    def parse_documents(self, documents):
        relations = {}
        for p_i, p in enumerate(self.models):
            relations[p_i] = p.parse_documents(documents)

        return relations
