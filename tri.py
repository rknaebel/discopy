import copy
import os
import ujson as json

import joblib
import numpy as np
from discopy.parser import DiscourseParser
from discopy.semi_utils import args
from discopy.semi_utils import extract_discourse_relations, evaluate_parser, load_corpus
from discopy.utils import init_logger

os.makedirs(args.dir, exist_ok=True)

logger = init_logger(os.path.join(args.dir, 'tri.log'))


def bootstrap_dataset(pdtb, parses, n_straps=3, ratio=0.7, replace=True):
    n_samples = int(len(parses) * ratio)
    doc_ids = list(parses.keys())
    straps = []
    for i in range(n_straps):
        strap_doc_ids = set(np.random.choice(doc_ids, size=n_samples, replace=replace))
        strap_pdtb = [r for r in pdtb if r['DocID'] in strap_doc_ids]
        strap_parses = {doc_id: doc for doc_id, doc in parses.items() if doc_id in strap_doc_ids}
        straps.append((strap_pdtb, strap_parses))
    return straps


class TriModel:
    def __init__(self):
        self.models = [DiscourseParser() for _ in range(3)]

    def train(self, pdtb, parses, bs_ratio=0.75):
        straps_train = bootstrap_dataset(pdtb, parses, ratio=bs_ratio)
        for p_i, p in enumerate(self.models):
            p.train(*straps_train[p_i])

    def save(self, path, iteration=0):
        for p_i, p in enumerate(self.models):
            p.save(os.path.join(path, str(iteration), str(p_i)))

    def score(self, pdtb, parses, iteration=0):
        for p_i, p in enumerate(self.models):
            p_path = os.path.join(args.dir, str(iteration), str(p_i), "parser.pkl")
            pred = extract_discourse_relations(p_path, parses)
            evaluate_parser(pdtb, pred)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(self, os.path.join(path, 'parser.joblib'))

    @staticmethod
    def from_path(path):
        return joblib.load(os.path.join(path, 'parser.joblib'))


def get_additional_data_tri(parser, additional_parses, parses_train, pdtb_train, confidence_threshold=0.70):
    preds = extract_discourse_relations(parser, additional_parses)
    confident_documents = [doc_id for doc_id, doc in preds.items()
                           if doc['Confidence'] > confidence_threshold]
    logger.info("Average Confidence: {}".format(np.mean([doc['Confidence'] for doc in preds.values()])))
    logger.info("Maximum Confidence: {}".format(np.max([doc['Confidence'] for doc in preds.values()])))
    logger.info("Found confident documents: {}".format(len(confident_documents)))
    if len(confident_documents) == 0:
        logger.info("No confident documents: took five best")
        confident_documents = [doc_id for doc_id, _ in
                               sorted(preds.items(), key=(lambda doc_id, doc: doc['Confidence']), reverse=True)][:5]

    parses_semi_train = copy.copy(parses_train)
    pdtb_semi_train = copy.copy(pdtb_train)
    for doc_id in confident_documents:
        parses_semi_train[doc_id] = additional_parses[doc_id]
        pdtb_semi_train.extend(preds[doc_id]['Relations'])

    return parses_semi_train, pdtb_semi_train


if __name__ == '__main__':
    pdtb_train = [json.loads(s) for s in open(os.path.join(args.pdtb, 'en.train/relations.json'), 'r')]
    parses_train = json.loads(open(os.path.join(args.pdtb, 'en.train/parses.json'), 'r').read())

    pdtb_test = [json.loads(s) for s in open(os.path.join(args.pdtb, 'en.test/relations.json'), 'r')]
    parses_test = json.loads(open(os.path.join(args.pdtb, 'en.test/parses.json'), 'r').read())

    logger.info('=' * 50)
    logger.info('=' * 50)
    logger.info('== init parser...')
    parsers = TriModel()

    logger.info('== train parsers...')
    parsers.train(pdtb_train, parses_train, bs_ratio=0.75)
    parsers.save(args.dir, iteration=0)

    logger.info('== extract discourse relations from test data')
    parsers.score(pdtb_test, parses_test, 0)

    logger.info('load additional data...')
    parses_semi = load_corpus(args.corpus)
    logger.info("loaded documents: {}".format(len(parses_semi)))

    for iteration in range(args.iters):
        logger.info("current Iteration: {}".format(iteration))
        logger.info('get additional data')
        parses_semi_train, pdtb_semi_train = get_additional_data_tri(parser_path, parses_semi, parses_train, pdtb_train,
                                                                     confidence_threshold=args.threshold)
        logger.info('re-train model')
        parsers.train(pdtb_semi_train, parses_semi_train)
        parser_path = os.path.join(args.dir, str(iteration + 1))
        parsers.save(args.dir, iteration=iteration + 1)
        pdtb_pred = extract_discourse_relations(parser_path, parses_test)
        evaluate_parser(pdtb_test, pdtb_pred)
