import os
import ujson as json

import numpy as np

from discopy.semi_utils import extract_discourse_relations, evaluate_parser, load_corpus
from discopy.semi_utils import get_arguments
from discopy.tri_parser import TriDiscourseParser
from discopy.utils import init_logger

args = get_arguments()

os.makedirs(args.dir, exist_ok=True)

logger = init_logger(os.path.join(args.dir, 'tri.log'))


def get_additional_data(parser, parses_semi, n_samples=200, confidence_threshold=0.70):
    parses_bootstrap = bootstrap_parses(parses_semi, n_straps=1, n_samples=n_samples, replace=False)[0]
    semi_pred_0 = extract_discourse_relations(os.path.join(parser, str(0)), parses_bootstrap)
    semi_pred_1 = extract_discourse_relations(os.path.join(parser, str(1)), parses_bootstrap)
    semi_pred_2 = extract_discourse_relations(os.path.join(parser, str(2)), parses_bootstrap)

    additional_docs_p0 = {}
    additional_docs_p1 = {}
    additional_docs_p2 = {}

    for doc_id in parses_bootstrap.keys():
        rs0 = semi_pred_0[doc_id]['Relations']
        rs1 = semi_pred_1[doc_id]['Relations']
        rs2 = semi_pred_2[doc_id]['Relations']

        doc_results = []
        for r in rs0:
            doc_results.append(
                any(relation_is_equal(r, r1) for r1 in rs1) and any(relation_is_equal(r, r2) for r2 in rs2))
        if np.mean(doc_results) > confidence_threshold:
            additional_docs_p0[doc_id] = rs0

        doc_results = []
        for r in rs1:
            doc_results.append(
                any(relation_is_equal(r, r0) for r0 in rs0) and any(relation_is_equal(r, r2) for r2 in rs2))
        if np.mean(doc_results) > confidence_threshold:
            additional_docs_p1[doc_id] = rs1

        doc_results = []
        for r in rs2:
            doc_results.append(
                any(relation_is_equal(r, r0) for r0 in rs0) and any(relation_is_equal(r, r1) for r1 in rs1))
        if np.mean(doc_results) > confidence_threshold:
            additional_docs_p2[doc_id] = rs2

    return additional_docs_p0, additional_docs_p1, additional_docs_p2


def relation_is_equal(r1, r2):
    if r1['Type'] == r2['Type'] == 'Implicit':
        return all([r1['Arg1']['RawText'] == r2['Arg1']['RawText'],
                    r1['Arg2']['RawText'] == r2['Arg2']['RawText'],
                    r1['Sense'][0] == r2['Sense'][0]])
    elif r1['Type'] == r2['Type'] == 'Explicit':
        return all([r1['Arg1']['RawText'] == r2['Arg1']['RawText'],
                    r1['Arg2']['RawText'] == r2['Arg2']['RawText'],
                    r1['Connective']['RawText'] == r2['Connective']['RawText'],
                    r1['ArgPos'] == r2['ArgPos'],
                    r1['Sense'][0] == r2['Sense'][0]])
    else:
        return False


def bootstrap_parses(parses, n_straps=3, ratio=0.7, n_samples=None, replace=True):
    n_samples = n_samples or int(len(parses) * ratio)
    doc_ids = list(parses.keys())
    straps = []
    for i in range(n_straps):
        strap_doc_ids = set(np.random.choice(doc_ids, size=n_samples, replace=replace))
        strap_parses = {doc_id: doc for doc_id, doc in parses.items() if doc_id in strap_doc_ids}
        straps.append(strap_parses)
    return straps


if __name__ == '__main__':
    pdtb_train = [json.loads(s) for s in open(os.path.join(args.conll, 'en.train/relations.json'), 'r')]
    parses_train = json.loads(open(os.path.join(args.conll, 'en.train/parses.json'), 'r').read())

    pdtb_test = [json.loads(s) for s in open(os.path.join(args.conll, 'en.test/relations.json'), 'r')]
    parses_test = json.loads(open(os.path.join(args.conll, 'en.test/parses.json'), 'r').read())

    logger.info('=' * 50)
    logger.info('=' * 50)
    logger.info('== init parser...')
    parsers = TriDiscourseParser()
    parser_path = args.dir

    logger.info('== train parsers...')
    if args.base_dir and os.path.exists(args.base_dir):
        logger.info('load base model from ' + args.base_dir)
        parsers = TriDiscourseParser.from_path(args.base_dir)
    else:
        logger.info('train base model')
        parsers.train(pdtb_train, parses_train, bs_ratio=0.6)
        parsers.save(parser_path)

    logger.info('== extract discourse relations from test data')
    parsers.score(pdtb_test, parses_test)

    logger.info('load additional data...')
    parses_semi = load_corpus(args.corpus)
    logger.info("loaded documents: {}".format(len(parses_semi)))

    for iteration in range(args.iters):
        logger.info("current Iteration: {}".format(iteration))
        logger.info('get additional data')
        new_pdtbs = get_additional_data(parser_path, parses_semi, n_samples=args.samples,
                                        confidence_threshold=args.threshold)
        logger.info('re-train model')
        parsers.train_more(new_pdtbs, parses_semi)
        parser_path = os.path.join(args.dir, str(iteration + 1))
        parsers.save(os.path.join(args.dir, str(iteration + 1)))
        for i in range(3):
            logger.info("Evaluate Parser {}".format(i))
            pdtb_pred = extract_discourse_relations(os.path.join(parser_path, str(i)), parses_test)
            evaluate_parser(pdtb_test, pdtb_pred)
