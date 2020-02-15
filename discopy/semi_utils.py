import argparse
import logging
import multiprocessing as mp
import ujson as json
from collections import Counter

import numpy as np

import discopy.evaluate.exact
import discopy.parsers.lin
import discopy.utils

logger = logging.getLogger('discopy.semi-utils')


def get_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--mode", help="",
                                 default='self')
    argument_parser.add_argument("--gpu", help="",
                                 default='0')
    argument_parser.add_argument("--dir", help="",
                                 default='tmp')
    argument_parser.add_argument("--out", help="",
                                 default='')
    argument_parser.add_argument("--fin", help="",
                                 default='')
    argument_parser.add_argument("--parser", help="",
                                 default='lin')
    argument_parser.add_argument("--train", help="",
                                 action='store_true')
    argument_parser.add_argument("--no-crf", help="",
                                 action='store_true')
    argument_parser.add_argument("--base-dir", help="",
                                 default='')
    argument_parser.add_argument("--pdtb", help="",
                                 default='/data/discourse/conll2016/')
    argument_parser.add_argument("--conll", help="",
                                 default='/data/discourse/conll2016/')
    argument_parser.add_argument("--parses", help="",
                                 default='/data/discourse/conll2016/')
    argument_parser.add_argument("--corpus", help="",
                                 default='')
    argument_parser.add_argument("--threshold", help="",
                                 default=0.9, type=float)
    argument_parser.add_argument("--iters", help="",
                                 default=10, type=int)
    argument_parser.add_argument("--samples", help="",
                                 default=100, type=int)
    argument_parser.add_argument("--estimators", help="",
                                 default=1, type=int)
    argument_parser.add_argument("--window-size", help="",
                                 default=150, type=int)
    argument_parser.add_argument("--skip-eval", help="",
                                 action='store_true')
    return argument_parser.parse_args()


def extract_discourse_relations(parser_path, parses):
    with mp.Pool(mp.cpu_count() // 2) as pool:
        preds = pool.starmap(extract_discourse_relation,
                             [(doc_id, doc, parser_path) for doc_id, doc in parses.items()], chunksize=3)
    return {doc['DocID']: doc for doc in preds}


def extract_discourse_relation(doc_id, doc, parser_path):
    parser = discopy.parsers.lin.LinParser.from_path(parser_path)
    parsed_relations = parser.parse_doc(doc)
    for p in parsed_relations:
        p['DocID'] = doc_id
    pred_docs = {
        'DocID': doc_id,
        'Relations': parsed_relations,
        'Confidence': np.mean([r['Confidence'] for r in parsed_relations]) if parsed_relations else 0
    }
    return pred_docs


def evaluate_parser(pdtb_gold, pdtb_pred, threshold=0.7):
    gold_relations = discopy.utils.load_relations(pdtb_gold)
    pred_relations = discopy.utils.load_relations([r for doc in pdtb_pred.values() for r in doc['Relations']])
    discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, threshold=threshold)


def fn(s):
    doc = discopy.utils.convert_to_conll(json.loads(s))
    return doc['DocID'].strip(), doc


def load_corpus(path):
    f = open(path, 'r').readlines()
    with mp.Pool(mp.cpu_count()) as pool:
        parses_semi = dict(pool.map(fn, f))

    return parses_semi


def get_explicit_stats(relations):
    distances = []
    spans = []
    spans_a1 = []
    spans_a2 = []
    for r in relations:
        arg1 = [i[2] for i in r['Arg1']['TokenList']]
        arg2 = [i[2] for i in r['Arg2']['TokenList']]
        spans_a1.append(max(arg1) - min(arg1))
        spans_a2.append(max(arg2) - min(arg2))
        spans.append(max(arg1 + arg2) - min(arg1 + arg2))
        if max(arg1) < min(arg2):
            distances.append('P')
        elif max(arg2) < min(arg1):
            distances.append('N')
        elif min(arg1) < min(arg2) < max(arg1):
            distances.append('A1Surround')
        elif min(arg2) < min(arg1) < max(arg2):
            distances.append('A2Surround')
        else:
            distances.append('Other')
    return distances, spans, spans_a1, spans_a2


def eval_parser(mode, parser, parses, relations):
    logger.info('EVAL component ({})'.format(mode))
    parser.score(relations, parses)
    logger.info('EXTRACT discourse relations from {} data'.format(mode))
    pred = parser.parse_documents(parses)
    logger.info('DISTRIBUTION info for {}'.format(mode))
    logger.info(get_relation_distances([r for doc in pred.values() for r in doc['Relations']]))
    evaluate_parser_explicit(relations, pred, 0.7)
    evaluate_parser_explicit(relations, pred, 0.8)
    evaluate_parser_explicit(relations, pred, 0.9)


def get_relation_distances(relations):
    explicits = [r for r in relations if r['Type'] == 'Explicit']
    distances, _, _, _ = get_explicit_stats(explicits)
    return Counter(distances)


def evaluate_parser_explicit(pdtb_gold, pdtb_pred, threshold=0.7):
    gold_relations = discopy.utils.load_relations(pdtb_gold)
    pred_relations = discopy.utils.load_relations([r for doc in pdtb_pred.values() for r in doc['Relations']])
    return discopy.evaluate.exact.evaluate_explicit_arguments(gold_relations, pred_relations, threshold=threshold)


def combine_data(parse_sets, relation_sets):
    parses = {doc_id: doc for parses in parse_sets for doc_id, doc in parses.items()}
    relations = [r for relations in relation_sets for r in relations]
    return parses, relations
