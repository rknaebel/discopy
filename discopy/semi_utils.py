import multiprocessing as mp
import ujson as json

import argparse
import numpy as np

import discopy.evaluate.exact
import discopy.parsers.lin
import discopy.utils


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
