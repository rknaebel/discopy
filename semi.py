import copy
import logging
import multiprocessing as mp
import os
import pickle
import ujson as json

import argparse
import numpy as np

import discopy.evaluate.exact
from discopy.parser import DiscourseParser
from discopy.utils import load_relations, convert_to_conll

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dir", help="",
                             default='tmp')
argument_parser.add_argument("--pdtb", help="",
                             default='/data/discourse/conll2016/')
argument_parser.add_argument("--parses", help="",
                             default='/data/discourse/conll2016/')
argument_parser.add_argument("--corpus", help="",
                             default='ted')
argument_parser.add_argument("--threshold", help="",
                             default=0.9, type=float)
args = argument_parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

logger = logging.getLogger('discopy')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.dir, 'semi.log'), mode='a')
# create file handler which logs even debug messages
fh.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def extract_discourse_relations(parser_path, parses):
    with mp.Pool(mp.cpu_count()) as pool:
        preds = pool.starmap(extract_discourse_relation,
                             [(doc_id, doc, parser_path) for doc_id, doc in parses.items()], chunksize=5)
    return {doc['DocID']: doc for doc in preds}


def extract_discourse_relation(doc_id, doc, parser_path):
    parser = pickle.load(open(parser_path, "rb"))
    parsed_relations = parser.parse_doc(doc)
    for p in parsed_relations:
        p['DocID'] = doc_id
    pred_docs = {
        'DocID': doc_id,
        'Relations': parsed_relations,
        'Confidence': np.mean([r['Confidence'] for r in parsed_relations]) if parsed_relations else 0
    }
    return pred_docs


def evaluate_parser(pdtb_gold, pdtb_pred):
    gold_relations = discopy.utils.load_relations(pdtb_gold)
    pred_relations = discopy.utils.load_relations([r for doc in pdtb_pred.values() for r in doc['Relations']])
    discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, threshold=0.7)


def get_additional_data(parser, additional_parses, parses_train, pdtb_train, confidence_threshold=0.70):
    preds = extract_discourse_relations(parser, additional_parses)
    confident_documents = [doc_id for doc_id, doc in preds.items()
                           if doc['Confidence'] > confidence_threshold]
    logger.info("Found confident documents: {}".format(len(confident_documents)))
    parses_semi_train = copy.copy(parses_train)
    pdtb_semi_train = copy.copy(pdtb_train)
    for doc_id in confident_documents:
        parses_semi_train[doc_id] = additional_parses[doc_id]
        pdtb_semi_train.extend(preds[doc_id]['Relations'])

    return parses_semi_train, pdtb_semi_train


def main():
    pdtb_train = [json.loads(s) for s in open(os.path.join(args.pdtb, 'en.train/relations.json'), 'r')]
    parses_train = json.loads(open(os.path.join(args.pdtb, 'en.train/parses.json'), 'r').read())

    # pdtb_dev = [json.loads(s) for s in open(os.path.join(args.pdtb, 'en.dev/relations.json'), 'r')]
    # parses_dev = json.loads(open(os.path.join(args.pdtb, 'en.dev/parses.json'), 'r').read())

    pdtb_test = [json.loads(s) for s in open(os.path.join(args.pdtb, 'en.test/relations.json'), 'r')]
    parses_test = json.loads(open(os.path.join(args.pdtb, 'en.test/parses.json'), 'r').read())

    logger.info('init parser...')
    parser = discopy.parser.DiscourseParser()
    parser_path = os.path.join(args.dir, '0', "parser.pkl")

    if os.path.exists(args.dir) and os.path.exists(parser_path):
        logger.info('load pre-trained parser...')
        parser.load(os.path.join(args.dir, "0"))
    else:
        logger.info('train parser...')
        parser.train(pdtb_train, parses_train)
        parser.save(os.path.join(args.dir, "0"))
        pickle.dump(parser, open(parser_path, 'wb'))
    logger.info('extract discourse relations from test data')
    pdtb_pred = extract_discourse_relations(parser_path, parses_test)
    evaluate_parser(pdtb_test, pdtb_pred)

    logger.info('load additional data...')
    parses_semi = {}
    for s in open(args.corpus, 'r'):
        doc = convert_to_conll(json.loads(s))
        doc['DocID'] = doc['DocID'].strip()
        parses_semi[doc['DocID']] = doc
    logger.info("loaded documents: {}".format(len(parses_semi)))

    for iteration in range(10):
        logger.info("current Iteration: {}".format(iteration))
        logger.info('get additional data')
        parses_semi_train, pdtb_semi_train = get_additional_data(parser_path, parses_semi, parses_train, pdtb_train,
                                                                 confidence_threshold=args.threshold)
        logger.info('re-train model')
        parser.train(pdtb_semi_train, parses_semi_train)
        parser_path = os.path.join(args.dir, str(iteration + 1), "parser.pkl")
        parser.save(os.path.join(args.dir, str(iteration + 1)))
        pickle.dump(parser, open(parser_path, 'wb'))
        pdtb_pred = extract_discourse_relations(parser_path, parses_test)
        evaluate_parser(pdtb_test, pdtb_pred)


if __name__ == '__main__':
    main()
