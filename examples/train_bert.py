import argparse
import os

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dir", help="",
                             default='tmp')
argument_parser.add_argument("--conll", help="",
                             default='')
argument_parser.add_argument("--parser", help="",
                             default='lin')
argument_parser.add_argument("--threshold", help="",
                             default=0.9, type=float)
argument_parser.add_argument("--gpu", help="",
                             default='0')
args = argument_parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from discopy.data.conll16 import get_conll_bert_dataset
from discopy.parsers.bilstm_args import NeuralConnectiveArgumentExtractor, NeuralExplicitArgumentExtractor

from tqdm import tqdm

import discopy.evaluate.exact
from discopy.utils import init_logger

os.makedirs(args.dir, exist_ok=True)

logger = init_logger()


def extract_discourse_relations(parser, parses):
    preds = []
    for doc_id, doc in tqdm(parses.items()):
        preds.append(extract_discourse_relation(doc_id, doc, parser))
    return {doc['DocID']: doc for doc in preds}


def extract_discourse_relation(doc_id, doc, parser):
    parsed_relations = parser.parse_doc(doc)
    for p in parsed_relations:
        p['DocID'] = doc_id
    pred_docs = {
        'DocID': doc_id,
        'Relations': parsed_relations,
    }
    return pred_docs


def evaluate_parser(pdtb_gold, pdtb_pred, threshold=0.7):
    gold_relations = discopy.utils.load_relations(pdtb_gold)
    pred_relations = discopy.utils.load_relations([r for doc in pdtb_pred.values() for r in doc['Relations']])
    return discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, threshold=threshold)


if __name__ == '__main__':
    logger.info('Init Parser...')
    if args.parser == 'nca':
        parser = NeuralConnectiveArgumentExtractor(hidden_size=128, rnn_size=128, use_bert=True)
        load_trees = True
    elif args.parser == 'nea':
        parser = NeuralExplicitArgumentExtractor(hidden_size=128, rnn_size=128, use_bert=True)
        load_trees = False
    else:
        raise ValueError('Parser not found')

    logger.info('Load Data')
    parses_train, pdtb_train = get_conll_bert_dataset(args.conll, 'en.train', load_trees=load_trees,
                                                      connective_mapping=True)
    parses_val, pdtb_val = get_conll_bert_dataset(args.conll, 'en.dev', load_trees=load_trees, connective_mapping=True)
    parses_test, pdtb_test = get_conll_bert_dataset(args.conll, 'en.test', load_trees=load_trees,
                                                    connective_mapping=True)
    # parses_blind, pdtb_blind = get_conll_bert_dataset(args.conll, 'en.blind-test', load_trees=load_trees,
    #                                                   connective_mapping=True)

    logger.info('Train end-to-end Parser...')
    parser.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=2)
    parser.save(os.path.join(args.dir))

    logger.info('extract discourse relations from test data')
    pdtb_pred = extract_discourse_relations(parser, parses_test)
    all_results = evaluate_parser(pdtb_test, pdtb_pred, threshold=args.threshold)
