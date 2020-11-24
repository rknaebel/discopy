import os

import click
from tqdm import tqdm

from discopy.parsers import get_parser

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')

from discopy.data.conll16 import get_conll_dataset
from discopy.utils import init_logger
import discopy.evaluate.exact

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


@click.command()
@click.argument('parser', type=str)
@click.argument('model-path', type=str)
@click.argument('conll-path', type=str)
@click.option('-t', '--threshold', default=0.9, type=str)
def main(parser, model_path, conll_path, threshold):
    parses_test, pdtb_test = get_conll_dataset(conll_path, 'en.test', load_trees=True, connective_mapping=True)
    parses_blind, pdtb_blind = get_conll_dataset(conll_path, 'en.blind-test', load_trees=True, connective_mapping=True)
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Load pre-trained Parser...')
    parser.load(model_path)
    logger.info('component evaluation (test)')
    parser.score(pdtb_test, parses_test)
    logger.info('extract discourse relations from test data')
    pdtb_pred = extract_discourse_relations(parser, parses_test)
    evaluate_parser(pdtb_test, pdtb_pred, threshold=threshold)
    logger.info('extract discourse relations from BLIND data')
    pdtb_pred = extract_discourse_relations(parser, parses_blind)
    evaluate_parser(pdtb_blind, pdtb_pred, threshold=threshold)


if __name__ == '__main__':
    main()
