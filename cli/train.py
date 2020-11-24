import os

import click
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')

from discopy.data.conll16 import get_conll_dataset
from discopy.parsers import get_parser
import discopy.evaluate.exact
from discopy.utils import init_logger

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
def main(parser, model_path, conll_path):
    parses_train, pdtb_train = get_conll_dataset(conll_path, 'en.train', load_trees=True, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(conll_path, 'en.dev', load_trees=True, connective_mapping=True)
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Train end-to-end Parser...')
    parser.fit(pdtb_train, parses_train, pdtb_val, parses_val)
    parser.save(os.path.join(model_path))


if __name__ == '__main__':
    main()
