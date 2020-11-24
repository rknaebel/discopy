import os

from tqdm import tqdm

from discopy.parsers import get_parser
from discopy.semi_utils import get_arguments

args = get_arguments()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from discopy.data.conll16 import get_conll_dataset
from discopy.utils import init_logger
import discopy.evaluate.exact

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
    parses_train, pdtb_train = get_conll_dataset(args.conll, 'en.train', load_trees=True, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(args.conll, 'en.dev', load_trees=True, connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset(args.conll, 'en.test', load_trees=True, connective_mapping=True)
    parses_blind, pdtb_blind = get_conll_dataset(args.conll, 'en.blind-test', load_trees=True, connective_mapping=True)

    logger.info('Init Parser...')
    parser = get_parser(args.parser)
    parser_path = args.dir

    if args.train:
        logger.info('Train end-to-end Parser...')
        parser.fit(pdtb_train, parses_train, pdtb_val, parses_val)
        parser.save(os.path.join(args.dir))
    elif os.path.exists(args.dir):
        logger.info('Load pre-trained Parser...')
        parser.load(args.dir)
    else:
        raise ValueError('Training and Loading not clear')

    logger.info('component evaluation (test)')
    parser.score(pdtb_test, parses_test)

    logger.info('extract discourse relations from test data')
    pdtb_pred = extract_discourse_relations(parser, parses_test)
    evaluate_parser(pdtb_test, pdtb_pred, threshold=args.threshold)

    logger.info('extract discourse relations from BLIND data')
    pdtb_pred = extract_discourse_relations(parser, parses_blind)
    evaluate_parser(pdtb_blind, pdtb_pred, threshold=args.threshold)
