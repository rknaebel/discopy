import argparse
import os
import ujson as json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm

import discopy.evaluate.exact
from discopy.parsers.gosh import GoshParser
from discopy.parsers.lin import LinParser
# from discopy.parsers.bilstm import BiLSTMDiscourseParser1, BiLSTMDiscourseParser2, BiLSTMDiscourseParser3
from discopy.utils import init_logger

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dir", help="",
                             default='tmp')
argument_parser.add_argument("--conll", help="",
                             default='')
argument_parser.add_argument("--parser", help="",
                             default='lin')
argument_parser.add_argument("--threshold", help="",
                             default=0.9, type=float)
args = argument_parser.parse_args()

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


parsers = {
    'lin': LinParser(),
    'gosh': GoshParser(),
    # 'bilstm1': BiLSTMDiscourseParser1(),
    # 'bilstm2': BiLSTMDiscourseParser2(),
    # 'bilstm3': BiLSTMDiscourseParser3(),
}

if __name__ == '__main__':
    pdtb_train = [json.loads(s) for s in open(os.path.join(args.conll, 'en.train/relations.json'), 'r')]
    parses_train = json.loads(open(os.path.join(args.conll, 'en.train/parses.json'), 'r').read())

    pdtb_val = [json.loads(s) for s in open(os.path.join(args.conll, 'en.dev/relations.json'), 'r')]
    parses_val = json.loads(open(os.path.join(args.conll, 'en.dev/parses.json'), 'r').read())

    pdtb_test = [json.loads(s) for s in open(os.path.join(args.conll, 'en.test/relations.json'), 'r')]
    parses_test = json.loads(open(os.path.join(args.conll, 'en.test/parses.json'), 'r').read())

    logger.info('Init Parser...')
    parser = parsers.get(args.parser, LinParser)
    parser_path = args.dir

    logger.info('Train end-to-end Parser...')
    parser.fit(pdtb_train, parses_train, pdtb_val, parses_val)
    parser.save(os.path.join(args.dir))

    logger.info('extract discourse relations from test data')
    pdtb_pred = extract_discourse_relations(parser, parses_test)
    all_results = evaluate_parser(pdtb_test, pdtb_pred, threshold=args.threshold)
