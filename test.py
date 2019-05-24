import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import ujson as json

import discopy.evaluate.exact
from discopy.parser import DiscourseParser
from discopy.parser_bilstm import BiLSTMDiscourseParser1, BiLSTMDiscourseParser2
# from discopy.parser_bilstm import BiLSTMDiscourseParser
from discopy.semi_utils import get_arguments
from discopy.utils import init_logger

args = get_arguments()

os.makedirs(args.dir, exist_ok=True)

logger = init_logger(os.path.join(args.dir, 'self.log'))


def extract_discourse_relations(parser, parses):
    preds = [extract_discourse_relation(doc_id, doc, parser) for doc_id, doc in parses.items()]
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
    discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, threshold=threshold)


parsers = {
    'lin': DiscourseParser(),
    'bilstm1': BiLSTMDiscourseParser1(),
    'bilstm2': BiLSTMDiscourseParser2(),
}

if __name__ == '__main__':
    pdtb_train = [json.loads(s) for s in open(os.path.join(args.conll, 'en.train/relations.json'), 'r')]
    parses_train = json.loads(open(os.path.join(args.conll, 'en.train/parses.json'), 'r').read())

    pdtb_val = [json.loads(s) for s in open(os.path.join(args.conll, 'en.dev/relations.json'), 'r')]
    parses_val = json.loads(open(os.path.join(args.conll, 'en.dev/parses.json'), 'r').read())

    pdtb_test = [json.loads(s) for s in open(os.path.join(args.conll, 'en.test/relations.json'), 'r')]
    parses_test = json.loads(open(os.path.join(args.conll, 'en.test/parses.json'), 'r').read())

    logger.info('Init Parser...')
    parser = parsers.get(args.parser, DiscourseParser)
    parser_path = args.dir

    # if args.base_dir and os.path.exists(os.path.join(args.base_dir, 'parser.joblib')):
    #     logger.info('load base model from ' + args.base_dir)
    #     parser = discopy.parser.DiscourseParser.from_path(args.base_dir)
    #     parser_path = args.base_dir
    if args.train:
        logger.info('Train end-to-end Parser...')
        parser.train(pdtb_train, parses_train, pdtb_val, parses_val)
        parser.save(os.path.join(args.dir))
    elif os.path.exists(args.dir):
        logger.info('Load pre-trained Parser...')
        parser.load(args.dir, parses_train)
    else:
        raise ValueError('Training and Loading not clear')

    # logger.info('component evaluation (test)')
    # parser.score(pdtb_test, parses_test)

    logger.info('extract discourse relations from test data')
    pdtb_pred = extract_discourse_relations(parser, parses_test)
    evaluate_parser(pdtb_test, pdtb_pred, threshold=args.threshold)
