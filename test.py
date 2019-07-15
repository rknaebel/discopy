import os
from discopy.semi_utils import get_arguments
args = get_arguments()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import ujson as json

import discopy.evaluate.exact
from discopy.parsers.lin import LinParser
from discopy.parsers.gosh import GoshParser
from discopy.parsers.bilstm import BiLSTMDiscourseParser1, BiLSTMDiscourseParser2, BiLSTMDiscourseParser3
from discopy.utils import init_logger


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
    return discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, threshold=threshold)


parsers = {
    'lin': LinParser(),
    'gosh': GoshParser(),
    'bilstm1': BiLSTMDiscourseParser1(),
    'bilstm2': BiLSTMDiscourseParser2(),
    'bilstm3': BiLSTMDiscourseParser3(),
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
    if args.out:
        with open(args.out, 'w') as fh:
            for doc_id, doc in pdtb_pred.items():
                for relation in doc['Relations']:
                    fh.write('{}\n'.format(json.dumps(relation)))

    all_results = evaluate_parser(pdtb_test, pdtb_pred, threshold=args.threshold)
