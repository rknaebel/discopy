import argparse
import os
import ujson as json

import discopy.evaluate.exact
from discopy.parsers.lin import LinParser
from discopy.utils import load_relations, init_logger

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--mode", help="",
                             default='parse')
argument_parser.add_argument("--dir", help="",
                             default='tmp')
argument_parser.add_argument("--pdtb", help="",
                             default='results')
argument_parser.add_argument("--parses", help="",
                             default='results')
argument_parser.add_argument("--epochs", help="",
                             default=10, type=int)
argument_parser.add_argument("--out", help="",
                             default='output.json')
argument_parser.add_argument("--eval-threshold", help="",
                             default=0.9, type=float)
args = argument_parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

logger = init_logger(os.path.join(args.dir, 'main.log'))


def main():
    if args.mode == 'train':
        parser = LinParser()
        logger.info('load PDTB data')
        pdtb = [json.loads(s) for s in open(args.pdtb, 'r').readlines()]
        parses = json.loads(open(args.parses).read())
        parser.train(pdtb, parses)
        parser.save(args.dir)
    elif args.mode == 'run':
        documents = json.loads(open(args.parses, mode='rb').read())
        parser = LinParser.from_path(args.dir)
        relations = parser.parse_documents(documents)
        with open(args.out, 'w') as fh:
            fh.writelines('\n'.join(['{}'.format(json.dumps(relation)) for relation in relations]))
    elif args.mode == 'eval':
        gold_relations = load_relations([json.loads(s) for s in open(args.pdtb, 'r').readlines()])
        pred_relations = load_relations([json.loads(s) for s in open(args.out, 'r').readlines()])
        discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, args.eval_threshold)
    elif args.mode == 'run-eval':
        logger.info('start run-eval')
        parser = LinParser.from_path(args.dir)
        logger.info('parse test documents')
        documents = json.loads(open(args.parses, mode='rb').read())
        relations = parser.parse_documents(documents)
        with open(args.out, 'w') as fh:
            fh.writelines('\n'.join(['{}'.format(json.dumps(relation)) for relation in relations]))
        logger.info('convert relations to conll format')
        gold_relations = load_relations([json.loads(s) for s in open(args.pdtb, 'r').readlines()])
        pred_relations = load_relations(relations)
        logger.info('evaluate on test data...')
        discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, args.eval_threshold)
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()
