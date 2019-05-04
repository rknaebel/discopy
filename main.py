import json

import argparse

import discopy.evaluate.exact
from discopy.parser import DiscourseParser
from discopy.utils import load_relations

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


def main():
    parser = DiscourseParser()

    if args.mode == 'train':
        pdtb = [json.loads(s) for s in open(args.pdtb, 'r').readlines()]
        parses = json.loads(open(args.parses).read())
        parser.train(pdtb, parses)
        parser.save(args.dir)
    elif args.mode == 'run':
        parser.load(args.dir)
        relations = parser.parse_file(args.parses)
        with open(args.out, 'w') as fh:
            fh.writelines('\n'.join(['{}'.format(json.dumps(relation)) for relation in relations]))
    elif args.mode == 'eval':
        gold_relations = load_relations([json.loads(s) for s in open(args.pdtb, 'r').readlines()])
        pred_relations = load_relations([json.loads(s) for s in open(args.out, 'r').readlines()])
        discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, args.eval_threshold)
    elif args.mode == 'run-eval':
        parser.load(args.dir)
        relations = parser.parse_file(args.parses)
        with open(args.out, 'w') as fh:
            fh.writelines('\n'.join(['{}'.format(json.dumps(relation)) for relation in relations]))
        gold_relations = load_relations([json.loads(s) for s in open(args.pdtb, 'r').readlines()])
        pred_relations = load_relations(relations)
        discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations, args.eval_threshold)
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()
