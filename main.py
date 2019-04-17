import json
import os

import argparse
from collections import defaultdict

import discopy.evaluate.exact
from discopy.utils import Relation
from discopy.parser import DiscourseParser

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
args = argument_parser.parse_args()


def load_relations(path):
    relations_json = [json.loads(s) for s in open(path, 'r').readlines()]
    relations = defaultdict(list)
    for r in relations_json:
        conn = [(i[2] if type(i) == list else i) for i in r['Connective']['TokenList']]
        arg1 = [(i[2] if type(i) == list else i) for i in r['Arg1']['TokenList']]
        arg2 = [(i[2] if type(i) == list else i) for i in r['Arg2']['TokenList']]
        senses = r['Sense']
        relations[r['DocID']].append(Relation(arg1, arg2, conn, senses))
    return relations


def main():
    parser = DiscourseParser()

    if args.mode == 'train':
        pdtb = [json.loads(s) for s in open(args.pdtb, 'r').readlines()]
        parses = json.loads(open(args.parses).read())
        parser.train(pdtb, parses, epochs=args.epochs)
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)
        parser.save(args.dir)
    elif args.mode == 'run':
        parser.load(args.dir)
        relations = parser.parse_file(args.parses)
        with open(args.out, 'w') as fh:
            fh.writelines('\n'.join(['{}'.format(json.dumps(relation)) for relation in relations]))
    elif args.mode == 'eval':
        gold_relations = load_relations('../discourse/data/conll2016/en.dev/relations.json')
        pred_relations = load_relations('output.json')
        discopy.evaluate.exact.evaluate_all(gold_relations, pred_relations)
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()
