import argparse
import os

import ujson as json
from tqdm import tqdm

from discopy.parser import DiscourseParser
from utils import convert_to_conll

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
                             default=10)
argument_parser.add_argument("--out", help="",
                             default='output.json')
args = argument_parser.parse_args()


def main():
    parser = DiscourseParser()

    if args.mode == 'train':
        pdtb = [json.loads(s) for s in open(args.pdtb, 'r').readlines()]
        parses = json.loads(open(args.parses).read())
        parser.train(pdtb, parses, epochs=args.epochs)
        parser.save(args.dir)
    elif args.mode == 'run':
        parser.load(args.dir)

        relations = []
        with open(args.parses, 'r') as fh_in, open(args.out, 'w') as fh_out:
            for idx, doc_line in enumerate(tqdm(fh_in)):
                doc = json.loads(doc_line)
                parsed_relations = parser.parse_doc(convert_to_conll(doc))
                for p in parsed_relations:
                    p['DocID'] = doc['DocID']
                relations.extend(parsed_relations)

                for relation in parsed_relations:
                    fh_out.write('{}\n'.format(json.dumps(relation)))

                if idx > 10:
                    break


if __name__ == '__main__':
    main()
