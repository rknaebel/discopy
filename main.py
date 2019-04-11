import json
import os

import argparse

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
argument_parser.add_argument("--out", help="",
                             default='output.json')
args = argument_parser.parse_args()


def main():
    parser = DiscourseParser()

    if args.mode == 'train':
        parser.train(args.pdtb, args.parses)
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)
        parser.save(args.dir)
    elif args.mode == 'run':
        parser.load(args.dir)
        relations = parser.parse_file(args.parses)
        with open(args.out, 'w') as fh:
            fh.write('[\n')
            fh.writelines(',\n'.join(['{}'.format(json.dumps(relation)) for relation in relations]))
            fh.write('\n]\n')


if __name__ == '__main__':
    main()
