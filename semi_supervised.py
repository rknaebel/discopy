import argparse
import os

import ujson as json
from tqdm import tqdm

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
                             default=10)
argument_parser.add_argument("--out", help="",
                             default='output.json')
args = argument_parser.parse_args()


def convert_to_conll(document):
    p = {
        'DocID': document['DocID'],
        'sentences': []
    }
    for sentence in document['sentences']:
        if not sentence:
            continue
        p['sentences'].append({
            'words': [(word, {
                'PartOfSpeech': pos,
            }) for word, pos in zip(sentence['Tokens'], sentence['POS'])],
            'parsetree': sentence['Parse'],
            'dependencies': [(dep, "{}-{}".format(*node1), "{}-{}".format(*node2)) for (dep, node1, node2) in
                             sentence['Dep']]
        })
    return p


def main():
    parser = DiscourseParser()

    if args.mode == 'train':
        parser.train(args.pdtb, args.parses, epochs=args.epochs)
        if not os.path.exists(args.dir):
            os.mkdir(args.dir)
        parser.save(args.dir)
    elif args.mode == 'run':
        parser.load(args.dir)

        documents = json.load(open(args.parses, 'r'))

        relations = []
        with open(args.out, 'w') as fh:

            for idx, doc in enumerate(tqdm(documents)):
                parsed_relations = parser.parse_doc(convert_to_conll(doc))
                for p in parsed_relations:
                    p['DocID'] = doc['DocID']
                relations.extend(parsed_relations)

                for relation in parsed_relations:
                    fh.write('{}\n'.format(json.dumps(relation)))

                if idx > 10:
                    break


if __name__ == '__main__':
    main()
