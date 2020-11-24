import argparse
import os
from pprint import pprint

import nltk
from discopy.parsers import get_parser

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import spacy
from discopy.utils import init_logger
import supar

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dir", help="",
                             default='tmp')
argument_parser.add_argument("--out", help="",
                             default='')
argument_parser.add_argument("--src", help="",
                             default='')
args = argument_parser.parse_args()

logger = init_logger()


def parse_text(text):
    nlp = spacy.load('en')
    con_parser = supar.Parser.load('crf-con-en')
    sentences = []
    offset = 0
    text = [line.strip() for line in text.splitlines(keepends=True) if line.strip() and len(line.split(' ')) > 1]
    for doc in nlp.pipe(text):
        sents = list(doc.sents)
        for sent in sents:
            # words
            words = [
                [t.text, {
                    'CharacterOffsetBegin': t.idx + offset,
                    'CharacterOffsetEnd': t.idx + len(t.text) + offset,
                    'Linkers': [],
                    'PartOfSpeech': t.tag_,
                    'NamedEntity': t.ent_type_,
                }]
                for t in sent
            ]

            # dependencies
            d_map = {
                'compound': 'nn'
            }
            inputs = [(w[0], w[1]['PartOfSpeech']) for w in words]
            try:
                parses = con_parser.predict([inputs], prob=False, verbose=False)
                parsetree = nltk.ParentedTree.convert(parses.trees[0])
            except:
                parsetree = None
            sentences.append({
                'words': words,
                'parsetree': parsetree,
                'dependencies': [(
                    d_map.get(t.dep_, t.dep_),
                    "{}-{}".format(t.head.text, t.head.i + 1),
                    "{}-{}".format(t.text, t.i + 1))
                    for t in sent],
                'sentence': sent.string,
                'sentOffset': sent[0].idx
            })
            offset += len(sent.string)
    return {
        'text': text,
        'sentences': sentences
    }


def main():
    logger.info('Init Parser...')
    parser = get_parser('lin')

    logger.info('Load pre-trained Parser...')
    parser.load(args.dir)

    if args.src:
        dfile = open(args.src, 'r').read()
    else:
        dfile = sys.stdin.read()

    parsed_text = parse_text(dfile)
    parsed_relations = parser.parse_doc(parsed_text)

    parsed_text['discourse'] = parsed_relations
    pprint(parsed_text)


if __name__ == '__main__':
    main()
