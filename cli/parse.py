import json
import os

# TODO run on gpu raises error: supar sequence length datatype problem
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import click
import nltk
from discopy.parsers import get_parser

import spacy
from discopy.utils import init_logger
import supar

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


def ptree_to_str(ptree):
    if ptree is not None:
        ptree = ptree._pformat_flat('', '()', False)
    return ptree


@click.command()
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-m', '--model-path', type=str)
def main(src, tgt, model_path):
    logger.info('Init Parser...')
    parser = get_parser('lin')
    logger.info('Load pre-trained Parser...')
    parser.load(model_path)
    parsed_text = parse_text(src.read())
    parsed_relations = parser.parse_doc(parsed_text)
    parsed_text['discourse'] = parsed_relations
    for sent in parsed_text['sentences']:
        sent['parsetree'] = ptree_to_str(sent['parsetree'])
    tgt.write(json.dumps(parsed_text))


if __name__ == '__main__':
    main()
