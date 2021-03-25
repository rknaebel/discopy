import datetime
import glob
import json
import os

import click
import spacy

from discopy.data.loaders.raw import load_texts
from discopy.data.relation import Relation


def extract_arguments(annos, text, doc_i):
    args = {}
    for a in annos:
        if a[0][0] == 'T':
            args[a[0]] = {
                'type': a[1].split(" ")[0],
                'offset': text.find(a[2]),
                'length': len(a[2]),
            }
    for a in annos:
        if a[0][0] == 'A':
            arg_type, arg_id, arg_stance = a[1].split(' ')
            args[arg_id]['stance'] = arg_stance
    arguments = []
    for r in annos:
        if r[0][0] == 'R':
            rtype, arg1, arg2 = r[1].split(' ')
            arg1 = args[arg1.split(':')[1]]
            arg2 = args[arg2.split(':')[1]]
            arguments.append({
                'Sense': [rtype],
                'ID': len(arguments),
                'Arg1': arg1,
                'Type': 'Argumentation',
                'DocID': f'essay_{doc_i}',
                'Arg2': arg2
            })
    return arguments


@click.command()
@click.option('-i', '--src', default='-', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(src, tgt):
    nlp = spacy.load('en')
    for doc_i, path in enumerate(sorted(glob.glob(os.path.join(src, '*.txt')))):
        content = open(path).readlines()
        doc = {
            'docID': f'essay_{doc_i:02}',
            'meta': {
                'title': content[0].strip(),
                'corpus': 'argumentative_essays',
                'date': datetime.datetime.now().isoformat()
            },
            'text': '\n'.join(p.strip() for p in content[2:]),
        }
        parses = load_texts(texts=[doc['text']], nlp=nlp)[0]
        doc['sentences'] = [s.to_json() for s in parses.sentences]
        annos = [tuple(a.strip().split("\t")) for a in open(path[:-3] + 'ann').readlines()]
        arguments = extract_arguments(annos, doc['text'], doc_i)
        words = parses.get_tokens()
        relations = [
            Relation([t for t in words if
                      arg['Arg1']['offset'] <= t.offset_begin <= (arg['Arg1']['offset'] + arg['Arg1']['length'])],
                     [t for t in words if
                      arg['Arg2']['offset'] <= t.offset_begin <= (arg['Arg2']['offset'] + arg['Arg2']['length'])],
                     [],
                     arg['Sense'], 'Argumentation') for arg in arguments
        ]
        doc['relations'] = [r.to_json(doc['docID'], r_i) for r_i, r in enumerate(relations)]
        tgt.write(json.dumps(doc) + '\n')


if __name__ == '__main__':
    main()
