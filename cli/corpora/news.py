import datetime

import click
import spacy
import ujson as json
from tqdm import tqdm

from discopy.data.loaders.raw import load_texts


@click.command()
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(src, tgt, limit):
    nlp = spacy.load('en')
    doc_i = 0
    for line in tqdm(src):
        if limit and doc_i > limit:
            break
        doc = json.loads(line)
        parses = load_texts(texts=[doc['Text']], nlp=nlp)[0]
        if len(parses.sentences) <= 2:
            continue
        parses = parses.to_json()
        doc = {
            'docID': f"{doc['Corpus']}_{doc_i:06}",
            'meta': {
                'title': doc['Meta']['title'],
                'corpus': doc['Corpus'],
                'date': datetime.datetime.now().isoformat(),
            },
            'text': parses['text'],
            'sentences': parses['sentences'],
        }
        tgt.write(json.dumps(doc) + '\n')
        doc_i += 1


if __name__ == '__main__':
    main()
