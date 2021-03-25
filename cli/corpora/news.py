import datetime

import click
import spacy
import ujson as json
from tqdm import tqdm

from discopy.data.loaders.raw import load_texts


@click.command()
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(src, tgt):
    nlp = spacy.load('en')
    for line in tqdm(src):
        doc = json.loads(line)
        parses = load_texts(texts=[doc['Text']], nlp=nlp)[0]
        doc = {
            'docID': doc['DocID'],
            'meta': {
                'title': doc['Meta']['title'],
                'corpus': doc['Corpus'],
                'date': datetime.datetime.now().isoformat(),
            },
            'text': [s.to_json() for s in parses.sentences],
            'sentences': [s.to_json() for s in parses.sentences],
        }
        tgt.write(json.dumps(doc) + '\n')


if __name__ == '__main__':
    main()
