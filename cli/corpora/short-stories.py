import csv
import datetime
import zipfile
from io import TextIOWrapper

import click
import spacy
import ujson as json
from tqdm import tqdm

from discopy.data.loaders.raw import load_texts


@click.command()
@click.option('-i', '--src', default='-', type=click.Path('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(src, tgt, limit):
    nlp = spacy.load('en')
    with zipfile.ZipFile(src, 'r') as zh:
        with zh.open(zh.filelist[0].filename, 'r') as fh:
            reader = csv.reader(TextIOWrapper(fh, 'utf-8'))
            for row_i, row in tqdm(enumerate(reader)):
                if row_i == 0:
                    continue
                if limit and row_i > limit:
                    break
                parses = load_texts(texts=[row[1]], nlp=nlp)[0]
                if len(parses.sentences) == 0:
                    continue
                parses = parses.to_json()
                doc = {
                    'docID': f"short-stories_{row_i:06}",
                    'meta': {
                        'title': row[0],
                        'corpus': 'short-stories',
                        'date': datetime.datetime.now().isoformat(),
                    },
                    'text': parses['text'],
                    'sentences': parses['sentences'],
                }
                tgt.write(json.dumps(doc) + '\n')


if __name__ == '__main__':
    main()
