import datetime
import re
import zipfile

import click
import spacy
import ujson as json
from tqdm import tqdm

from discopy.data.loaders.raw import load_texts

WS = re.compile(r'[\s\xa0]+')


def clean_text(s):
    return re.sub(WS, ' ', s)


@click.command()
@click.option('-i', '--src', default='-', type=click.Path('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(src, tgt, limit):
    nlp = spacy.load('en')
    doc_i = 0
    with zipfile.ZipFile(src, 'r') as zh:
        with zh.open(zh.filelist[0].filename, 'r') as fh:
            for row_i, row in tqdm(enumerate(fh.read().decode().splitlines())):
                if row_i == 0:
                    continue
                if limit and row_i > limit:
                    break
                doc = json.loads(row)
                contents = re.sub(WS, ' ', doc['contents'])
                parses = load_texts(texts=[contents], nlp=nlp)[0]
                if len(parses.sentences) <= 2:
                    continue
                parses = parses.to_json()
                doc = {
                    'docID': f"press-gov_{doc_i:05}",
                    'meta': {
                        'title': re.sub(WS, ' ', doc['title']),
                        'corpus': 'press-gov',
                        'published': doc['date'],
                        'date': datetime.datetime.now().isoformat(),
                        'components': doc['components']
                    },
                    'text': parses['text'],
                    'sentences': parses['sentences'],
                }
                tgt.write(json.dumps(doc) + '\n')
                doc_i += 1


if __name__ == '__main__':
    main()
