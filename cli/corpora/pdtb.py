import datetime
import json
import os
from collections import defaultdict

import click
from tqdm import tqdm


@click.command()
@click.option('-i', '--src', default='-', type=click.Path(exists=True))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(src, tgt):
    relations_grouped = defaultdict(list)
    docs = json.load(open(os.path.join(src, "parses.json")))
    relations = [json.loads(line) for line in open(os.path.join(src, "relations.json"))]
    for rel in relations:
        relations_grouped[rel['DocID']].append(rel)

    for doc_id, doc in tqdm(docs.items()):
        sentences = [''.join([sentence['words'][0][0]] +
                             [('' if sentence['words'][t_i][1]['CharacterOffsetEnd'] == t[1][
                                 'CharacterOffsetBegin'] else ' ') + t[0]
                              for t_i, t in enumerate(sentence['words'][1:])]) for sentence in doc['sentences']]
        doc = {
            'docID': doc_id,
            'meta': {
                'corpus': 'pdtb',
                'part': os.path.basename(src),
                'date': datetime.datetime.now().isoformat(),
            },
            'text': "\n".join(sentences),
            'sentences': doc['sentences'],
            'relations': relations_grouped[doc_id],
        }
        tgt.write(json.dumps(doc) + '\n')


if __name__ == '__main__':
    main()
