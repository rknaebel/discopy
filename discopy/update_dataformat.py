import os
import sys
import ujson as json

import benepar
import spacy
from benepar.spacy_plugin import BeneparComponent
from tqdm import tqdm

sys.path.append('/home/users/rknaebel/project/discourse/')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

nlp = spacy.load('en')
benepar.download('benepar_en2')
nlp.add_pipe(BeneparComponent("benepar_en2"))

parses_train = json.loads(open('../data/conll2016/en.train/parses.json').read())

new_parses = {}
for doc_id, doc_item in tqdm(parses_train.items()):
    sentences = []
    for sent_item in doc_item['sentences']:
        words = sent_item['words']
        s = [words[0][0]]
        for i, w in enumerate(words[1:]):
            nb_whitespaces = w[1]['CharacterOffsetBegin'] - words[i][1]['CharacterOffsetEnd']
            s.append("{}{}".format(" " * nb_whitespaces, w[0]))
        sentences.append(''.join(s))

    new_parses[doc_id] = {
        'DocID': doc_id,
        'sentences': [],
    }
    for sent_no, doc in enumerate(tqdm(nlp.pipe(sentences), leave=False)):
        sent_offset = doc_item['sentences'][sent_no]['words'][0][1]['CharacterOffsetBegin']
        sent = list(doc.sents)[0]
        new_parses[doc_id]['sentences'].append({
            'DocID': doc_id,
            'SentenceNumber': sent_no,
            'Sentence': sent.string.strip(),
            'Length': len(sent.string),
            'Tokens': [t.text for t in sent],
            'POS': [t.pos_ for t in sent],
            'Offset': [t.idx - sent[0].idx for t in sent],
            'Dep': [(t.dep_, (t.head.text, t.head.i), (t.text, t.i)) for t in sent],
            'SentenceOffset': sent_offset,
            'Parse': sent._.parse_string,
            'NER_iob': [t.ent_iob_ for t in sent],
            'NER_type': [t.ent_type_ for t in sent],
        })

with open('/data/discourse/pdtb2_new.json', 'w') as fh:
    for doc_id, doc in new_parses.items():
        fh.write("{}\n".format(json.dumps(doc)))
