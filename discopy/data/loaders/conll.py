import json
import os
from collections import defaultdict

from discopy.data.doc import Token, DepRel, Relation, Document, Sentence


def load_conll_dataset(conll_path):
    parses_path = os.path.join(conll_path, 'parses.json')
    relations_path = os.path.join(conll_path, 'relations.json')
    docs = {}

    parses = json.loads(open(parses_path, 'r').read())
    pdtb = defaultdict(list)
    for relation in [json.loads(s) for s in open(relations_path, 'r').readlines()]:
        pdtb[relation['DocID']].append(relation)
    for doc_id, doc in parses.items():
        words = []
        token_offset = 0
        sents = []
        for sent_i, sent in enumerate(doc['sentences']):
            sent_words = [
                Token(token_offset + w_i, sent_i, w_i, t['CharacterOffsetBegin'], t['CharacterOffsetEnd'], surface,
                      t['PartOfSpeech'])
                for w_i, (surface, t) in enumerate(sent['words'])
            ]
            words.extend(sent_words)
            token_offset += len(sent_words)
            dependencies = [
                DepRel(rel=rel,
                       head=words[int(head.split('-')[-1]) - 1] if not head.startswith('ROOT') else None,
                       dep=words[int(dep.split('-')[-1]) - 1]
                       ) for rel, head, dep in sent['dependencies']
            ]
            sents.append(Sentence(sent_words, dependencies=dependencies, parsetree=sent['parsetree']))

        relations = [
            Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                     [words[i[2]] for i in rel['Arg2']['TokenList']],
                     [words[i[2]] for i in rel['Connective']['TokenList']],
                     rel['Sense'], rel['Type']) for rel in pdtb.get(doc_id, [])
        ]

        docs[doc_id] = Document(doc_id=doc_id, sentences=sents, relations=relations)

    return docs
