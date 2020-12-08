import json
import os
from collections import defaultdict
from typing import List

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.data.doc import DepRel, Document, Sentence
from discopy.data.relation import Relation
from discopy.data.token import Token


def load_conll_dataset(conll_path: str, simple_connectives=False) -> List[Document]:
    parses_path = os.path.join(conll_path, 'parses.json')
    relations_path = os.path.join(conll_path, 'relations.json')
    docs = []
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
        doc_pdtb = pdtb.get(doc_id, [])
        if simple_connectives:
            connective_head(doc_pdtb)
        relations = [
            Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                     [words[i[2]] for i in rel['Arg2']['TokenList']],
                     [words[i[2]] for i in rel['Connective']['TokenList']],
                     rel['Sense'], rel['Type']) for rel in doc_pdtb
        ]
        docs.append(Document(doc_id=doc_id, sentences=sents, relations=relations))
    return docs


def connective_head(pdtb):
    chm = ConnHeadMapper()
    for r in filter(lambda r: (r['Type'] == 'Explicit') and (len(r['Connective']['CharacterSpanList']) == 1), pdtb):
        head, head_idxs = chm.map_raw_connective(r['Connective']['RawText'])
        r['Connective']['TokenList'] = [r['Connective']['TokenList'][i] for i in head_idxs]
        r['Connective']['RawText'] = head
        r['Connective']['CharacterSpanList'] = [
            [r['Connective']['TokenList'][0][0], r['Connective']['TokenList'][-1][1]]]
