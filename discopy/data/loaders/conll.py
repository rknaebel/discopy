import json
import os
from collections import defaultdict
from typing import List

import joblib

from discopy.conn_head_mapper import ConnHeadMapper
from discopy.data.doc import ParsedDocument, BertDocument
from discopy.data.sentence import ParsedSentence, DepRel, BertSentence
from discopy.data.relation import Relation
from discopy.data.token import Token
from transformers import AutoTokenizer, TFAutoModel


def load_parsed_conll_dataset(conll_path: str, simple_connectives=False) -> List[ParsedDocument]:
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
            sents.append(ParsedSentence(sent_words, dependencies=dependencies, parsetree=sent['parsetree']))
        doc_pdtb = pdtb.get(doc_id, [])
        if simple_connectives:
            connective_head(doc_pdtb)
        relations = [
            Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                     [words[i[2]] for i in rel['Arg2']['TokenList']],
                     [words[i[2]] for i in rel['Connective']['TokenList']],
                     rel['Sense'], rel['Type']) for rel in doc_pdtb
        ]
        docs.append(ParsedDocument(doc_id=doc_id, sentences=sents, relations=relations))
    return docs


def load_bert_conll_dataset(conll_path: str, simple_connectives=False, limit=0, cache_dir='') -> List[BertDocument]:
    if cache_dir and os.path.exists(cache_dir):
        return joblib.load(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = TFAutoModel.from_pretrained('bert-base-cased')
    parses_path = os.path.join(conll_path, 'parses.json')
    relations_path = os.path.join(conll_path, 'relations.json')
    docs = []
    parses = json.loads(open(parses_path, 'r').read())
    pdtb = defaultdict(list)
    for relation in [json.loads(s) for s in open(relations_path, 'r').readlines()]:
        pdtb[relation['DocID']].append(relation)
    for doc_id, doc in parses.items():
        if limit > 0 and len(docs) > limit:
            break
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
            sents.append(BertSentence.from_tokens(sent_words, tokenizer, model))
        doc_pdtb = pdtb.get(doc_id, [])
        if simple_connectives:
            connective_head(doc_pdtb)
        relations = [
            Relation([words[i[2]] for i in rel['Arg1']['TokenList']],
                     [words[i[2]] for i in rel['Arg2']['TokenList']],
                     [words[i[2]] for i in rel['Connective']['TokenList']],
                     rel['Sense'], rel['Type']) for rel in doc_pdtb
        ]
        docs.append(BertDocument(doc_id=doc_id, sentences=sents, relations=relations))
    if cache_dir:
        joblib.dump(docs, cache_dir)
    return docs


def connective_head(pdtb):
    chm = ConnHeadMapper()
    for rel in filter(lambda r: (r['Type'] == 'Explicit') and (len(r['Connective']['CharacterSpanList']) == 1), pdtb):
        head, head_idxs = chm.map_raw_connective(rel['Connective']['RawText'])
        rel['Connective']['TokenList'] = [rel['Connective']['TokenList'][i] for i in head_idxs]
        rel['Connective']['RawText'] = head
        rel['Connective']['CharacterSpanList'] = [
            [rel['Connective']['TokenList'][0][0], rel['Connective']['TokenList'][-1][1]]]
