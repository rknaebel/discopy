import logging
import multiprocessing
import ujson as json

import nltk
import spacy
from benepar.spacy_plugin import BeneparComponent

from discopy.conn_head_mapper import ConnHeadMapper

logger = logging.getLogger('discopy')


def load_parse_trees(doc_id, parse_strings):
    results = []
    for sent_id, sent in enumerate(parse_strings):
        try:
            ptree = nltk.Tree.fromstring(sent.strip())
            if not ptree.leaves():
                logger.warning('Failed on empty tree')
                results.append(None)
            else:
                results.append(ptree)
        except ValueError:
            logger.warning('Failed to parse doc {} idx {}'.format(doc_id, sent_id))
            results.append(None)
    return doc_id, results


def get_conll_dataset(data_path, mode, load_trees=False, connective_mapping=True):
    pdtb = [json.loads(s) for s in open('{}/{}/relations.json'.format(data_path, mode), 'r').readlines()]
    parses = json.loads(open('{}/{}/parses.json'.format(data_path, mode), 'r').read())
    if load_trees:
        with multiprocessing.Pool(4) as pool:
            args = [(doc_id, [s['parsetree'] for s in doc['sentences']])
                    for doc_id, doc in parses.items()]
            parse_trees = pool.starmap(load_parse_trees, args, chunksize=5)
        for doc_id, ptrees in parse_trees:
            for sent_i, ptree in enumerate(ptrees):
                parses[doc_id]['sentences'][sent_i]['parsetree'] = nltk.ParentedTree.convert(ptree)
    if connective_mapping:
        chm = ConnHeadMapper()
        for r in filter(lambda r: (r['Type'] == 'Explicit') and (len(r['Connective']['CharacterSpanList']) == 1), pdtb):
            head, head_idxs = chm.map_raw_connective(r['Connective']['RawText'])
            r['Connective']['TokenList'] = [r['Connective']['TokenList'][i] for i in head_idxs]
            r['Connective']['RawText'] = head
            r['Connective']['CharacterSpanList'] = [
                [r['Connective']['TokenList'][0][0], r['Connective']['TokenList'][-1][1]]]

    return parses, pdtb


def parse_sentence(nlp, sentence):
    words = [w[0] for w in sentence['words']]
    doc = nlp(words)
    sent = list(doc.sents)[0]
    return {
        'dependencies': [(t.dep_, "{}-{}".format(t.head.text, t.head.i + 1), "{}-{}".format(t.text, t.i + 1)) for t in
                         sent],
        'parsetree': sent._.parse_string,
        'words': [
            (w_orig[0], {
                'CharacterOffsetBegin': w_orig[1]['CharacterOffsetBegin'],
                'CharacterOffsetEnd': w_orig[1]['CharacterOffsetEnd'],
                'Linkers': w_orig[1]['Linkers'],
                'PartOfSpeech': w_doc.tag_
            }) for w_doc, w_orig in zip(sent, sentence['words'])
        ]
    }
    # return {
    #     'Sentence': sent.string.strip(),
    #     'Length': len(sent.string),
    #     'Tokens': [t.text for t in sent],
    #     'POS': [t.tag_ for t in sent],
    #     'Offset': [t.idx - sent[0].idx for t in sent],
    #     'Dep': [(t.dep_, (t.head.text, t.head.i), (t.text, t.i)) for t in sent],
    #     'SentenceOffset': offset + sent[0].idx,
    #     'Parse': sent._.parse_string,
    #     'NER_iob': [t.ent_iob_ for t in sent],
    #     'NER_type': [t.ent_type_ for t in sent],
    # }


def get_conll_dataset_extended(data_path, mode, connective_mapping=True):
    parses, pdtb = get_conll_dataset(data_path, mode, False, connective_mapping)
    nlp = spacy.load('en')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    nlp.add_pipe(BeneparComponent("benepar_en2"))
    for doc_id, doc in parses.items():
        for sent_idx, sentence in enumerate(doc['sentences']):
            doc['sentences'][sent_idx] = parse_sentence(nlp, sentence)
    return parses, pdtb
