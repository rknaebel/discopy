import logging
import multiprocessing
import ujson as json

import nltk

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
