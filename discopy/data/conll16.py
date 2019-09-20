import logging
import ujson as json

import nltk

from discopy.conn_head_mapper import ConnHeadMapper

logger = logging.getLogger('discopy')


def get_conll_dataset(data_path, mode, load_trees=False, connective_mapping=True):
    pdtb = [json.loads(s) for s in open('{}/{}/relations.json'.format(data_path, mode), 'r').readlines()]
    parses = json.loads(open('{}/{}/parses.json'.format(data_path, mode), 'r').read())
    if load_trees:
        for doc_id, doc in parses.items():
            for sent_id, sent in enumerate(doc['sentences']):
                try:
                    ptree = nltk.ParentedTree.fromstring(sent['parsetree'])
                    if not ptree.leaves():
                        logger.warning('Failed on empty tree')
                        sent['parsetree'] = None
                    else:
                        sent['parsetree'] = ptree
                except ValueError:
                    logger.warning('Failed to parse doc {} idx {}'.format(doc['DocID'], sent_id))
                    sent['parsetree'] = None
    if connective_mapping:
        chm = ConnHeadMapper()
        for r in filter(lambda r: (r['Type'] == 'Explicit') and (len(r['Connective']['CharacterSpanList']) == 1), pdtb):
            head, head_idxs = chm.map_raw_connective(r['Connective']['RawText'])
            r['Connective']['TokenList'] = [r['Connective']['TokenList'][i] for i in head_idxs]
            r['Connective']['RawText'] = head
            r['Connective']['CharacterSpanList'] = [
                [r['Connective']['TokenList'][0][0], r['Connective']['TokenList'][-1][1]]]

    return parses, pdtb
