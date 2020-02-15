import multiprocessing
import os
import ujson as json

import benepar
import nltk
import spacy
from tqdm import tqdm

from discopy.data.conll16 import load_parse_trees


def load_corpus(path, file_name):
    docs_path = "{}/{}_new.json".format(path, file_name)
    print("get corpus:", file_name)
    if os.path.exists(docs_path):
        documents = json.load(open(docs_path, 'r'))
        with multiprocessing.Pool(4) as pool:
            args = [(doc_id, [s['parsetree'] for s in doc['sentences']])
                    for doc_id, doc in documents.items()]
            parse_trees = pool.starmap(load_parse_trees, args, chunksize=5)
        for doc_id, ptrees in parse_trees:
            for sent_i, ptree in enumerate(ptrees):
                documents[doc_id]['sentences'][sent_i]['parsetree'] = nltk.ParentedTree.convert(ptree)

    else:
        nlp = spacy.load('en')
        nlp.tokenizer = nlp.tokenizer.tokens_from_list
        parser = benepar.Parser('benepar_en2')
        tbwt = nltk.TreebankWordTokenizer()
        tbwd = nltk.treebank.TreebankWordDetokenizer()
        documents = {}
        with open("{}/{}.json".format(path, file_name), 'r') as corpus_fh:
            for raw_doc in tqdm(corpus_fh.readlines()):
                try:
                    doc = convert_to_conll(json.loads(raw_doc), nlp, parser, tbwt, tbwd)
                    documents[doc['DocID'].strip()] = doc
                except Exception as e:
                    print('Failed to parse document:')
                    print(e)
                    continue
        print('save corpus')
        json.dump(documents, open(docs_path, 'w'))
    return documents


def convert_to_conll(document, nlp, parser, tbwt, tbwd):
    text = " ".join([
        tbwd.detokenize(tbwt.tokenize(s)).replace(r'\"', '``').replace('"', "''")
        for s in nltk.sent_tokenize(document['Text'])])
    sentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    ptrees = parser.parse_sents(sentences)
    doc = nlp.pipe(sentences)
    res = []
    offset = 0
    for sent, ptree in tqdm(zip(doc, ptrees), total=len(sentences), desc='sentences', leave=False):
        words = []
        for tok in sent:
            token_offset = text.find(tok.text, offset)
            offset = token_offset + len(tok.text)
            words.append((tok.text, {
                'CharacterOffsetBegin': token_offset,
                'CharacterOffsetEnd': offset,
                'Linkers': [],
                'PartOfSpeech': tok.tag_,
                'SimplePartOfSpeech': tok.pos_,
                'Lemma': tok.lemma_,
                'Shape': tok.shape_,
                'EntIOB': tok.ent_iob_,
                'EntType': tok.ent_type_
            }))

        sentence_conll = {
            'dependencies': [(t.dep_, "{}-{}".format(t.head.text, t.head.i + 1), "{}-{}".format(t.text, t.i + 1)) for t
                             in
                             sent],
            'parsetree': ptree._pformat_flat('', '()', False),
            'words': words,
        }
        res.append(sentence_conll)
    return {
        'Corpus': document['Corpus'],
        'DocID': document['DocID'],
        'raw': document['Text'],
        'text': text,
        'sentences': res,
    }
