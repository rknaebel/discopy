import argparse
import os

import joblib

from discopy.data.conll16 import get_conll_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm

from discopy.utils import init_logger

import spacy

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--model", help="",
                             default='en_trf_bertbaseuncased_lg')
argument_parser.add_argument("--conll", help="",
                             default='')
args = argument_parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

logger = init_logger()


def normalize_token(t):
    if t == "n't":
        return "not"
    else:
        return t


def get_doc_sentences(parses):
    return {doc_id: [[normalize_token(w[0]) for w in s['words']] for s in doc['sentences']] for doc_id, doc in
            parses.items()}


def transform_to_bert_emebddings(nlp, parses, use_all_embeddings=False):
    parses_bert = {}
    sentences = get_doc_sentences(parses)
    for doc_id, doc in tqdm(parses.items()):
        parses_bert[doc_id] = []
        docs = nlp.pipe(sentences[doc_id])
        for sent, embd in zip(doc['sentences'], docs):
            parses_bert[doc_id].append(embd._.trf_last_hidden_state[[a[0] for a in embd._.trf_alignment]])
    return parses_bert


def main():
    nlp = spacy.load(args.model)
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

    parses_train, pdtb_train = get_conll_dataset(args.conll, 'en.train', load_trees=False, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(args.conll, 'en.dev', load_trees=False, connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset(args.conll, 'en.test', load_trees=False, connective_mapping=True)
    parses_blind, pdtb_blind = get_conll_dataset(args.conll, 'en.blind-test', load_trees=False, connective_mapping=True)

    bert_train = transform_to_bert_emebddings(nlp, parses_train)
    bert_val = transform_to_bert_emebddings(nlp, parses_val)
    bert_test = transform_to_bert_emebddings(nlp, parses_test)
    bert_blind = transform_to_bert_emebddings(nlp, parses_blind)

    joblib.dump(bert_train, os.path.join(args.conll, 'en.train', 'bert.joblib'))
    joblib.dump(bert_val, os.path.join(args.conll, 'en.dev', 'bert.joblib'))
    joblib.dump(bert_test, os.path.join(args.conll, 'en.test', 'bert.joblib'))
    joblib.dump(bert_blind, os.path.join(args.conll, 'en.blind-test', 'bert.joblib'))


if __name__ == '__main__':
    main()
