import os

import click
import joblib
from spacy_transformers import TransformersLanguage, TransformersWordPiecer, TransformersTok2Vec

from discopy.data.conll16 import get_conll_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm

from discopy.utils import init_logger

logger = init_logger()


def normalize_token(t):
    if t == "n't":
        return "not"
    else:
        return t


def get_doc_sentences(parses):
    return {doc_id: [[normalize_token(w[0]) for w in s['words']] for s in doc['sentences']] for doc_id, doc in
            parses.items()}


def transform_to_bert_emebddings(nlp, parses):
    parses_bert = {}
    sentences = get_doc_sentences(parses)
    for doc_id, doc in tqdm(parses.items()):
        parses_bert[doc_id] = []
        docs = nlp.pipe(sentences[doc_id])
        for sent, embd in zip(doc['sentences'], docs):
            parses_bert[doc_id].append(embd._.trf_last_hidden_state[[a[0] for a in embd._.trf_alignment]])
    return parses_bert


@click.command()
@click.argument('model', type=str)
@click.argument('model-name', type=str)
@click.argument('conll-path', type=str)
@click.argument('lang', type=str)
def main(model, model_name, conll_path, lang):
    nlp = TransformersLanguage(trf_name=model, meta={"lang": lang})
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    nlp.add_pipe(TransformersWordPiecer.from_pretrained(nlp.vocab, model))
    nlp.add_pipe(TransformersTok2Vec.from_pretrained(nlp.vocab, model))
    nlp.tokenizer = nlp.tokenizer.tokens_from_list

    parses_train, pdtb_train = get_conll_dataset(conll_path, 'en.train', load_trees=False, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(conll_path, 'en.dev', load_trees=False, connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset(conll_path, 'en.test', load_trees=False, connective_mapping=True)
    parses_blind, pdtb_blind = get_conll_dataset(conll_path, 'en.blind-test', load_trees=False, connective_mapping=True)
    parses_pcc, pdtb_pcc = get_conll_dataset(conll_path, 'de.pcc', load_trees=False, connective_mapping=False,
                                             types=None)

    bert_train = transform_to_bert_emebddings(nlp, parses_train)
    bert_val = transform_to_bert_emebddings(nlp, parses_val)
    bert_test = transform_to_bert_emebddings(nlp, parses_test)
    bert_blind = transform_to_bert_emebddings(nlp, parses_blind)
    bert_pcc = transform_to_bert_emebddings(nlp, parses_pcc)

    joblib.dump(bert_train, os.path.join(conll_path, 'en.train', '{}.joblib'.format(model_name)))
    joblib.dump(bert_val, os.path.join(conll_path, 'en.dev', '{}.joblib'.format(model_name)))
    joblib.dump(bert_test, os.path.join(conll_path, 'en.test', '{}.joblib'.format(model_name)))
    joblib.dump(bert_blind, os.path.join(conll_path, 'en.blind-test', '{}.joblib'.format(model_name)))
    joblib.dump(bert_pcc, os.path.join(conll_path, 'de.pcc', '{}.joblib'.format(model_name)))


if __name__ == '__main__':
    main()
