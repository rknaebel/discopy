import bz2
import sys

import click

from discopy.evaluate.conll import evaluate_docs, print_results
from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_parsed_conll_dataset
from discopy_data.data.loaders.json import load_documents

logger = init_logger()


def get_fh(path):
    if path.endswith('json'):
        fh = open(path)
    elif path.endswith('bz2'):
        fh = bz2.open(path, 'rt')
    elif path == '-':
        fh = sys.stdin
    else:
        raise ValueError('Unknown data ending')
    return fh


@click.command()
@click.argument('conll-path', type=click.Path())
@click.argument('pred-path', type=click.Path())
@click.option('-t', '--threshold', default=0.9, type=float)
def main(conll_path, pred_path, threshold):
    gold_docs = load_parsed_conll_dataset(conll_path, simple_connectives=True)
    pred_docs = load_documents(get_fh(pred_path))
    pred_doc_ids = {doc.doc_id for doc in pred_docs}
    gold_docs = [doc for doc in gold_docs if doc.doc_id in pred_doc_ids]
    if not pred_docs or not gold_docs:
        logger.warning('No documents found')
        return
    print_results(evaluate_docs(
        [doc.with_relations([r for r in doc.relations if r.is_explicit()]) for doc in gold_docs],
        [doc.with_relations([r for r in doc.relations if r.is_explicit()]) for doc in pred_docs],
        threshold=threshold), title='EXPLICIT')
    print_results(evaluate_docs(
        [doc.with_relations([r for r in doc.relations if not r.is_explicit()]) for doc in gold_docs],
        [doc.with_relations([r for r in doc.relations if not r.is_explicit()]) for doc in pred_docs],
        threshold=threshold), title='NON-EXPLICIT')
    print_results(evaluate_docs(gold_docs, pred_docs, threshold=threshold), title='ALL')


if __name__ == '__main__':
    main()
