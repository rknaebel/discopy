import os

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')

import click

from discopy.data.loaders.conll import load_conll_dataset
from discopy.evaluate.conll import evaluate_docs, print_results
from discopy.parsers.utils import get_parser
from discopy.utils import init_logger

logger = init_logger()


@click.command()
@click.argument('parser', type=str)
@click.argument('model-path', type=str)
@click.argument('conll-path', type=str)
@click.option('-t', '--threshold', default=0.9, type=str)
def main(parser, model_path, conll_path, threshold):
    docs_test = load_conll_dataset(os.path.join(conll_path, 'en.test'), simple_connectives=True)
    docs_blind = load_conll_dataset(os.path.join(conll_path, 'en.blind-test'), simple_connectives=True)
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Load pre-trained Parser...')
    parser.load(model_path)
    logger.info('component evaluation (test)')
    parser.score(docs_test)
    logger.info('extract discourse relations from test data')
    preds = [parser(d) for d in docs_test]
    print_results(evaluate_docs(docs_test, preds, threshold=threshold))
    logger.info('extract discourse relations from BLIND data')
    preds = [parser(d) for d in docs_blind]
    print_results(evaluate_docs(docs_blind, preds, threshold=threshold))


if __name__ == '__main__':
    main()
