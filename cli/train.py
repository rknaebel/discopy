import os

import click

from discopy.data.loaders.conll import load_parsed_conll_dataset
from discopy.parsers.utils import get_parser
from discopy.utils import init_logger

logger = init_logger()


@click.command()
@click.argument('parser', type=str)
@click.argument('model-path', type=str)
@click.argument('conll-path', type=str)
def main(parser, model_path, conll_path):
    docs_train = load_parsed_conll_dataset(os.path.join(conll_path, 'en.train'), simple_connectives=True)
    docs_val = load_parsed_conll_dataset(os.path.join(conll_path, 'en.dev'), simple_connectives=True)
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Train end-to-end Parser...')
    parser.fit(docs_train)
    logger.info('Scores on validation data')
    parser.score(docs_val)
    logger.info('Save Parser')
    parser.save(os.path.join(model_path))


if __name__ == '__main__':
    main()
