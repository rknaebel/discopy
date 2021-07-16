import json

import click

from discopy.parsers.utils import get_parser
from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_parsed_conll_dataset

logger = init_logger()


@click.command()
@click.argument('parser', type=str)
@click.argument('conll-path', type=str)
@click.argument('model-path', type=str)
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(parser, conll_path, model_path, tgt):
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Load pre-trained Parser...')
    parser.load(model_path)
    docs = load_parsed_conll_dataset(conll_path)
    for doc in docs:
        doc = parser.parse_doc(doc)
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()
