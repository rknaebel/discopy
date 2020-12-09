import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import click
# TODO run on gpu raises error: supar sequence length datatype problem
from discopy.data.loaders.raw import load_texts
from discopy.parsers.utils import get_parser
from discopy.utils import init_logger

logger = init_logger()


@click.command()
@click.argument('parser', type=str)
@click.argument('model-path', type=str)
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(parser, model_path, src, tgt):
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Load pre-trained Parser...')
    parser.load(model_path)
    parsed_text = load_texts([src.read()])[0]
    doc = parser(parsed_text)
    tgt.write(doc.to_json())


if __name__ == '__main__':
    main()
