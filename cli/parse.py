import os

from discopy.data.update import update_dataset_parses

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import click
# TODO run on gpu raises error: supar sequence length datatype problem
from discopy.data.loaders.raw import load_texts
from discopy.parsers.utils import get_parser
from discopy.utils import init_logger


@click.command()
@click.argument('parser', type=str)
@click.argument('model-path', type=str)
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(parser, model_path, src, tgt):
    logger = init_logger()
    logger.info('Init Parser...')
    parser = get_parser(parser)
    logger.info('Load pre-trained Parser...')
    parser.load(model_path)
    docs = load_texts([src.read()])
    update_dataset_parses(docs)
    doc = parser(docs[0])
    tgt.write(str(doc))


if __name__ == '__main__':
    main()
