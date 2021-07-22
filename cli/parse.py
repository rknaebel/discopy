import json

import click

# TODO run on gpu raises error: supar sequence length datatype problem
from discopy.parsers.utils import get_parser
from discopy.utils import init_logger
from discopy_data.data.doc import Document


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
    for line_i, line in enumerate(src):
        doc = Document.from_json(json.loads(line))
        doc = parser(doc)
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()
