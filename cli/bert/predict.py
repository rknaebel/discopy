import json
import os

import click

from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_bert_conll_dataset


@click.command()
@click.argument('bert-model', type=str)
@click.argument('model-path', type=str)
@click.argument('conll-path', type=str)
@click.option('--cache-path', default='', type=str)
@click.option('-o', '--tgt', default='-', type=click.File('w'))
def main(bert_model, conll_path, cache_path, model_path, tgt):
    logger = init_logger()
    logger.info('Load train data')
    dataset_part = os.path.basename(conll_path)
    docs = load_bert_conll_dataset(conll_path,
                                   cache_dir=os.path.join(cache_path, f'{dataset_part}.{bert_model}.joblib'),
                                   bert_model=bert_model)
    logger.info('Init Parser')
    parser = ParserPipeline.from_config(model_path)
    logger.info('LOAD model')
    parser.load(model_path)
    logger.info(f'Predict on dataset: {conll_path}')
    for doc in docs:
        doc = parser.parse_doc(doc)
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()
