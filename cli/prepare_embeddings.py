import os

import click

from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_bert_conll_dataset

logger = init_logger()


@click.command()
@click.argument('conll-path', type=str)
@click.argument('cache-path', type=str)
@click.argument('bert-model', type=str)
def main(conll_path, cache_path, bert_model):
    conll_part = os.path.basename(conll_path)
    cache_path = os.path.join(cache_path, f'{conll_part}.{bert_model}.joblib')
    if not os.path.exists(conll_path):
        raise FileNotFoundError("Conll dir not found.")
    load_bert_conll_dataset(conll_path,
                            cache_dir=cache_path,
                            bert_model=bert_model)


if __name__ == '__main__':
    main()
