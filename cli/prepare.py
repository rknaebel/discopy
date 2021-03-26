import os

import click

from discopy.data.loaders.conll import load_bert_conll_dataset
from discopy.utils import init_logger

logger = init_logger()


@click.command()
@click.argument('conll-path', type=str)
@click.option('-m', '--mode', default='conll', type=str)
@click.option('-b', '--bert-model', type=str)
def main(conll_path, mode, bert_model=""):
    conll_dir = os.path.join(conll_path, f'en.{mode}')
    cache_path = os.path.join(conll_path, f'en.{mode}.{bert_model}.joblib')
    if not os.path.exists(conll_dir):
        raise FileNotFoundError("Conll file not found.")
    load_bert_conll_dataset(conll_dir,
                            cache_dir=cache_path,
                            bert_model=bert_model)


if __name__ == '__main__':
    main()
