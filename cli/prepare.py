import os

import click

from discopy.data.loaders.conll import load_bert_conll_dataset
from discopy.utils import init_logger

logger = init_logger()


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
def main(bert_model, conll_path):
    for mode in ['train', 'dev', 'test', 'blind-test']:
        conll_dir = os.path.join(conll_path, f'en.{mode}')
        cache_path = os.path.join(conll_path, f'en.{mode}.{bert_model}.joblib')
        if os.path.exists(conll_dir) and not os.path.exists(cache_path):
            load_bert_conll_dataset(conll_dir,
                                    cache_dir=cache_path,
                                    bert_model=bert_model)


if __name__ == '__main__':
    main()
