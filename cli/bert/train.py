import logging
import os

import click

from discopy.components.argument.bert.conn import ConnectiveArgumentExtractor
from discopy.components.sense.explicit.bert_conn_sense import ConnectiveSenseClassifier
from discopy.components.sense.implicit.bert_adj_sentence import NonExplicitRelationClassifier
from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_bert_conll_dataset

logger = logging.getLogger('discopy')


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
@click.argument('save-path', type=str)
@click.option('--simple-connectives', is_flag=True)
@click.option('--sense-level', default=2, type=int)
def main(bert_model, conll_path, save_path, simple_connectives, sense_level):
    logger = init_logger()
    logger.info('Load train data')
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         simple_connectives=simple_connectives,
                                         cache_dir=os.path.join(conll_path, f'en.train.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_level)
    logger.info('Load dev data')
    docs_val = load_bert_conll_dataset(os.path.join(conll_path, 'en.dev'),
                                       simple_connectives=simple_connectives,
                                       cache_dir=os.path.join(conll_path, f'en.dev.{bert_model}.joblib'),
                                       bert_model=bert_model,
                                       sense_level=sense_level)
    logger.info('Init model')
    parser = ParserPipeline([
        ConnectiveSenseClassifier(input_dim=docs_val[0].get_embedding_dim(), used_context=1),
        ConnectiveArgumentExtractor(window_length=100, input_dim=docs_val[0].get_embedding_dim(), hidden_dim=512,
                                    rnn_dim=512, ckpt_path=save_path),
        NonExplicitRelationClassifier(input_dim=docs_val[0].get_embedding_dim(), arg_length=50),
    ])
    logger.info('Train model')
    parser.fit(docs_train, docs_val)
    parser.save(save_path)
    parser.score(docs_val)


if __name__ == "__main__":
    main()
