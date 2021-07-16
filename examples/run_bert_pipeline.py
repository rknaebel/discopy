import logging
import os

import click

from discopy.components.argument.bert.conn import ConnectiveArgumentExtractor
from discopy.components.sense.explicit.bert_conn_sense import ConnectiveSenseClassifier
from discopy.components.sense.implicit.bert_adj_sentence import NonExplicitRelationClassifier
from discopy.evaluate.conll import print_results, evaluate_docs
from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_bert_conll_dataset

logger = logging.getLogger('discopy')


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
@click.argument('save-path', type=str)
@click.option('--simple-connectives', is_flag=True)
@click.option('-s', '--sense-lvl', default=2, type=int)
def main(bert_model, conll_path, save_path, simple_connectives, sense_lvl=2):
    logger = init_logger()
    logger.info('Load train data')
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         simple_connectives=simple_connectives,
                                         cache_dir=os.path.join(conll_path, f'en.train.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_lvl)
    logger.info('Load dev data')
    docs_val = load_bert_conll_dataset(os.path.join(conll_path, 'en.dev'),
                                       simple_connectives=simple_connectives,
                                       cache_dir=os.path.join(conll_path, f'en.dev.{bert_model}.joblib'),
                                       bert_model=bert_model,
                                       sense_level=sense_lvl)
    docs_test = load_bert_conll_dataset(os.path.join(conll_path, 'en.test'),
                                        simple_connectives=simple_connectives,
                                        cache_dir=os.path.join(conll_path, f'en.test.{bert_model}.joblib'),
                                        bert_model=bert_model,
                                        sense_level=sense_lvl)
    docs_blind = load_bert_conll_dataset(os.path.join(conll_path, 'en.blind-test'),
                                         simple_connectives=simple_connectives,
                                         cache_dir=os.path.join(conll_path, f'en.blind-test.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_lvl)
    logger.info('Init model')
    parser = ParserPipeline([
        ConnectiveSenseClassifier(input_dim=docs_val[0].get_embedding_dim(), used_context=1),
        ConnectiveArgumentExtractor(window_length=100, input_dim=docs_val[0].get_embedding_dim(), hidden_dim=256,
                                    rnn_dim=512, ckpt_path=save_path),
        NonExplicitRelationClassifier(input_dim=docs_val[0].get_embedding_dim(), arg_length=50),
    ])
    logger.info('Train model')
    parser.fit(docs_train, docs_val)
    parser.save(save_path)
    logger.info('LOAD model')
    parser.load(save_path)
    parser.score(docs_val)
    logger.info('Evaluate parser TEST')
    docs_test_expl = [d.with_relations(d.get_explicit_relations()) for d in docs_test]
    test_preds = parser.parse(docs_test_expl)
    print_results(evaluate_docs(docs_test_expl, test_preds, threshold=0.7), title='test-0.7')
    print_results(evaluate_docs(docs_test_expl, test_preds, threshold=0.9), title='test-0.9')
    logger.info('Evaluate parser BLIND')
    docs_blind_expl = [d.with_relations(d.get_explicit_relations()) for d in docs_blind]
    blind_preds = parser.parse(docs_blind_expl)
    print_results(evaluate_docs(docs_blind_expl, blind_preds, threshold=0.7), title='blind-0.7')
    print_results(evaluate_docs(docs_blind_expl, blind_preds, threshold=0.9), title='blind-0.9')


if __name__ == "__main__":
    main()
