import bz2
import json
import logging
import os
import random
import time

from discopy.components.argument.bert.implicit import ImplicitArgumentExtractor
from discopy.components.sense.implicit.bert_arguments import ArgumentSenseClassifier
from discopy_data.data.doc import Document

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import click
import tensorflow as tf

from discopy.components.argument.bert.conn import ConnectiveArgumentExtractor
from discopy.components.sense.explicit.bert_conn_sense import ConnectiveSenseClassifier
from discopy.evaluate.conll import print_results, evaluate_docs
from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
from discopy_data.data.loaders.conll import load_bert_embeddings

logger = logging.getLogger('discopy')
tf.get_logger().setLevel(logging.WARNING)


def load_docs(bzip_file_path):
    docs = []
    for line_i, line in enumerate(bz2.open(filename=bzip_file_path, mode='rt')):
        try:
            docs.append(Document.from_json(json.loads(line)))
        except json.JSONDecodeError:
            continue
        except EOFError:
            break
    return docs


def split_train_test(xs, ratio=0.9):
    xs = xs[:]
    num_samples = int(len(xs) * ratio)
    random.shuffle(xs)
    return xs[:num_samples], xs[num_samples:]


def print_results_latex(explicits, non_explicits, title=''):
    explicit_values = '{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(
        explicits['sense'][2] * 100, explicits['conn'][2] * 100,
        explicits['arg1'][2] * 100, explicits['arg2'][2] * 100, explicits['arg12'][2] * 100)
    non_explicit_values = '{:.2f} & {:.2f} & {:.2f} & {:.2f}'.format(
        non_explicits['sense'][2] * 100,
        non_explicits['arg1'][2] * 100, non_explicits['arg2'][2] * 100, non_explicits['arg12'][2] * 100)
    logger.info('{} & {} & {} \\\\'.format(title, explicit_values, non_explicit_values))


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
@click.argument('save-path', type=str)
@click.option('--cache-path', default='', type=str)
@click.option('--simple-connectives', is_flag=True)
@click.option('-s', '--sense-lvl', default=2, type=int)
@click.option('--eval-only', is_flag=True)
@click.option('-f', '--output-format', default='text', type=click.Choice(['text', 'latex'], case_sensitive=False))
@click.option('--conn-length', default=1, type=int)
@click.option('--conn-hidden-dim', default=256, type=int)
@click.option('--expl-length', default=100, type=int)
@click.option('--args-rnn-dim', default=256, type=int)
@click.option('--args-hidden-dim', default=128, type=int)
@click.option('--impl-length', default=70, type=int)
@click.option('--impl-rnn-dim', default=128, type=int)
@click.option('--impl-hidden-dim', default=64, type=int)
def main(bert_model, conll_path, save_path, cache_path, simple_connectives, sense_lvl, eval_only, output_format,
         conn_length, conn_hidden_dim, expl_length, args_rnn_dim, args_hidden_dim,
         impl_length, impl_rnn_dim, impl_hidden_dim):
    cache_path = cache_path if len(cache_path) else conll_path
    os.makedirs(save_path, exist_ok=True)
    logger = init_logger(path=os.path.join(save_path, 'pipeline.log'))
    logger.info('Load data')

    docs = load_docs(conll_path)
    docs = load_bert_embeddings(docs, cache_dir=cache_path,
                                bert_model=bert_model)

    docs_train, docs_test = split_train_test(docs)
    docs_train, docs_val = split_train_test(docs_train)

    logger.info('Init model')
    parser = ParserPipeline([
        ConnectiveSenseClassifier(input_dim=docs_val[0].get_embedding_dim(), used_context=conn_length,
                                  hidden_dim=conn_hidden_dim),
        ConnectiveArgumentExtractor(window_length=expl_length, input_dim=docs_val[0].get_embedding_dim(),
                                    hidden_dim=args_hidden_dim, rnn_dim=args_rnn_dim),
        ImplicitArgumentExtractor(window_length=impl_length, input_dim=docs_val[0].get_embedding_dim(),
                                  hidden_dim=args_hidden_dim, rnn_dim=args_rnn_dim),
        ArgumentSenseClassifier(input_dim=docs_val[0].get_embedding_dim(), arg_length=int(impl_length / 2),
                                hidden_dim=impl_hidden_dim, rnn_dim=impl_rnn_dim),
    ])
    if not eval_only:
        logger.info('Load train data')
        logger.info('Train model')
        parser.fit(docs_train, docs_val)
        parser.save(save_path)
    logger.info('LOAD model')
    parser.load(save_path)
    for title, docs_eval in [('TEST', docs_test), ]:
        logger.info(f'Evaluate parser {title}')
        time_start = time.time()
        test_preds = parser.parse(docs_eval)
        logger.info(f'{title} absolute parse time: {(time.time() - time_start)}')
        logger.info(f'{title} avg parse time: {(time.time() - time_start) / len(docs_eval)}s')
        for threshold in [0.7, 0.95]:
            res_explicit = evaluate_docs(
                [d.with_relations(d.get_explicit_relations()) for d in docs_eval],
                [d.with_relations(d.get_explicit_relations()) for d in test_preds],
                threshold=threshold)
            res_non_explicit = evaluate_docs(
                [d.with_relations([r for r in d.relations if not r.is_explicit() and r.type != 'AltLex']) for d in
                 docs_eval],
                [d.with_relations([r for r in d.relations if not r.is_explicit() and r.type != 'AltLex']) for d in
                 test_preds],
                threshold=threshold)
            if output_format == 'latex':
                model_name = os.path.basename(save_path)
                print_results_latex(res_explicit, res_non_explicit, title=f'{model_name}-{title}-{threshold}')
            else:
                print_results(res_explicit, title=f'{title}-EXPLICIT-{threshold}')
                print_results(res_non_explicit, title=f'{title}-NON-EXPLICIT-{threshold}')


if __name__ == "__main__":
    main()
