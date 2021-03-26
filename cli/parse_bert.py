import json
import os

import click
from tqdm import tqdm

from discopy.components.argument.bert.conn import ConnectiveArgumentExtractor
from discopy.components.sense.explicit.bert_conn_sense import ConnectiveSenseClassifier
from discopy.data.loaders.raw import load_json
from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
from transformers import AutoTokenizer, TFAutoModel
from discopy.components.nn.bert import get_sentence_embeddings


@click.command()
@click.argument('model-path', type=str)
@click.option('-m', '--bert-model', default='bert-base-cased', type=str)
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(model_path, bert_model, src, tgt, limit):
    logger = init_logger()
    logger.info('Init Parser...')
    configs = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    parser = ParserPipeline([
        ConnectiveSenseClassifier(input_dim=configs[0]['input_dim'], used_context=configs[0]['used_context']),
        ConnectiveArgumentExtractor(window_length=configs[1]['window_length'], input_dim=configs[1]['input_dim'],
                                    hidden_dim=configs[1]['hidden_dim'], rnn_dim=configs[1]['rnn_dim']),
    ])
    parser.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = TFAutoModel.from_pretrained(bert_model)
    logger.info('Load pre-trained Parser...')
    for line_i, line in tqdm(enumerate(src)):
        if limit and line_i > limit:
            break
        doc_json = json.loads(line)
        doc = load_json(doc_json)
        if len(doc.sentences) == 0:
            continue
        for sent_i, sent in enumerate(doc.sentences):
            sent_words = sent.tokens
            embeddings = get_sentence_embeddings(sent_words, tokenizer, model)
            doc.sentences[sent_i].embeddings = embeddings
        doc = parser(doc)
        tgt.write(json.dumps(doc.to_json()) + '\n')


if __name__ == '__main__':
    main()
