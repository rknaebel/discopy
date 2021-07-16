import json

import click
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModel

from discopy.components.nn.bert import get_sentence_embeddings
from discopy.parsers.pipeline import ParserPipeline
from discopy.utils import init_logger
from discopy_data.data.doc import Document


@click.command()
@click.argument('model-path', type=str)
@click.argument('bert-model', type=str)
@click.option('-i', '--src', default='-', type=click.File('r'))
@click.option('-o', '--tgt', default='-', type=click.File('w'))
@click.option('-l', '--limit', default=0, type=int)
def main(model_path, bert_model, src, tgt, limit):
    logger = init_logger()
    logger.info('Init Parser...')
    parser = ParserPipeline.from_config(model_path)
    parser.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = TFAutoModel.from_pretrained(bert_model)
    logger.info('Load pre-trained Parser...')
    for line_i, line in tqdm(enumerate(src)):
        if limit and line_i >= limit:
            break
        doc = Document.from_json(json.loads(line))
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
