import logging
import os
from typing import List

import click

from discopy.components.argument.bert.abstract import AbstractArgumentExtractor
from discopy.components.nn.windows import predict_discourse_windows_for_id, reduce_relation_predictions, extract_windows
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_bert_conll_dataset
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')


class ExplicitArgumentExtractor(AbstractArgumentExtractor):
    model_name = 'neural_explicit_extract'

    def __init__(self, window_length, input_dim, hidden_dim, rnn_dim):
        super().__init__(window_length, input_dim, hidden_dim, rnn_dim, nb_classes=4, explicits_only=True,
                         positives_only=False)

    @staticmethod
    def from_config(config: dict):
        return ExplicitArgumentExtractor(window_length=config['window_length'], input_dim=config['input_dim'],
                                         hidden_dim=config['hidden_dim'], rnn_dim=config['rnn_dim'])

    def parse(self, doc: Document, relations: List[Relation] = None,
              batch_size=64, strides=1, max_distance=0.5, **kwargs):
        offset = self.window_length // 2
        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()
        windows = extract_windows(doc_bert, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=batch_size)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
def main(bert_model, conll_path):
    logger = init_logger()
    logger.info('Load dev data')
    docs_val = load_bert_conll_dataset(os.path.join(conll_path, 'en.dev'),
                                       simple_connectives=True,
                                       cache_dir=os.path.join(conll_path, f'en.dev.{bert_model}.joblib'),
                                       bert_model=bert_model)
    logger.info('Init model')
    clf = ExplicitArgumentExtractor(window_length=100, input_dim=docs_val[0].get_embedding_dim(), hidden_dim=256,
                                    rnn_dim=512)
    logger.info('Load train data')
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         simple_connectives=True,
                                         cache_dir=os.path.join(conll_path, f'en.train.{bert_model}.joblib'),
                                         bert_model=bert_model)
    logger.info('Train model')
    clf.fit(docs_train, docs_val)
    clf.save('models/nn')
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    print(clf.parse(docs_val[0], docs_val[0].get_explicit_relations()))


if __name__ == "__main__":
    main()
