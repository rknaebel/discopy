import logging
import os
from typing import List

import click
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from discopy.components.argument.bert.abstract import AbstractArgumentExtractor
from discopy.components.nn.windows import predict_discourse_windows_for_id, reduce_relation_predictions, \
    extract_windows, PDTBWindowSequence
from discopy.evaluate.conll import evaluate_docs, print_results
from discopy.utils import init_logger
from discopy_data.data.doc import Document
from discopy_data.data.loaders.conll import load_bert_conll_dataset
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')


class ImplicitArgumentExtractor(AbstractArgumentExtractor):
    model_name = 'neural_implicit_extract'

    def __init__(self, window_length, input_dim, hidden_dim, rnn_dim):
        super().__init__(window_length, input_dim, hidden_dim, rnn_dim, nb_classes=3)

    @staticmethod
    def from_config(config: dict):
        return ImplicitArgumentExtractor(window_length=config['window_length'], input_dim=config['input_dim'],
                                         hidden_dim=config['hidden_dim'], rnn_dim=config['rnn_dim'])

    @staticmethod
    def get_non_explicit_documents(docs):
        return [doc.with_relations([r for r in doc.relations if r.type in {'Implicit', 'EntRel'}]) for doc in docs]

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        docs_train = self.get_non_explicit_documents(docs_train)
        docs_val = self.get_non_explicit_documents(docs_val)
        self.sense_map = {v: k for k, v in enumerate(
            ['NoSense'] + sorted({s for doc in docs_train for rel in doc.relations for s in rel.senses}))}
        ds_train = PDTBWindowSequence(docs_train, self.window_length, self.sense_map, batch_size=self.batch_size,
                                      nb_classes=self.nb_classes,
                                      explicits_only=self.explicits_only,
                                      positives_only=self.positives_only)
        ds_val = PDTBWindowSequence(docs_val, self.window_length, self.sense_map, batch_size=self.batch_size,
                                    nb_classes=self.nb_classes,
                                    explicits_only=self.explicits_only,
                                    positives_only=self.positives_only)
        logging.info(f"class weight balance (train) {ds_train.get_balanced_class_weights()}")
        logging.info(f"class weight balance (val) {ds_val.get_balanced_class_weights()}")
        if not self.compiled:
            self.model.compile(loss=self.get_loss(ds_train.get_balanced_class_weights()),
                               optimizer=self.optimizer,
                               metrics=self.metrics)
            self.compiled = True
        self.model.summary()
        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=0.00001,
                                                 verbose=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, min_delta=0.001, restore_best_weights=True,
                                             verbose=1),
            # SkMetrics(ds_val, self.targets),
        ]
        if self.checkpoint_path:
            os.makedirs(os.path.join(self.checkpoint_path, self.model_name), exist_ok=True)
            self.callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path, self.model_name, 'model.ckp'),
                                                   save_best_only=True,
                                                   save_weights_only=True))
            self.callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(self.checkpoint_path, self.model_name,
                                                                            'logs.csv')))
        self.model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=self.epochs,
            callbacks=self.callbacks,
            max_queue_size=5,
            workers=2
        )

    def score(self, docs: List[Document]):
        docs = self.get_non_explicit_documents(docs)
        ds = PDTBWindowSequence(docs, self.window_length, self.sense_map, batch_size=self.batch_size,
                                nb_classes=self.nb_classes,
                                explicits_only=self.explicits_only,
                                positives_only=self.positives_only)
        y = [(windows, args) for windows, args in ds]
        windows = np.concatenate([windows for windows, args in y], axis=0)
        args = np.concatenate([args for windows, args in y], axis=0)
        y_pred = np.concatenate(self.model.predict(windows, batch_size=self.batch_size).argmax(-1))
        y = np.concatenate(args.argmax(-1))
        logger.info("Evaluation: {}".format(self.model_name))
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=self.targets, labels=range(len(self.targets)),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)

    def parse(self, doc: Document, relations: List[Relation] = None,
              batch_size=128, strides=1, max_distance=0.25, **kwargs):
        if not relations:
            relations = []
        offset = self.window_length // 2
        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()
        windows = extract_windows(doc_bert, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=batch_size)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        for relation in relations_hat:
            relation.type = "Implicit"
        relations.extend(relations_hat)
        return relations


@click.command()
@click.argument('bert-model', type=str)
@click.argument('conll-path', type=str)
@click.option('--simple-connectives', is_flag=True)
@click.option('-s', '--sense-lvl', default=2, type=int)
def main(bert_model, conll_path, simple_connectives, sense_lvl):
    logger = init_logger()
    logger.info('Load data')
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
    docs_train = load_bert_conll_dataset(os.path.join(conll_path, 'en.train'),
                                         simple_connectives=simple_connectives,
                                         cache_dir=os.path.join(conll_path, f'en.train.{bert_model}.joblib'),
                                         bert_model=bert_model,
                                         sense_level=sense_lvl)
    logger.info('Init model')
    clf = ImplicitArgumentExtractor(window_length=100, input_dim=docs_val[0].get_embedding_dim(), hidden_dim=128,
                                    rnn_dim=256)
    clf.batch_size = 256
    logger.info('Train model')
    clf.fit(docs_train, docs_val)
    clf.save('models/nn')
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    for title, docs_eval in [('TEST', docs_test), ('BLIND', docs_blind)]:
        logger.info(f'Evaluate parser {title}')
        preds = [d.with_relations(clf.parse(d)) for d in docs_eval]
        for threshold in [0.7, 0.95]:
            res = evaluate_docs(
                [d.with_relations([r for r in d.relations if r.type in ['Implicit', 'EntRel']]) for d in docs_eval],
                preds,
                threshold=threshold)
            print_results(res, title=f'{title}-{threshold}')


if __name__ == "__main__":
    main()
