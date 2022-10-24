import json
import logging
import os
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from discopy.components.component import Component
from discopy.components.nn.windows import PDTBWindowSequence
from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation

logger = logging.getLogger('discopy')


class SkMetrics(tf.keras.callbacks.Callback):
    def __init__(self, ds, targets, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.ds = ds
        self.targets = targets

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.concatenate([self.model.predict(windows).argmax(-1) for windows, args in self.ds]).flatten()
        y = np.concatenate([args.argmax(-1) for windows, args in self.ds]).flatten()
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=self.targets, labels=range(len(self.targets)),
                                       digits=4)
        logger.info("Classification Report EPOCH {}".format(epoch))
        for line in report.split('\n'):
            logger.info(line)


def get_model(max_seq_len, hidden_dim, rnn_dim, nb_classes, input_size):
    x = y = tf.keras.layers.Input(shape=(max_seq_len, input_size), name='window-input')
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True), name='rnn')(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim, return_sequences=True), name='rnn2')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(hidden_dim, activation='relu', name='dense')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(nb_classes, activation='softmax', name='args')(y)
    model = tf.keras.models.Model(x, y)
    return model


class AbstractArgumentExtractor(Component):
    used_features = ['vectors']

    def __init__(self, window_length, input_dim, hidden_dim, rnn_dim, nb_classes, explicits_only=False,
                 positives_only=False, ckpt_path: str = ''):
        super().__init__()
        self.window_length = window_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.nb_classes = nb_classes
        self.targets = ['None', 'Arg1', 'Arg2', 'Conn'][:self.nb_classes]
        self.explicits_only = explicits_only
        self.positives_only = positives_only
        self.checkpoint_path = ckpt_path
        if ckpt_path and not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        self.model = get_model(self.window_length, self.hidden_dim, self.rnn_dim, nb_classes, input_dim)
        self.compiled = False
        self.sense_map = {}
        self.callbacks = []
        self.epochs = 50
        self.batch_size = 512
        self.metrics = [
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="ROC"),
        ]
        self.optimizer = tf.keras.optimizers.Adam(amsgrad=True)

    def get_config(self):
        return {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'window_length': self.window_length,
            'hidden_dim': self.hidden_dim,
            'rnn_dim': self.rnn_dim,
            'nb_classes': self.nb_classes,
            'explicits_only': self.explicits_only,
            'positives_only': self.positives_only,
            'checkpoint_path': self.checkpoint_path,
            'sense_map': self.sense_map,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }

    @staticmethod
    def from_config(config: dict):
        raise NotImplementedError()

    def get_loss(self, class_weights):
        def loss(onehot_labels, logits):
            c_weights = np.array([class_weights[i] for i in range(self.nb_classes)])
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
            weights = tf.reduce_sum(tf.multiply(tf.cast(onehot_labels, tf.float32), c_weights), axis=-1)
            weighted_losses = unweighted_losses * weights  # reduce the result to get your final loss
            total_loss = tf.reduce_mean(weighted_losses)
            return total_loss

        return loss

    def load(self, path):
        if not os.path.exists(os.path.join(path, self.model_name)):
            raise FileNotFoundError("Model not found.")
        self.sense_map = json.load(open(os.path.join(path, self.model_name, 'senses.json'), 'r'))
        self.model = tf.keras.models.load_model(os.path.join(path, self.model_name), compile=False)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, self.model_name))
        json.dump(self.sense_map, open(os.path.join(path, self.model_name, 'senses.json'), 'w'))

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        self.sense_map = {v: k for k, v in enumerate(
            ['NoSense'] + sorted({s for doc in docs_train for rel in doc.relations for s in rel.senses}))}
        ds_train = PDTBWindowSequence(docs_train, self.window_length, self.sense_map, batch_size=self.batch_size,
                                      nb_classes=self.nb_classes,
                                      explicits_only=self.explicits_only,
                                      positives_only=self.positives_only)
        ds_val = PDTBWindowSequence(docs_val, self.window_length, self.sense_map, batch_size=self.batch_size,
                                    nb_classes=self.nb_classes,
                                    explicits_only=self.explicits_only,
                                    positives_only=self.positives_only,
                                    use_shuffle=False)
        logging.info(f"Training samples: {len(ds_train.instances)} (Docs={len(ds_train.docs)})")
        logging.info(f"class weight balance (train) {ds_train.get_balanced_class_weights()}")
        logging.info(f"Validation samples: {len(ds_val.instances)} (Docs={len(ds_val.docs)})")
        logging.info(f"class weight balance (val) {ds_val.get_balanced_class_weights()}")
        if not self.compiled:
            self.model.compile(loss=self.get_loss(ds_train.get_balanced_class_weights()),
                               optimizer=self.optimizer,
                               metrics=self.metrics)
            self.compiled = True
        self.model.summary()
        self.callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2, min_lr=0.00001,
                                                 verbose=2),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True,
                                             verbose=1),
            # TODO fix metrics computation - gpu memory problems
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
            verbose=2,
            validation_data=ds_val,
            epochs=self.epochs,
            callbacks=self.callbacks,
            max_queue_size=100,
        )

    def score(self, docs: List[Document]):
        ds = PDTBWindowSequence(docs, self.window_length, self.sense_map, batch_size=self.batch_size,
                                nb_classes=self.nb_classes,
                                explicits_only=self.explicits_only,
                                positives_only=self.positives_only)
        y_pred = np.concatenate([self.model.predict(windows).argmax(-1) for windows, args in ds]).flatten()
        y = np.concatenate([args.argmax(-1) for windows, args in ds]).flatten()
        logger.info("Evaluation: {}".format(self.model_name))
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=self.targets, labels=range(len(self.targets)),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)

    def parse(self, doc: Document, relations: List[Relation] = None, **kwargs):
        raise NotImplementedError()
