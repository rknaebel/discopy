import logging
import os
import pickle
import sys

from tensorflow.keras.callbacks import Callback

from discopy.data.conll16 import get_conll_dataset

logger = logging.getLogger('discopy')

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report

from discopy.labeling.neural.bilstm import BiLSTMx, get_balanced_class_weights, logger, load_embedding_matrix, \
    BertBiLSTMx
from discopy.labeling.neural.utils import get_vocab, generate_pdtb_features, extract_windows, \
    predict_discourse_windows_for_id, reduce_relation_predictions, generate_pdtb_features_bert, extract_windows_bert
from discopy.utils import init_logger


class SkMetrics(Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.concatenate(self.model.predict(self.x_val).argmax(-1))
        y = np.concatenate(self.y_val.argmax(-1))
        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report EPOCH {}".format(epoch))
        for line in report.split('\n'):
            logger.info(line)


class AbstractArgumentExtractor:

    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, embedding_input_dim=0):
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.no_rnn = no_rnn
        self.no_dense = no_dense
        self.no_crf = no_crf
        self.positives_only = False
        self.explicits_only = False

        self.path = ''
        self.model = None
        self.embd = None
        self.vocab = {}
        self.callbacks = []

        # whether to use pre embedded tokens (such as bert encodings)
        self.embedding_input_dim = embedding_input_dim

    def load(self, path):
        self.vocab = pickle.load(open(os.path.join(path, self.path + '.vocab'), 'rb'))
        self.init_model()
        self.model.model.load_weights(os.path.join(path, self.path))

    def save(self, path):
        self.model.model.save_weights(os.path.join(path, self.path))
        pickle.dump(self.vocab, open(os.path.join(path, self.path + '.vocab'), 'wb'))

    def init_model(self):
        if self.embedding_input_dim == 0:
            embeddings = load_embedding_matrix('/data/word_vectors/glove.6B.300d.txt', self.vocab, 300)
            self.embd = Embedding(len(self.vocab), 300,
                                  weights=[embeddings],
                                  input_length=self.window_length,
                                  trainable=False)
            self.model = BiLSTMx(self.embd, self.window_length, self.hidden_dim, self.rnn_dim,
                                 self.no_rnn, self.no_dense, 4)
        else:
            self.model = BertBiLSTMx(768, self.window_length, self.hidden_dim, self.rnn_dim,
                                     self.no_rnn, self.no_dense, 4)

    def fit(self, pdtb, parses, pdtb_val, parses_val, epochs=25, save_path='', init_model=True):
        if init_model:
            self.vocab = get_vocab(parses)
            self.init_model()
        if self.embedding_input_dim == 0:
            X_train, y_train = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length,
                                                      explicits_only=self.explicits_only,
                                                      positives_only=self.positives_only)
            X_val, y_val = generate_pdtb_features(pdtb_val, parses_val, self.vocab, self.window_length,
                                                  explicits_only=self.explicits_only,
                                                  positives_only=self.positives_only)
        else:
            print('generate bert training data')
            X_train, y_train = generate_pdtb_features_bert(pdtb, parses, self.window_length,
                                                           explicits_only=self.explicits_only,
                                                           positives_only=self.positives_only)
            print('generate bert val data')
            X_val, y_val = generate_pdtb_features_bert(pdtb_val, parses_val, self.window_length,
                                                       explicits_only=self.explicits_only,
                                                       positives_only=self.positives_only)
            print('train data', X_train.shape, y_train.shape)
            print('val data', X_val.shape, y_val.shape)

        if init_model:
            def scheduler(epoch):
                if epoch < 10:
                    return 0.001
                else:
                    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

            class_weights = get_balanced_class_weights(y_train)
            logger.info('Class weights: {}'.format(class_weights))
            self.model.compile(class_weights)
            self.model.summary()
            self.callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=2, min_lr=0.00001, verbose=2),
                # tf.keras.callbacks.LearningRateScheduler(scheduler),
                EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
                SkMetrics(X_val, y_val),
            ]
            if save_path:
                self.callbacks.append(ModelCheckpoint(os.path.join(save_path, self.path + '.ckp'), save_best_only=True,
                                                      save_weights_only=True))
                self.callbacks.append(CSVLogger(os.path.join(save_path, 'logs.csv')))
        self.model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=512, callbacks=self.callbacks)

    def fit_noisy(self, pdtb, parses, pdtb_val, parses_val, pdtb_noisy, parses_noisy, epochs=25, save_path='',
                  init_model=True):
        if init_model:
            self.vocab = get_vocab(parses)
            self.init_model()

        X_train, y_train = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length,
                                                  explicits_only=self.explicits_only,
                                                  positives_only=self.positives_only)
        X_val, y_val = generate_pdtb_features(pdtb_val, parses_val, self.vocab, self.window_length,
                                              explicits_only=self.explicits_only,
                                              positives_only=self.positives_only)
        X_noisy, y_noisy = generate_pdtb_features(pdtb_noisy, parses_noisy, self.vocab, self.window_length,
                                                  explicits_only=self.explicits_only,
                                                  positives_only=True)
        X_train = np.concatenate([X_train, X_noisy])
        y_train = np.concatenate([y_train, y_noisy])

        if init_model:
            def scheduler(epoch):
                if epoch < 10:
                    return 0.001
                else:
                    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

            class_weights = get_balanced_class_weights(y_train)
            logger.info('Class weights: {}'.format(class_weights))
            self.model.compile(class_weights)
            self.model.summary()
            self.callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=2, min_lr=0.00001, verbose=2),
                # tf.keras.callbacks.LearningRateScheduler(scheduler),
                EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
                SkMetrics(X_val, y_val),
            ]
            if save_path:
                self.callbacks.append(ModelCheckpoint(os.path.join(save_path, self.path + '.ckp'), save_best_only=True,
                                                      save_weights_only=True))
                self.callbacks.append(CSVLogger(os.path.join(save_path, 'logs.csv')))
        self.model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=256, callbacks=self.callbacks)

    def score_on_features(self, X, y):
        y_pred = np.concatenate(self.model.predict(X).argmax(-1))
        y = np.concatenate(y.argmax(-1))
        logger.info("Evaluation: {}".format(self.path))
        # logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        # prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        # logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        # logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

        report = classification_report(y, y_pred,
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length,
                                      self.explicits_only, self.positives_only)
        self.score_on_features(X, y)


class ArgumentExtractBiLSTM(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, explicits_only=False):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf)
        self.explicits_only = explicits_only
        self.path = 'bilstm_all_{}_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
            int(self.explicits_only)
        )

    def extract_arguments(self, words, strides=1, max_distance=0.5):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        windows = extract_windows(tokens, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=512)
        # print(y_hat.shape)
        # print(y_hat)
        # print(y_hat.argmax(-1))
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


class BertArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, explicits_only=False):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, embedding_input_dim=768)
        self.explicits_only = explicits_only
        self.path = 'bilstm_all_{}_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
            int(self.explicits_only)
        )

    def extract_arguments(self, tokens, strides=1, max_distance=0.5):
        offset = self.window_length // 2
        windows = extract_windows_bert(tokens, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=256)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


class BiLSTMConnectiveArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf)
        self.path = 'bilstm_arg_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
        )
        self.positives_only = True
        self.explicits_only = True

    def extract_arguments(self, words, relations):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=512)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        relations = [relations_hat[i] for i in conn_pos]
        return relations

    def get_window_probs(self, words):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=512)
        return y_hat

    def get_relations_for_window_probs(self, y_hat, words, relations):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        probs = y_hat.max(-1).mean(-1)[conn_pos]
        relations = [relations_hat[i] for i in conn_pos]
        return relations, probs


class BertConnectiveArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, embedding_input_dim=768)
        self.path = 'bert_arg_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
        )
        self.positives_only = True
        self.explicits_only = True

    def extract_arguments(self, tokens, relations):
        offset = self.window_length // 2
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        word_windows = extract_windows_bert(tokens, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=256)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        relations = [relations_hat[i] for i in conn_pos]
        return relations

    def get_window_probs(self, tokens):
        offset = self.window_length // 2
        word_windows = extract_windows_bert(tokens, self.window_length, 1, offset)
        y_hat = self.model.predict(word_windows, batch_size=256)
        return y_hat

    def get_relations_for_window_probs(self, y_hat, tokens, relations):
        offset = self.window_length // 2
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, 1, offset)
        probs = y_hat.max(-1).mean(-1)[conn_pos]
        relations = [relations_hat[i] for i in conn_pos]
        return relations, probs


if __name__ == "__main__":
    logger = init_logger(path='info.log')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    data_path = sys.argv[1]
    parses_train, pdtb_train = get_conll_dataset(data_path, 'en.train', load_trees=False, connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset(data_path, 'en.dev', load_trees=False, connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset(data_path, 'en.test', load_trees=False, connective_mapping=True)

    window_length = 100
    rnn_dim = 128
    hidden_dim = 64
    epochs = 10
    path = '../bilstm-weighted'

    clf = BiLSTMConnectiveArgumentExtractor(window_length=window_length, hidden_dim=hidden_dim, rnn_dim=rnn_dim,
                                            no_rnn=False, no_dense=False, no_crf=True)
    logger.info('Train Dense model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=epochs)
    # clf.save(path)
    # logger.info('Evaluation on VAL')
    # clf.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)

    clf = ArgumentExtractBiLSTM(window_length=window_length, hidden_dim=hidden_dim, rnn_dim=rnn_dim,
                                no_rnn=False, no_dense=False, no_crf=True, explicits_only=True)
    logger.info('Train Dense model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=epochs)
    # clf.save(path)
    # logger.info('Evaluation on VAL')
    # clf.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)

    clf = ArgumentExtractBiLSTM(window_length=window_length, hidden_dim=hidden_dim, rnn_dim=rnn_dim,
                                no_rnn=False, no_dense=False, no_crf=True, explicits_only=False)
    logger.info('Train Dense model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=epochs)
    # clf.save(path)
    # logger.info('Evaluation on VAL')
    # clf.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)
