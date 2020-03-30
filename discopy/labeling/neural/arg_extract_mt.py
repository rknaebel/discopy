import logging
import os
import pickle
import sys

import h5py
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

from discopy.data.conll16 import get_conll_dataset

logger = logging.getLogger('discopy')

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report

from discopy.labeling.neural.bilstm_mt import BiLSTMx, get_balanced_class_weights, logger, load_embedding_matrix, \
    BertBiLSTMx
from discopy.labeling.neural.utils import get_vocab, generate_pdtb_features, predict_discourse_windows_for_id, \
    reduce_relation_predictions, extract_windows
from discopy.utils import init_logger


class SkMetrics(Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y1_val = y_val[0]
        self.y2_val = y_val[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_val)
        y1_pred, y2_pred = y_pred
        y1 = self.y1_val.argmax(-1).flatten()
        y2 = self.y2_val.argmax(-1).flatten()
        logger.info("Evaluation Epoch: {}".format(epoch))
        report = classification_report(y1, y1_pred.argmax(-1).flatten(),
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)
        report = classification_report(y2, y2_pred.argmax(-1).flatten(),
                                       output_dict=False,
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)


def extract_inputs(parses, vocab, is_bert=False):
    inputs = {}
    for doc_id, doc in tqdm(parses.items()):
        inputs[doc_id] = {'sentences': []}
        for sent in doc['sentences']:
            if is_bert:
                inputs[doc_id]['sentences'].append({
                    'tokens': sent['bert']
                })
            else:
                inputs[doc_id]['sentences'].append({
                    'tokens': np.array([vocab.get(word[0].lower(), 1) for word in sent['words']])
                })
    return inputs


def extract_outputs(pdtb, sense_map, sense_level=2):
    return [{
        'Arg1': {t[2] for t in r['Arg1']['TokenList']},
        'Arg2': {t[2] for t in r['Arg2']['TokenList']},
        'Conn': {t[2] for t in r['Connective']['TokenList']},
        'Sense': sense_map.get('.'.join(sense.split('.')[:sense_level]), 0),
        'Type': r['Type'],
        'DocID': r['DocID']
    } for r in pdtb if r['Type'] in ('Explicit', 'Implicit') for sense in r['Sense']]


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
        self.arch = ''
        self.model = None
        self.embd = None
        self.vocab = {}
        self.sense_map = {}
        self.callbacks = []

        # whether to use pre embedded tokens (such as bert encodings)
        self.use_indices = embedding_input_dim == 0

    def load(self, path):
        self.vocab = pickle.load(open(os.path.join(path, self.path + '.vocab'), 'rb'))
        self.sense_map = pickle.load(open(os.path.join(path, self.path + '.senses'), 'rb'))
        self.init_model()
        self.model.model.load_weights(os.path.join(path, self.path))

    def save(self, path):
        self.model.model.save_weights(os.path.join(path, self.path))
        pickle.dump(self.vocab, open(os.path.join(path, self.path + '.vocab'), 'wb'))
        pickle.dump(self.sense_map, open(os.path.join(path, self.path + '.senses'), 'wb'))

    def init_model(self):
        if self.arch == 'bilstm':
            embeddings = load_embedding_matrix('/data/word_vectors/glove.6B.300d.txt', self.vocab, 300)
            self.embd = Embedding(len(self.vocab), 300,
                                  weights=[embeddings],
                                  input_length=self.window_length,
                                  trainable=False)
            self.model = BiLSTMx(self.embd, self.window_length, self.hidden_dim, self.rnn_dim,
                                 self.no_rnn, self.no_dense, (4, max(self.sense_map.values()) + 1))
        elif self.arch == 'bert':
            self.model = BertBiLSTMx(768, self.window_length, self.hidden_dim, self.rnn_dim,
                                     self.no_rnn, self.no_dense, (4, max(self.sense_map.values()) + 1))

    def fit_on_features(self, x_train, y_train, x_val, y_val, epochs=25, save_path='', init_model=True):
        if init_model:
            def scheduler(epoch):
                if epoch < 10:
                    return 0.001
                else:
                    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

            class_weights = get_balanced_class_weights(y_train[0][()])
            logger.info('Class weights: {}'.format(class_weights))
            self.model.compile(class_weights)
            self.model.summary()
            self.callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=2, min_lr=0.00001, verbose=2),
                EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
                SkMetrics(x_val, y_val),
            ]
            if save_path:
                self.callbacks.append(ModelCheckpoint(os.path.join(save_path, self.path + '.ckp'), save_best_only=True,
                                                      save_weights_only=True))
                self.callbacks.append(CSVLogger(os.path.join(save_path, 'logs.csv')))
        self.model.fit(x_train, y_train, x_val, y_val, epochs=epochs, batch_size=512, callbacks=self.callbacks)

    def fit(self, pdtb, parses, pdtb_val, parses_val, epochs=25, save_path='', init_model=True, sense_level=2):
        if init_model:
            self.vocab = get_vocab(parses)
            self.sense_map = dict((v, k + 1) for k, v in enumerate(
                np.unique(['.'.join(sense.split('.')[:sense_level]) for r in pdtb for sense in r['Sense']])
            ))
            self.init_model()
        data_file_path = '/cache/{}_{}_{}_{}.hdf5'.format(
            'idx' if self.use_indices else 'bert',
            self.window_length,
            self.explicits_only,
            self.positives_only
        )
        if not os.path.exists(data_file_path):
            logger.warning('File not found, generate data: ' + data_file_path)
            with h5py.File(data_file_path, 'w') as f:
                parses = extract_inputs(parses, self.vocab, is_bert=(not self.use_indices))
                pdtb = extract_outputs(pdtb, self.sense_map)
                x_train, y1_train, y2_train = generate_pdtb_features(pdtb, parses, self.window_length,
                                                                     explicits_only=self.explicits_only,
                                                                     positives_only=self.positives_only)
                x_train = f.create_dataset('x_train', data=x_train)
                y1_train = f.create_dataset('y1_train', data=y1_train)
                y2_train = f.create_dataset('y2_train', data=y2_train)
                parses_val = extract_inputs(parses_val, self.vocab, is_bert=(not self.use_indices))
                pdtb_val = extract_outputs(pdtb_val, self.sense_map)
                x_val, y1_val, y2_val = generate_pdtb_features(pdtb_val, parses_val, self.window_length,
                                                               explicits_only=self.explicits_only,
                                                               positives_only=self.positives_only)
                x_val = f.create_dataset('x_val', data=x_val)
                y1_val = f.create_dataset('y1_val', data=y1_val)
                y2_val = f.create_dataset('y2_val', data=y2_val)
        f = h5py.File(data_file_path, 'r')
        x_train = f['x_train']
        y1_train = f['y1_train']
        y2_train = f['y2_train']
        x_val = f['x_val']
        y1_val = f['y1_val']
        y2_val = f['y2_val']
        self.fit_on_features(x_train, [y1_train, y2_train], x_val[()], [y1_val[()], y2_val[()]], epochs, save_path,
                             init_model)

    def fit_noisy(self, pdtb, parses, pdtb_val, parses_val, pdtb_noisy, parses_noisy, epochs=25, save_path='',
                  init_model=True, sense_level=2):
        if init_model:
            self.vocab = get_vocab(parses)
            self.sense_map = dict((v, k + 1) for k, v in enumerate(
                np.unique(['.'.join(sense.split('.')[:sense_level]) for r in pdtb for sense in r['Sense']])
            ))
            self.init_model()
        parses = extract_inputs(parses, self.vocab, is_bert=(not self.use_indices))
        pdtb = extract_outputs(pdtb, self.sense_map)
        x_train, y1_train, y2_train = generate_pdtb_features(pdtb, parses, self.window_length,
                                                             explicits_only=self.explicits_only,
                                                             positives_only=self.positives_only)
        parses_val = extract_inputs(parses_val, self.vocab, is_bert=(not self.use_indices))
        pdtb_val = extract_outputs(pdtb_val, self.sense_map)
        x_val, y1_val, y2_val = generate_pdtb_features(pdtb_val, parses_val, self.window_length,
                                                       explicits_only=self.explicits_only,
                                                       positives_only=self.positives_only)
        parses_noisy = extract_inputs(parses_noisy, self.vocab, is_bert=(not self.use_indices))
        pdtb_noisy = extract_outputs(pdtb_noisy, self.sense_map)
        x_noisy, y1_noisy, y2_noisy = generate_pdtb_features(pdtb_noisy, parses_noisy, self.window_length,
                                                             explicits_only=self.explicits_only,
                                                             positives_only=True)
        x_train = np.concatenate([x_train, x_noisy])
        y1_train = np.concatenate([y1_train, y1_noisy])
        y2_train = np.concatenate([y2_train, y2_noisy])

        self.fit_on_features(x_train, [y1_train, y2_train], x_val, [y1_val, y2_val], epochs, save_path, init_model)

    def score_on_features(self, X, y1, y2):
        y_pred = self.model.predict(X)
        # print(y_pred)
        # print(y1, y2)
        y1_pred, y2_pred = y_pred
        y1 = y1.argmax(-1).flatten()
        y2 = y2.argmax(-1).flatten()
        # print(y1.shape, y2.shape, y1_pred.shape, y2_pred.shape)
        logger.info("Evaluation: {}".format(self.path))
        # logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        # prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        # logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        # logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

        report = classification_report(y1, y1_pred.argmax(-1).flatten(),
                                       output_dict=False,
                                       target_names=['None', 'Arg1', 'Arg2', 'Conn'], labels=range(4),
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)
        report = classification_report(y2, y2_pred.argmax(-1).flatten(),
                                       output_dict=False,
                                       digits=4)
        logger.info("Classification Report")
        for line in report.split('\n'):
            logger.info(line)

    def score(self, pdtb, parses):
        parses = extract_inputs(parses, self.vocab, is_bert=(not self.use_indices))
        pdtb = extract_outputs(pdtb, self.sense_map)
        X, y1, y2 = generate_pdtb_features(pdtb, parses, self.window_length,
                                           self.explicits_only, self.positives_only)
        self.score_on_features(X, y1, y2)


class ArgumentExtractBiLSTM(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, explicits_only=False):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf)
        self.explicits_only = explicits_only
        self.path = 'bilstm_all_{}_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
            int(self.explicits_only)
        )
        self.arch = 'bilstm'

    def extract_arguments(self, words, strides=1, max_distance=0.5):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        windows = extract_windows(tokens, self.window_length, strides, offset)
        y1_pred, y2_pred = self.model.predict(windows, batch_size=512)
        relations_hat = predict_discourse_windows_for_id(tokens, y1_pred, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


class BertArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, explicits_only=False):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, embedding_input_dim=768)
        self.explicits_only = explicits_only
        self.path = 'bert_all_{}_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
            int(self.explicits_only)
        )
        self.arch = 'bert'

    def extract_arguments(self, tokens, strides=1, max_distance=0.5):
        offset = self.window_length // 2
        windows = extract_windows(tokens, self.window_length, strides, offset)
        y1_pred, y2_pred = self.model.predict(windows, batch_size=256)
        relations_hat = predict_discourse_windows_for_id(tokens, y1_pred, strides, offset)
        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)
        return relations_hat


class BiLSTMConnectiveArgumentExtractor(AbstractArgumentExtractor):
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf):
        super().__init__(window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf)
        self.path = 'bilstm_arg_{}_extract.h5'.format(
            'dense' if self.no_crf else 'crf',
        )
        self.arch = 'bilstm'
        self.positives_only = True
        self.explicits_only = True

    def extract_arguments(self, words, relations):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
        y1_pred, y2_pred = self.model.predict(word_windows, batch_size=512)
        relations_hat = predict_discourse_windows_for_id(tokens, y1_pred, 1, offset)
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
        self.arch = 'bert'
        self.positives_only = True
        self.explicits_only = True

    def extract_arguments(self, tokens, relations):
        offset = self.window_length // 2
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
        y1_pred, y2_pred = self.model.predict(word_windows, batch_size=256)
        relations_hat = predict_discourse_windows_for_id(tokens, y1_pred, 1, offset)
        relations = [relations_hat[i] for i in conn_pos]
        return relations

    def get_window_probs(self, tokens):
        offset = self.window_length // 2
        word_windows = extract_windows(tokens, self.window_length, 1, offset)
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
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)

    clf = ArgumentExtractBiLSTM(window_length=window_length, hidden_dim=hidden_dim, rnn_dim=rnn_dim,
                                no_rnn=False, no_dense=False, no_crf=True, explicits_only=True)
    logger.info('Train Dense model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=epochs)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)

    clf = ArgumentExtractBiLSTM(window_length=window_length, hidden_dim=hidden_dim, rnn_dim=rnn_dim,
                                no_rnn=False, no_dense=False, no_crf=True, explicits_only=False)
    logger.info('Train Dense model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=epochs)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)
