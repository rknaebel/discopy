import logging
import os
import ujson as json
from collections import Counter
from collections import defaultdict

import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score

from discopy.utils import init_logger, Relation

logger = logging.getLogger('discopy')

from keras import Input, Model
from keras.layers import Bidirectional, Dense, Dropout, Embedding
from keras.layers import CuDNNLSTM as LSTM

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_marginal_accuracy


class BiLSTMx:

    def __init__(self, embd_layer, max_seq_len, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, nb_classes):
        learn_mode = 'marginal'
        test_mode = 'marginal'
        self.no_crf = no_crf

        x = Input(shape=(max_seq_len,), name='window-input')
        y = embd_layer(x)
        # y = SpatialDropout1D(0.2)(y)
        if not no_rnn:
            y = Bidirectional(LSTM(rnn_dim, return_sequences=True, name='hidden-rnn'))(y)
        if not no_dense:
            y = Dense(hidden_dim, activation='relu', name='hidden-dense')(y)
            y = Dropout(0.2)(y)
        if no_crf:
            y1 = Dense(nb_classes, activation='softmax', name='args')(y)
        else:
            y1 = CRF(nb_classes, test_mode=test_mode, learn_mode=learn_mode, name='args')(y)

        self.x = x
        self.y_args = y1

        self.model = Model(self.x, self.y_args)

    def compile(self):
        loss = {'args': crf_loss}
        metrics = {'args': crf_marginal_accuracy}

        if self.no_crf:
            self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                               metrics=['accuracy']
                               )
        else:
            self.model.compile(optimizer="adam",
                               loss=loss,
                               metrics=metrics)

    def fit(self, x_train, y_train, x_val, y_val, epochs, batch_size, callbacks):
        self.model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       verbose=1,
                       callbacks=callbacks)

    def predict(self, X, batch_size=64):
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def summary(self, fh=None):
        if fh:
            self.model.summary(print_fn=lambda line: fh.write("{}\n".format(line)))
        else:
            self.model.summary()


def group_by_doc_id(pdtb, explicits_only=False):
    pdtb_by_doc = defaultdict(list)
    if explicits_only:
        pdtb = filter(lambda r: r['Type'] == 'Explicit', pdtb)
    for r in pdtb:
        pdtb_by_doc[r['DocID']].append(r)
    return pdtb_by_doc


def get_vocab(parses):
    vocab = Counter(w.lower() for doc in parses.values() for s in doc['sentences'] for w, wd in s['words'])
    vocab = {v: idx for idx, v in enumerate(['<PAD>', '<UKN>'] + sorted([v for v, c in vocab.items() if c > 3]))}
    return vocab


def generate_pdtb_features(pdtb, parses, vocab, window_length, explicits_only=False, positives_only=False):
    document_features = {}
    pdtb_group = group_by_doc_id(pdtb, explicits_only)
    for doc_id, doc in parses.items():
        words = [w[0] for s in doc['sentences'] for w in s['words']]
        pos = [w[1]['PartOfSpeech'] for s in doc['sentences'] for w in s['words']]
        doc_labels = []
        relations = pdtb_group[doc_id]
        for r in relations:
            r_labels = []
            arg1_idxs = {t[2] for t in r['Arg1']['TokenList']}
            arg2_idxs = {t[2] for t in r['Arg2']['TokenList']}
            conn_idxs = {t[2] for t in r['Connective']['TokenList']}
            for w_i, w in enumerate(words):
                if w_i in arg1_idxs:
                    r_labels.append(1)
                elif w_i in arg2_idxs:
                    r_labels.append(2)
                elif w_i in conn_idxs:
                    r_labels.append(3)
                else:
                    r_labels.append(0)
            doc_labels.append(np.array(r_labels))
        document_features[doc_id] = {
            'Relations': np.array(doc_labels),
            'Words': np.array(words),
            'POS': np.array(pos),
        }

    X, y = [], []
    for doc_id, features in document_features.items():
        X_doc, y_doc = extract_document_training_windows(features, vocab, size=window_length,
                                                         positives_only=positives_only)
        if len(X_doc) and len(y_doc):
            X.append(X_doc)
            y.append(y_doc)

    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def extract_document_training_windows(document, vocab, size=100, positives_only=False):
    left_side = size // 2
    right_size = size - left_side
    repeats = 5

    hashes = np.array([vocab.get(w.lower(), 1) for w in document['Words']])
    hashes = np.pad(hashes, mode='constant', pad_width=(size, size), constant_values=0)
    relations = document['Relations']
    if len(relations):
        relations = np.pad(relations, ((0, 0), (size, size)), mode='constant', constant_values=0)

        centroids = relations.argmax(1)
        pos = np.zeros((len(relations) * repeats, size, 2), dtype=int)

        for i, (relation, centroid) in enumerate(zip(relations, centroids)):
            pos[i, :, 0] = hashes[centroid - left_side:centroid + right_size]
            pos[i, :, 1] = relation[centroid - left_side:centroid + right_size]

            i += len(relations)
            pos[i, :, 0] = hashes[centroid - 1 - left_side:centroid - 1 + right_size]
            pos[i, :, 1] = relation[centroid - 1 - left_side:centroid - 1 + right_size]

            i += len(relations)
            pos[i, :, 0] = hashes[centroid + 1 - left_side:centroid + 1 + right_size]
            pos[i, :, 1] = relation[centroid + 1 - left_side:centroid + 1 + right_size]

            i += len(relations)
            pos[i, :, 0] = hashes[centroid - 2 - left_side:centroid - 2 + right_size]
            pos[i, :, 1] = relation[centroid - 2 - left_side:centroid - 2 + right_size]

            i += len(relations)
            pos[i, :, 0] = hashes[centroid + 2 - left_side:centroid + 2 + right_size]
            pos[i, :, 1] = relation[centroid + 2 - left_side:centroid + 2 + right_size]
        if not positives_only:
            centroids_mask = np.zeros_like(hashes)
            centroids_mask[:size] = 1
            centroids_mask[-size:] = 1
            for r in range(repeats - 1):
                centroids_mask[centroids - r] = 1
                centroids_mask[centroids + r] = 1

            non_centroids = np.arange(len(centroids_mask))[centroids_mask == 0]
            neg = np.zeros((len(non_centroids), size, 2), dtype=int)
            for i, centroid in enumerate(non_centroids):
                neg[i, :, 0] = hashes[centroid - left_side:centroid + right_size]

            neg = neg[np.random.choice(len(neg), len(pos))]
            data = np.concatenate([pos, neg])
        else:
            data = pos
    elif not positives_only:
        centroids_mask = np.zeros_like(hashes)
        centroids_mask[:size] = 1
        centroids_mask[-size:] = 1
        non_centroids = np.arange(len(centroids_mask))[centroids_mask == 0]
        neg = np.zeros((len(non_centroids), size, 2), dtype=int)
        for i, centroid in enumerate(non_centroids):
            neg[i, :, 0] = hashes[centroid - left_side:centroid + right_size]
        data = neg
    else:
        return [], []

    X = data[:, :, 0]
    y = np.clip(data[:, :, 1], 0, 3)
    y = (np.arange(4) == y[..., None]).astype(bool)
    return X, y


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embedding_matrix(vector_path, word_index, emb_dim):
    save_path = "{}.npy".format(vector_path)
    if os.path.exists(save_path):
        return np.load(save_path)
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(vector_path, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    filled = 0
    all_embs = np.stack(list(embeddings_index.values()))
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), (len(word_index), emb_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            filled += 1
    print('Filled {} of {} word vectors.'.format(filled, len(word_index)))
    np.save(save_path, embedding_matrix)
    return embedding_matrix


def extract_windows(tokens, window_length, strides, offset):
    nb_tokens = len(tokens)
    tokens = np.pad(tokens, (window_length, window_length), mode='constant', constant_values=0)
    windows = []
    for i in range(0, nb_tokens, strides):
        window = tokens[i + window_length - offset:i + 2 * window_length - offset]
        windows.append(window)
    windows = np.stack(windows)

    return windows


def extract_conn_windows(tokens, conns, window_length):
    print('Conns', conns)
    tokens = np.pad(tokens, (window_length, window_length), mode='constant', constant_values=0)
    windows = []
    start_idxs = []
    for i in conns:
        start_idxs.append(i - window_length // 2)
        i += window_length
        window = tokens[i - window_length // 2:i + window_length // 2]
        windows.append(window)
        print(start_idxs[-1])
        print(windows[-1])
    windows = np.stack(windows)
    print('starts', start_idxs)
    return start_idxs, windows


def extract_relation_from_window(window_pred, start_idx, nb_tokens):
    idxs = np.arange(len(window_pred)) + start_idx
    pred = window_pred.argmax(-1)
    relation = Relation()

    for p, t in zip(pred, idxs):
        if t < 0 or t >= nb_tokens:
            continue
        if p == 1:
            relation.arg1.add(t)
        if p == 2:
            relation.arg2.add(t)
        if p == 3:
            relation.conn.add(t)
    return relation


def predict_discourse_windows_for_id(tokens, windows, strides, offset, start_idxs=None):
    nb_tokens = len(tokens)
    relations_hat = []
    if start_idxs:
        for i, (w, s) in enumerate(zip(windows, start_idxs)):
            relations_hat.append(extract_relation_from_window(w, s, nb_tokens))
            print(relations_hat[-1])
    else:
        start_idx = -offset
        for i, w in enumerate(windows):
            relations_hat.append(extract_relation_from_window(w, start_idx, nb_tokens))
            start_idx += strides
    return relations_hat


def reduce_relation_predictions(relations, max_distance=0.5):
    if len(relations) == 0:
        return []
    combined = []
    current = relations[0]
    distances = [relations[i].distance(relations[i + 1]) for i in range(len(relations) - 1)]
    for i, d in enumerate(distances):
        next_rel = relations[i + 1]
        if d < max_distance:
            current = current | next_rel
        else:
            combined.append(current)
            current = next_rel

    # filter invalid relations: either argument is empty
    combined = [r for r in combined if r.arg1 and r.arg2]

    return combined


class ArgumentExtractBiLSTMCRF:
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf, explicits_only=False):
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.no_rnn = no_rnn
        self.no_dense = no_dense
        self.no_crf = no_crf
        self.explicits_only = explicits_only

        self.embd = None
        self.model = None
        self.vocab = {}

    def load(self, path):
        self.model.model.load_weights(os.path.join(path, 'bilstm_all_{}_extract.h5'.format(int(self.explicits_only))))

    def save(self, path):
        self.model.model.save_weights(os.path.join(path, 'bilstm_all_{}_extract.h5'.format(int(self.explicits_only))))

    def init_model(self, parses):
        self.vocab = get_vocab(parses)
        embeddings = load_embedding_matrix('/data/word_vectors/glove.6B.300d.txt', self.vocab, 300)
        self.embd = Embedding(len(self.vocab), 300,
                              weights=[embeddings],
                              input_length=self.window_length,
                              trainable=True)

        self.model = BiLSTMx(self.embd, self.window_length, self.hidden_dim, self.rnn_dim,
                             self.no_rnn, self.no_dense, self.no_crf, 4)

        self.model.summary()
        self.model.compile()

    def fit(self, pdtb, parses, pdtb_val, parses_val, epochs=10, save_path=''):
        self.init_model(parses)

        X_train, y_train = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length,
                                                  explicits_only=self.explicits_only)
        X_val, y_val = generate_pdtb_features(pdtb_val, parses_val, self.vocab, self.window_length,
                                              explicits_only=self.explicits_only)

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=2),
            EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
        ]
        if save_path:
            callbacks.append(ModelCheckpoint(os.path.join(save_path, 'bilstm_all_extract.ckp.h5'), save_best_only=True,
                                             save_weights_only=True))
            callbacks.append(CSVLogger(os.path.join(save_path, 'logs.csv')))

        self.model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=512, callbacks=callbacks)

    def score_on_features(self, X, y):
        y_pred = np.concatenate(self.model.predict(X).argmax(-1))
        y = np.concatenate(y.argmax(-1))
        logger.info("Evaluation: BiLSTM CRF Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length)
        self.score_on_features(X, y)

    def extract_arguments(self, words, strides=1, max_distance=0.5):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])

        windows = extract_windows(tokens, self.window_length, strides, offset)
        y_hat = self.model.predict(windows, batch_size=512)

        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)

        relations_hat = reduce_relation_predictions(relations_hat, max_distance=max_distance)

        return relations_hat


class ArgumentExtractBiLSTMCRFwithConn:
    def __init__(self, window_length, hidden_dim, rnn_dim, no_rnn, no_dense, no_crf):
        self.window_length = window_length
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.no_rnn = no_rnn
        self.no_dense = no_dense
        self.no_crf = no_crf

        self.embd = None
        self.model = None
        self.vocab = {}

    def load(self, path):
        self.model.model.load_weights(os.path.join(path, 'bilstm_arg_extract.h5'))

    def save(self, path):
        self.model.model.save_weights(os.path.join(path, 'bilstm_arg_extract.h5'))

    def init_model(self, parses):
        self.vocab = get_vocab(parses)
        embeddings = load_embedding_matrix('/data/word_vectors/glove.6B.300d.txt', self.vocab, 300)
        self.embd = Embedding(len(self.vocab), 300,
                              weights=[embeddings],
                              input_length=self.window_length,
                              trainable=True)

        self.model = BiLSTMx(self.embd, self.window_length, self.hidden_dim, self.rnn_dim,
                             self.no_rnn, self.no_dense, self.no_crf, 4)

        self.model.summary()
        self.model.compile()

    def fit(self, pdtb, parses, pdtb_val, parses_val, epochs=10, save_path=''):
        self.init_model(parses)

        X_train, y_train = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length, explicits_only=True,
                                                  positives_only=True)
        X_val, y_val = generate_pdtb_features(pdtb_val, parses_val, self.vocab, self.window_length, explicits_only=True,
                                              positives_only=True)

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=2),
            EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
        ]
        if save_path:
            callbacks.append(ModelCheckpoint(os.path.join(save_path, 'bilstm_arg_extract.ckp.h5'), save_best_only=True,
                                             save_weights_only=True))
            callbacks.append(CSVLogger(os.path.join(save_path, 'logs.csv')))

        self.model.fit(X_train, y_train, X_val, y_val,
                       epochs=epochs, batch_size=512, callbacks=callbacks)

    def score_on_features(self, X, y):
        y_pred = np.concatenate(self.model.predict(X).argmax(-1))
        y = np.concatenate(y.argmax(-1))
        logger.info("Evaluation: BiLSTM CRF Model")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

    def score(self, pdtb, parses):
        X, y = generate_pdtb_features(pdtb, parses, self.vocab, self.window_length)
        self.score_on_features(X, y)

    def extract_arguments(self, words, relations, strides=1, max_distance=0.5):
        offset = self.window_length // 2
        tokens = np.array([self.vocab.get(w.lower(), 1) for w in words])
        conn_pos = [min([i[2] for i in r['Connective']['TokenList']]) for r in relations]
        word_windows = extract_windows(tokens, self.window_length, strides, offset)
        y_hat = self.model.predict(word_windows, batch_size=512)
        relations_hat = predict_discourse_windows_for_id(tokens, y_hat, strides, offset)
        relations = [relations_hat[i] for i in conn_pos]
        return relations


if __name__ == "__main__":
    logger = init_logger(path='info.log')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    pdtb_train = [json.loads(s) for s in
                  open('/data/discourse/conll2016/en.train/relations.json', 'r').readlines()]
    parses_train = json.loads(open('/data/discourse/conll2016/en.train/parses.json').read())
    pdtb_val = [json.loads(s) for s in open('/data/discourse/conll2016/en.dev/relations.json', 'r').readlines()]
    parses_val = json.loads(open('/data/discourse/conll2016/en.dev/parses.json').read())
    pdtb_test = [json.loads(s) for s in open('/data/discourse/conll2016/en.test/relations.json', 'r').readlines()]
    parses_test = json.loads(open('/data/discourse/conll2016/en.test/parses.json').read())

    clf = ArgumentExtractBiLSTMCRFwithConn(window_length=150, hidden_dim=64, rnn_dim=256, no_rnn=False, no_dense=False,
                                           no_crf=False)
    logger.info('Train CRF model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=15, save_path='../bilstm-tmp')
    clf.save('../bilstm-tmp')
    logger.info('Evaluation on VAL')
    clf.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)

    clf = ArgumentExtractBiLSTMCRF(window_length=150, hidden_dim=64, rnn_dim=256, no_rnn=False, no_dense=False,
                                   no_crf=False, explicits_only=True)
    logger.info('Train CRF model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=15, save_path='../bilstm-tmp')
    clf.save('../bilstm-tmp')
    logger.info('Evaluation on VAL')
    clf.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)

    clf = ArgumentExtractBiLSTMCRF(window_length=150, hidden_dim=64, rnn_dim=256, no_rnn=False, no_dense=False,
                                   no_crf=False, explicits_only=False)
    logger.info('Train CRF model')
    clf.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=15, save_path='../bilstm-tmp')
    clf.save('../bilstm-tmp')
    logger.info('Evaluation on VAL')
    clf.score(pdtb_val, parses_val)
    logger.info('Evaluation on TEST')
    clf.score(pdtb_test, parses_test)
