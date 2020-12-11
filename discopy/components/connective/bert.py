import logging
import os
from typing import List

import click
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from discopy.components.component import Component
from discopy.components.connective.base import get_connective_candidates
from discopy.data.doc import Document, Sentence
from discopy.data.loaders.conll import load_conll_dataset
from discopy.data.relation import Relation
from discopy.utils import init_logger

logger = logging.getLogger('discopy')


def get_conn_model(in_size, out_size, hidden_size, hidden_size2=256):
    if out_size == 1:
        out_act = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        out_act = 'softmax'
        loss = 'sparse_categorical_crossentropy'
    ipt_conn = tf.keras.layers.Input(shape=(in_size,), name='connective')
    out_conn = tf.keras.layers.Dense(hidden_size, activation='selu')(ipt_conn)
    out_conn = tf.keras.layers.Dense(hidden_size2, activation='selu')(out_conn)
    out_conn = tf.keras.layers.Dense(out_size, activation=out_act)(out_conn)
    model = tf.keras.models.Model(ipt_conn, out_conn)
    model.compile('adam', loss, metrics=['accuracy'])
    return model


simple_map = {
    "''": '"',
    "``": '"',
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "n't": "not"
}


def get_sentence_embeddings(sent: Sentence, tokenizer, model, device='cpu'):
    subtokens = [tokenizer.tokenize(simple_map.get(t.surface, t.surface)) for t in sent.tokens]
    lengths = [len(s) for s in subtokens]
    tokens_ids = tokenizer.convert_tokens_to_ids([ts for t in subtokens for ts in t])
    tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
    tokens_pt = torch.tensor([tokens_ids]).to(device)
    outputs, pooled = model(tokens_pt)
    embeddings = np.zeros((len(lengths), outputs.shape[-1]), np.float32)
    len_left = 1
    outputs = outputs.detach().cpu().numpy()[0]
    for i, length in enumerate(lengths):
        embeddings[i] = outputs[len_left:len_left + length].mean(axis=0)
    return outputs


def get_bert_features(idxs, doc_bert, used_context=0):
    idxs = list(idxs)
    doc_bert = np.array(doc_bert)
    pad = np.zeros_like(doc_bert[0])
    embd = doc_bert[idxs].mean(axis=0)
    if used_context > 0:
        left = [doc_bert[i] if i >= 0 else pad for i in range(min(idxs) - used_context, min(idxs))]
        right = [doc_bert[i] if i < len(doc_bert) else pad for i in range(max(idxs) + 1, max(idxs) + 1 + used_context)]
        embd = np.concatenate(left + [embd] + right).flatten()
    return embd


def generate_pdtb_features(docs: List[Document], used_context=0, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = AutoModel.from_pretrained('bert-base-cased').eval().to(device)
    features = []
    for doc in tqdm(docs):
        doc_bert = np.concatenate([get_sentence_embeddings(s, tokenizer, model, device) for s in doc.sentences])
        global_id_map = {(s_i, t.local_idx): t.idx for s_i, s in enumerate(doc.sentences) for t in s.tokens}
        conns = {tuple(t.idx for t in r.conn.tokens) for r in doc.relations}
        for sent_i, sentence in enumerate(doc.sentences):
            for connective_candidate in get_connective_candidates(sentence):
                conn_idxs = tuple(global_id_map[(sent_i, i)] for i, c in connective_candidate)
                if conn_idxs in conns:
                    features.append((get_bert_features(conn_idxs, doc_bert, used_context), 1))
                else:
                    features.append((get_bert_features(conn_idxs, doc_bert, used_context), 0))
    x, y = list(zip(*features))
    return np.stack(x), np.array(y)


class ConnectiveClassifier(Component):
    def __init__(self, used_context: int = 0, device: str = 'cpu'):
        in_size = 768 + 2 * used_context * 768
        self.model = get_conn_model(in_size, 1, 1024)
        self.used_context = used_context
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        print("Used device:", self.device)

    def load(self, path):
        if not os.path.exists(os.path.join(path, f'connective_nn_{self.used_context}.model')):
            raise FileNotFoundError("Model not found.")
        self.model = tf.keras.models.load_model(os.path.join(path, f'connective_nn_{self.used_context}.model'))

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, f'connective_nn_{self.used_context}.model'))

    def fit(self, docs_train: List[Document], docs_val: List[Document] = None):
        if docs_val is None:
            raise ValueError("Validation data is missing.")
        x_train, y_train = generate_pdtb_features(docs_train, used_context=self.used_context, device=self.device)
        x_val, y_val = generate_pdtb_features(docs_val, used_context=self.used_context, device=self.device)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), verbose=1, shuffle=True, epochs=10,
                       batch_size=64,
                       callbacks=[
                           tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0)
                       ])

    def score_on_features(self, x, y):
        y_pred = self.model.predict(x).round()
        logger.info("Evaluation: Connective")
        logger.info("    Acc  : {:<06.4}".format(accuracy_score(y, y_pred)))
        prec, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='macro')
        logger.info("    Macro: P {:<06.4} R {:<06.4} F1 {:<06.4}".format(prec, recall, f1))
        logger.info("    Kappa: {:<06.4}".format(cohen_kappa_score(y, y_pred)))

    def score(self, docs):
        x, y = generate_pdtb_features(docs, used_context=self.used_context, device=self.device)
        self.score_on_features(x, y)

    def parse(self, doc: Document, *args, **kwargs):
        relations: List[Relation] = []
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        model = AutoModel.from_pretrained('bert-base-cased')
        doc_bert = np.concatenate([get_sentence_embeddings(s, tokenizer, model) for s in doc.sentences])
        global_id_map = {(s_i, t.local_idx): t.idx for s_i, s in enumerate(doc.sentences) for t in s.tokens}
        for sent_i, sent in enumerate(doc.sentences):
            for connective_candidate in get_connective_candidates(sent):
                conn_idxs = tuple(global_id_map[(sent_i, i)] for i, c in connective_candidate)
                features = get_bert_features(conn_idxs, doc_bert, self.used_context)
                pred = self.model.predict(np.expand_dims(features, axis=0))
                if pred > 0.5:
                    conn_tokens = [sent.tokens[i] for i, c in connective_candidate]
                    relations.append(Relation(
                        conn=conn_tokens,
                        type='Explicit'
                    ))
        return relations


@click.command()
@click.argument('conll-path')
def main(conll_path):
    logger = init_logger()
    docs_val = load_conll_dataset(os.path.join(conll_path, 'en.dev'))[:5]
    clf = ConnectiveClassifier(device='cuda')
    try:
        clf.load('models/nn')
    except FileNotFoundError:
        docs_train = load_conll_dataset(os.path.join(conll_path, 'en.train'))[:20]
        logger.info('Train model')
        clf.fit(docs_train, docs_val)
        logger.info('Evaluation on TRAIN')
        clf.score(docs_train)
        clf.save('models/nn')
    logger.info('Evaluation on TEST')
    clf.score(docs_val)
    logger.info('Parse one document')
    print(docs_val[0].to_json())
    print(clf.parse(docs_val[0], []))


if __name__ == "__main__":
    main()
