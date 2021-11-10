import math
from collections import Counter
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from discopy_data.data.doc import Document
from discopy_data.data.relation import Relation
from discopy_data.data.token import Token


def get_class_weights(y, smooth_factor=0.0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)
    if smooth_factor > 0.0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


class PDTBWindowSequence(tf.keras.utils.Sequence):
    def __init__(self, docs: List[Document], window_length: int, sense_map, batch_size: int, nb_classes: int,
                 explicits_only: bool = False, positives_only: bool = False):
        self.rng = np.random.default_rng()
        self.docs = []
        for doc in tqdm(docs):
            extraction = extract_document_training_windows(doc, sense_map, window_length, explicits_only,
                                                           positives_only)
            if extraction is not None:
                doc_windows, arg_labels, senses = extraction
                arg_labels = np.clip(arg_labels, 0, nb_classes - 1)
                arg_labels = (np.arange(nb_classes) == arg_labels[..., None]).astype(bool)
                senses = (np.arange(senses.max() + 1) == senses[..., None]).astype(bool)
                self.docs.append({
                    'doc_id': doc.doc_id,
                    'embeddings': doc.get_embeddings(),
                    'nb': len(senses),
                    'windows': doc_windows,
                    'args': arg_labels,
                    'senses': senses
                })
        self.instances = np.array(
            [[doc_id, i] for doc_id, doc_windows in enumerate(self.docs) for i in range(doc_windows['nb'])])
        self.rng.shuffle(self.instances)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.instances) / self.batch_size)

    def __getitem__(self, idx):
        idxs = self.instances[idx * self.batch_size:(idx + 1) * self.batch_size]
        windows = np.stack([self.docs[doc_id]['embeddings'][self.docs[doc_id]['windows'][i]] for doc_id, i in idxs])
        args = np.stack([self.docs[doc_id]['args'][i] for doc_id, i in idxs])
        return windows, args

    def on_epoch_end(self):
        self.rng.shuffle(self.instances)

    def get_balanced_class_weights(self):
        y = np.concatenate([doc['args'] for doc in self.docs])
        y = y.argmax(-1).flatten()
        return get_class_weights(y, 0)


def extract_document_training_windows(doc: Document, sense_map, size=100, explicits_only: bool = False,
                                      positives_only=False):
    """
    Args:
        doc (Document):
        sense_map:
        size:
        explicits_only:
        positives_only:
    """
    # bert_doc = doc.get_embeddings()
    bert_doc = np.arange(len(doc.get_tokens()))
    left_side = size // 2
    right_size = size - left_side
    repeats = 3
    doc_relations = [r for r in doc.relations if not explicits_only or r.is_explicit()]
    if not len(doc_relations):
        return None

    # hashes = np.pad(bert_doc, pad_width=((size, size), (0, 0)), mode='constant', constant_values=0)
    hashes = np.pad(bert_doc, pad_width=size, mode='constant', constant_values=0)
    relations = get_relations_matrix(doc_relations, len(bert_doc), size)
    centroids = relations.argmax(1)

    # differentiate context and non-context embeddings
    pos_word_window = np.zeros((len(relations) * repeats, size), dtype=np.uint16)
    pos_rel_window = np.zeros((len(relations) * repeats, size), dtype=np.uint8)
    pos_senses = np.repeat([sense_map.get(r.senses[0], 0) for r in doc_relations], repeats)
    i = 0
    for relation, centroid in zip(relations, centroids):
        # pos_word_window[i, :] = hashes[centroid - 2 - left_side:centroid - 2 + right_size]
        # pos_rel_window[i, :] = relation[centroid - 2 - left_side:centroid - 2 + right_size]
        # i += 1
        pos_word_window[i, :] = hashes[centroid - 1 - left_side:centroid - 1 + right_size]
        pos_rel_window[i, :] = relation[centroid - 1 - left_side:centroid - 1 + right_size]
        i += 1
        pos_word_window[i, :] = hashes[centroid - left_side:centroid + right_size]
        pos_rel_window[i, :] = relation[centroid - left_side:centroid + right_size]
        i += 1
        pos_word_window[i, :] = hashes[centroid + 1 - left_side:centroid + 1 + right_size]
        pos_rel_window[i, :] = relation[centroid + 1 - left_side:centroid + 1 + right_size]
        i += 1
        # pos_word_window[i, :] = hashes[centroid + 2 - left_side:centroid + 2 + right_size]
        # pos_rel_window[i, :] = relation[centroid + 2 - left_side:centroid + 2 + right_size]
        # i += 1
    if not positives_only:
        centroids_mask = np.zeros(len(hashes))
        centroids_mask[:size] = 1
        centroids_mask[-size:] = 1
        for r in range(repeats - 1):
            centroids_mask[centroids - r] = 1
            centroids_mask[centroids + r] = 1

        non_centroids = np.arange(len(centroids_mask))[centroids_mask == 0]
        neg_word_window = np.zeros((len(non_centroids), size), dtype=np.uint16)
        neg_rel_window = np.zeros((len(non_centroids), size), dtype=np.uint8)
        for i, centroid in enumerate(non_centroids):
            neg_word_window[i, :] = hashes[centroid - left_side:centroid + right_size]

        neg_idxs = np.random.permutation(len(neg_word_window))[:len(pos_word_window)]
        neg_word_window = neg_word_window[neg_idxs]
        neg_rel_window = neg_rel_window[neg_idxs]
        neg_senses = np.zeros_like(pos_senses)

        token_windows = np.concatenate([pos_word_window, neg_word_window])
        arg_labels = np.concatenate([pos_rel_window, neg_rel_window])
        senses = np.concatenate([pos_senses, neg_senses])
    else:
        token_windows = pos_word_window
        arg_labels = pos_rel_window
        senses = pos_senses
    return token_windows, arg_labels, senses


def get_relations_matrix(doc_relations: List[Relation], doc_length: int, pad_size: int):
    relations = np.zeros((len(doc_relations), doc_length), np.uint8)
    for r_i, r in enumerate(doc_relations):
        relations[r_i, [t.idx for t in r.arg1.tokens]] = 1
        relations[r_i, [t.idx for t in r.arg2.tokens]] = 2
        relations[r_i, [t.idx for t in r.conn.tokens]] = 3
    return np.pad(relations, pad_width=((0, 0), (pad_size, pad_size)), mode='constant', constant_values=0)


def extract_windows(embeddings, size, strides, offset):
    """
    Args:
        embeddings:
        size:
        strides:
        offset:
    """
    nb_tokens = len(embeddings)
    embeddings = np.pad(embeddings, pad_width=((size, size), (0, 0)), mode='constant', constant_values=0)
    windows = []
    for i in range(0, nb_tokens, strides):
        window = embeddings[i + size - offset:i + 2 * size - offset]
        windows.append(window)
    windows = np.stack(windows)
    return windows


def extract_relation_from_window(window_pred, start_idx, tokens):
    """
    Args:
        window_pred:
        start_idx:
        tokens:
    """
    nb_tokens = len(tokens)
    idxs = np.arange(len(window_pred)) + start_idx
    pred = window_pred.argmax(-1)
    relation = Relation()

    for p, t in zip(pred, idxs):
        if t < 0:
            continue
        if t >= nb_tokens:
            break
        if p == 1:
            relation.arg1.add(tokens[t])
        if p == 2:
            relation.arg2.add(tokens[t])
        if p == 3:
            relation.conn.add(tokens[t])
    return relation


def predict_discourse_windows_for_id(tokens: List[Token], windows: np.array, strides: int, offset: int,
                                     start_idxs: int = None):
    """
    Args:
        tokens:
        windows (np.array):
        strides (int):
        offset (int):
        start_idxs (int):
    """
    relations_hat = []
    if start_idxs:
        for i, (w, s) in enumerate(zip(windows, start_idxs)):
            relations_hat.append(extract_relation_from_window(w, s, tokens))
    else:
        start_idx = -offset
        for i, w in enumerate(windows):
            relations_hat.append(extract_relation_from_window(w, start_idx, tokens))
            start_idx += strides
    return relations_hat


# TODO check: majority does not always fit the limit... maybe iter each token instead and take major class
def major_merge_relations(relations: List[Relation]):
    """
    Args:
        relations:
    """
    lim = (len(relations) + 1) // 2  # increase by one to get the majority
    arg1_ctr = Counter(a1 for r in relations for a1 in r.arg1.tokens)
    arg2_ctr = Counter(a2 for r in relations for a2 in r.arg2.tokens)
    conn_ctr = Counter(c for r in relations for c in r.conn.tokens)
    relation = Relation()
    for a1, a1_c in arg1_ctr.items():
        if a1_c > lim:
            relation.arg1.add(a1)
    for a2, a2_c in arg2_ctr.items():
        if a2_c > lim:
            relation.arg2.add(a2)
    for conn, conn_c in conn_ctr.items():
        if conn_c > lim:
            relation.conn.add(conn)
    return relation


def reduce_relation_predictions(relations: List[Relation], max_distance: float = 0.5):
    """
    Args:
        relations:
        max_distance (float):
    """
    if len(relations) == 0:
        return []
    combined = []
    current = [relations[0]]
    distances = [relations[i].distance(relations[i + 1]) for i in range(len(relations) - 1)]
    for i, d in enumerate(distances):
        next_rel = relations[i + 1]
        if d < max_distance:
            current.append(next_rel)
        else:
            combined.append(current)
            current = [next_rel]

    # filter invalid relations: either argument is empty
    combined = [[r for r in rels if r.is_valid()] for rels in combined]
    # TODO whats the best number of partial relations to depend on?
    combined = [rr for rr in combined if len(rr) > 2]
    combined = [major_merge_relations(rr) for rr in combined]
    combined = [r for r in combined if r.is_valid()]

    return combined
