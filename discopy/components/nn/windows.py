import os
from collections import Counter
from typing import List

import click
import numpy as np
from tqdm import tqdm

from discopy.data.doc import BertDocument
from discopy.data.loaders.conll import load_bert_conll_dataset
from discopy.data.relation import Relation
from discopy.data.token import Token


def generate_pdtb_features(docs: List[BertDocument], window_length: int, sense_map,
                           explicits_only: bool = False, positives_only: bool = False):
    """
    Args:
        docs:
        window_length (int):
        sense_map:
        explicits_only (bool):
        positives_only (bool):
    """
    token_windows, arg_labels, senses = [], [], []
    for doc in tqdm(docs):
        doc_bert = doc.get_embeddings()
        res = extract_document_training_windows(doc, doc_bert, sense_map, window_length, explicits_only, positives_only)
        if res is not None:
            doc_windows, doc_arg_labels, doc_senses = res
            token_windows.append(doc_windows)
            arg_labels.append(doc_arg_labels)
            senses.append(doc_senses)
    arg_labels = np.concatenate(arg_labels)
    arg_labels = np.clip(arg_labels, 0, 3)
    arg_labels = (np.arange(4) == arg_labels[..., None]).astype(bool)
    senses = np.concatenate(senses)
    senses = (np.arange(senses.max() + 1) == senses[..., None]).astype(bool)
    token_windows = np.concatenate(token_windows)
    # create shuffled indices for return values
    perm = np.random.permutation(len(arg_labels))
    return token_windows[perm], arg_labels[perm], senses[perm]


def extract_document_training_windows(doc: BertDocument, bert_doc, sense_map, size=100,
                                      explicits_only: bool = False, positives_only=False):
    """
    Args:
        doc (BertDocument):
        bert_doc:
        sense_map:
        size:
        explicits_only:
        positives_only:
    """
    left_side = size // 2
    right_size = size - left_side
    repeats = 3
    doc_relations = [r for r in doc.relations if not explicits_only or r.type == 'Explicit']
    if not len(doc_relations):
        return None

    relations = np.zeros((len(doc_relations), len(bert_doc)), np.uint8)
    for r_i, r in enumerate(doc_relations):
        relations[r_i, [t.idx for t in r.conn.tokens]] = 3
        relations[r_i, [t.idx for t in r.arg2.tokens]] = 2
        relations[r_i, [t.idx for t in r.arg1.tokens]] = 1

    hashes = np.pad(bert_doc, mode='constant', pad_width=((size, size), (0, 0)), constant_values=0)
    relations = np.pad(relations, ((0, 0), (size, size)), mode='constant', constant_values=0)
    centroids = relations.argmax(1)

    # differentiate context and non-context embeddings
    pos_word_window = np.zeros((len(relations) * repeats, size, hashes.shape[1]), dtype=float)
    pos_rel_window = np.zeros((len(relations) * repeats, size), dtype=int)
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
        neg_word_window = np.zeros((len(non_centroids), size, hashes.shape[1]), dtype=float)
        neg_rel_window = np.zeros((len(non_centroids), size), dtype=int)
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
    combined = [[r for r in rr if r.arg1 and r.arg2] for rr in combined]
    # TODO whats the best number of partial relations to depend on?
    combined = [rr for rr in combined if len(rr) > 2]
    combined = [major_merge_relations(rr) for rr in combined]
    combined = [r for r in combined if r.arg1 and r.arg2]

    return combined
