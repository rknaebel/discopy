import multiprocessing
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from discopy.utils import Relation


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


def extract_document_features(doc, relations):
    words = np.concatenate([s['tokens'] for s in doc['sentences']])
    # pos = [w[1]['PartOfSpeech'] for s in doc['sentences'] for w in s['words']]
    doc_labels = []
    doc_senses = [r['Sense'] for r in relations]
    for r in relations:
        r_labels = []
        for w_i in range(len(words)):
            if w_i in r['Conn']:
                r_labels.append(3)
            elif w_i in r['Arg2']:
                r_labels.append(2)
            elif w_i in r['Arg1']:
                r_labels.append(1)
            else:
                r_labels.append(0)
        doc_labels.append(np.array(r_labels))
    return {
        'Relations': np.array(doc_labels),
        'Words': np.array(words),
        'Senses': np.array(doc_senses),
    }


def process_windows(document_features, window_length, positives_only):
    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
        params = ((features, window_length, positives_only) for features in document_features.values())
        windows = pool.starmap(extract_document_training_windows, params, chunksize=11)
    return zip(*filter(None, windows))


def generate_pdtb_features(pdtb, parses, window_length, explicits_only=False, positives_only=False):
    document_features = {}
    pdtb_group = group_by_doc_id(pdtb, explicits_only)
    for doc_id, doc in tqdm(parses.items()):
        document_features[doc_id] = extract_document_features(doc, pdtb_group[doc_id])
    token_windows, arg_labels, senses = process_windows(document_features, window_length, positives_only)

    arg_labels = np.concatenate(arg_labels)
    arg_labels = np.clip(arg_labels, 0, 3)
    arg_labels = (np.arange(4) == arg_labels[..., None]).astype(bool)
    senses = np.concatenate(senses)
    senses = (np.arange(senses.max() + 1) == senses[..., None]).astype(bool)
    token_windows = np.concatenate(token_windows)

    # create shuffled indices for return values
    perm = np.random.permutation(len(arg_labels))
    return token_windows[perm], arg_labels[perm], senses[perm]


def extract_document_training_windows(document, size=100, positives_only=False):
    left_side = size // 2
    right_size = size - left_side
    repeats = 3
    use_indices = len(document['Words'].shape) == 1
    relations = document['Relations']

    if not len(relations):
        return None

    padding = (size, size) if use_indices else ((size, size), (0, 0))
    hashes = np.pad(document['Words'], mode='constant', pad_width=padding, constant_values=0)
    relations = np.pad(relations, ((0, 0), (size, size)), mode='constant', constant_values=0)
    centroids = relations.argmax(1)

    # differentiate context and non-context embeddings
    if use_indices:
        pos_word_window = np.zeros((len(relations) * repeats, size), dtype=int)
    else:
        pos_word_window = np.zeros((len(relations) * repeats, size, hashes.shape[1]), dtype=float)
    pos_rel_window = np.zeros((len(relations) * repeats, size), dtype=int)
    pos_senses = np.repeat(document['Senses'], repeats)

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
        if use_indices:
            neg_word_window = np.zeros((len(non_centroids), size), dtype=int)
        else:
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


def extract_windows(tokens, size, strides, offset):
    nb_tokens = len(tokens)
    use_indices = len(tokens.shape) == 1
    padding = (size, size) if use_indices else ((size, size), (0, 0))
    tokens = np.pad(tokens, padding, mode='constant', constant_values=0)
    windows = []
    for i in range(0, nb_tokens, strides):
        window = tokens[i + size - offset:i + 2 * size - offset]
        windows.append(window)
    windows = np.stack(windows)

    return windows


def extract_relation_from_window(window_pred, start_idx, nb_tokens):
    idxs = np.arange(len(window_pred)) + start_idx
    pred = window_pred.argmax(-1)
    relation = Relation()

    for p, t in zip(pred, idxs):
        if t < 0:
            continue
        if t >= nb_tokens:
            break
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
    else:
        start_idx = -offset
        for i, w in enumerate(windows):
            relations_hat.append(extract_relation_from_window(w, start_idx, nb_tokens))
            start_idx += strides
    return relations_hat


def major_merge_relations(relations):
    lim = (len(relations) + 1) // 2  # increase by one to get the majority
    arg1_ctr = Counter(a1 for r in relations for a1 in r.arg1)
    arg2_ctr = Counter(a2 for r in relations for a2 in r.arg2)
    conn_ctr = Counter(c for r in relations for c in r.conn)
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


def reduce_relation_predictions(relations, max_distance=0.5):
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
