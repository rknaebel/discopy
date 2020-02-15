from collections import defaultdict, Counter

import numpy as np

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
    words = [w[0] for s in doc['sentences'] for w in s['words']]
    pos = [w[1]['PartOfSpeech'] for s in doc['sentences'] for w in s['words']]
    doc_labels = []
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
    return {
        'Relations': np.array(doc_labels),
        'Words': np.array(words),
        'POS': np.array(pos),
    }


def generate_pdtb_features(pdtb, parses, vocab, window_length, explicits_only=False, positives_only=False):
    document_features = {}
    pdtb_group = group_by_doc_id(pdtb, explicits_only)
    for doc_id, doc in parses.items():
        document_features[doc_id] = extract_document_features(doc, pdtb_group[doc_id])

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
        i = 0
        for relation, centroid in zip(relations, centroids):
            pos[i, :, 0] = hashes[centroid - 2 - left_side:centroid - 2 + right_size]
            pos[i, :, 1] = relation[centroid - 2 - left_side:centroid - 2 + right_size]
            i += 1
            pos[i, :, 0] = hashes[centroid - 1 - left_side:centroid - 1 + right_size]
            pos[i, :, 1] = relation[centroid - 1 - left_side:centroid - 1 + right_size]
            i += 1
            pos[i, :, 0] = hashes[centroid - left_side:centroid + right_size]
            pos[i, :, 1] = relation[centroid - left_side:centroid + right_size]
            i += 1
            pos[i, :, 0] = hashes[centroid + 1 - left_side:centroid + 1 + right_size]
            pos[i, :, 1] = relation[centroid + 1 - left_side:centroid + 1 + right_size]
            i += 1
            pos[i, :, 0] = hashes[centroid + 2 - left_side:centroid + 2 + right_size]
            pos[i, :, 1] = relation[centroid + 2 - left_side:centroid + 2 + right_size]
            i += 1
        if not positives_only:
            # print(hashes)
            centroids_mask = np.zeros_like(hashes)
            centroids_mask[:size] = 1
            centroids_mask[-size:] = 1
            for r in range(repeats - 1):
                centroids_mask[centroids - r] = 1
                centroids_mask[centroids + r] = 1
            # print(centroids_mask)

            non_centroids = np.arange(len(centroids_mask))[centroids_mask == 0]
            # print(non_centroids)
            neg = np.zeros((len(non_centroids), size, 2), dtype=int)
            for i, centroid in enumerate(non_centroids):
                neg[i, :, 0] = hashes[centroid - left_side:centroid + right_size]

            neg = neg[np.random.choice(len(neg), int(len(pos)))]
            data = np.concatenate([pos, neg])
        else:
            data = pos
    else:
        return [], []

    X = data[:, :, 0]
    y = np.clip(data[:, :, 1], 0, 3)
    y = (np.arange(4) == y[..., None]).astype(bool)
    return X, y


def extract_windows(tokens, window_length, strides, offset):
    nb_tokens = len(tokens)
    tokens = np.pad(tokens, (window_length, window_length), mode='constant', constant_values=0)
    windows = []
    for i in range(0, nb_tokens, strides):
        window = tokens[i + window_length - offset:i + 2 * window_length - offset]
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
