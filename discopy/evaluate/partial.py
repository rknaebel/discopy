from typing import List, Tuple

import numpy as np

from discopy.utils import Relation


def rel_alignment_score(g_relation: Relation, p_relation: Relation) -> float:
    arg1_f1 = arg1_alignment_score(g_relation, p_relation)
    arg2_f1 = arg2_alignment_score(g_relation, p_relation)
    return (arg1_f1 + arg2_f1) / 2


def arg1_alignment_score(g_relation: Relation, p_relation: Relation) -> float:
    arg1_overlap = g_relation.arg1 & p_relation.arg1
    if arg1_overlap:
        return compute_f1_span(g_relation.arg1, p_relation.arg1)
    else:
        return 0.0


def arg2_alignment_score(g_relation: Relation, p_relation: Relation) -> float:
    arg2_overlap = g_relation.arg2 & p_relation.arg2
    if arg2_overlap:
        return compute_f1_span(g_relation.arg2, p_relation.arg2)
    else:
        return 0.0


def compute_f1_span(g_index_set: set, p_index_set: set) -> float:
    """Compute F1 score for a given pair of token list"""
    correct = float(len(g_index_set.intersection(p_index_set)))
    if correct == 0.0:
        return 0.0
    precision = correct / len(p_index_set)
    recall = correct / len(g_index_set)
    return 2 * (precision * recall) / (precision + recall)


def align_relations(doc_gold_list: List[Relation], doc_predicted_list: List[Relation], partial_match_cutoff: float):
    """Aligning two lists of relations
    Input:
        gold_list : a list of ground truth relations
        predicted_list : a list of predicted relations
    Returns:
        A list of alignments between gold and predicted relations
    """
    relation_alignment = _align(doc_gold_list, doc_predicted_list, rel_alignment_score, partial_match_cutoff)
    arg1_alignment = _align(doc_gold_list, doc_predicted_list, arg1_alignment_score, partial_match_cutoff)
    arg2_alignment = _align(doc_gold_list, doc_predicted_list, arg2_alignment_score, partial_match_cutoff)
    return arg1_alignment, arg2_alignment, relation_alignment


def _align(gold_list: List[Relation], predicted_list: List[Relation], alignment_score_fn, partial_match_cutoff: float):
    """Align the gold standard and the predicted discourse relations in the same doc
    """
    rel_score_matrix, rel_adjacency = compute_score_matrix(gold_list, predicted_list, alignment_score_fn,
                                                           partial_match_cutoff)
    # print("score matrix", rel_score_matrix)
    _, index_alignment = _recurs_align_relations(0, set(), len(predicted_list), rel_score_matrix, rel_adjacency)
    rel_alignment = []
    for i, j in index_alignment:
        # if i < 0 or j < 0:
        #     continue
        g_relation = gold_list[i] if i != -1 else None
        p_relation = predicted_list[j] if j != -1 else None
        rel_alignment.append((g_relation, p_relation))
    # print('found alignment')
    return rel_alignment


def compute_score_matrix(gold_list: List[Relation], predicted_list: List[Relation], alignment_score_fn,
                         partial_match_cutoff: float):
    """Compute the weighted adjacency matrix for alignment
    This score matrix serves as an adjacency matrix for searching for
    the best alignment.
    """
    score_matrix = np.zeros((len(gold_list), len(predicted_list)))
    adjacency = np.zeros((len(gold_list), len(predicted_list)))
    for i, g_relation in enumerate(gold_list):
        for j, p_relation in enumerate(predicted_list):
            score = alignment_score_fn(g_relation, p_relation)
            if score >= partial_match_cutoff:
                score_matrix[i, j] = score
                adjacency[i][j] = 1.0
    return score_matrix, adjacency


def _recurs_align_relations(gi: int, pi_used_set: set, num_predicted, score_matrix, adjacency):
    # print('call:', gi, pi_used_set)
    if gi == len(score_matrix):
        alignment = [(-1, pi)
                     for pi in range(num_predicted) if pi not in pi_used_set]
        return 0, alignment
    max_score = 0.0
    max_alignment = []
    found_maximal_match = False
    max_index = max(pi_used_set) if pi_used_set else 0
    possible_idxs = [i for i in np.where(score_matrix[gi] > 0)[0] if i > max_index]
    # print(gi, pi_used_set, possible_idxs)
    for pi in possible_idxs:
        alignment_score = score_matrix[gi][pi]
        # perfect match or one-to-one already
        found_maximal_match = (alignment_score == 1) or \
                              (adjacency.sum(0)[pi] == 1 and len(score_matrix[gi]) == 1)
        if alignment_score > 0 and pi not in pi_used_set:
            pi_used_set.add(pi)
            score, alignment = _recurs_align_relations(
                gi + 1, pi_used_set, num_predicted, score_matrix, adjacency)
            if alignment_score + score >= max_score:
                max_score = alignment_score + score
                max_alignment = alignment + [(gi, pi)]
            pi_used_set.remove(pi)

        if found_maximal_match:
            break

    if not found_maximal_match:
        # print("no maximal match")
        score, alignment = _recurs_align_relations(
            gi + 1, pi_used_set, num_predicted, score_matrix, adjacency)
        if score >= max_score:
            max_score = score
            max_alignment = alignment + [(gi, -1)]
    return max_score, max_alignment


def partial_evaluate(gold_list: List[Relation], predicted_list: List[Relation], partial_match_cutoff: float):
    """Evaluate the parse output with partial matching for arguments
    """
    arg1_alignment, arg2_alignment, relation_alignment = align_relations(gold_list, predicted_list,
                                                                         partial_match_cutoff)
    arg1_match_prf, arg2_match_prf, total_match_prf = evaluate_args(arg1_alignment, arg2_alignment,
                                                                    partial_match_cutoff)
    entire_relation_match_prf = evaluate_rel_arg_whole_rel(relation_alignment, partial_match_cutoff)
    conn_relation_match_prf = evaluate_rel_conn_whole_rel(relation_alignment, partial_match_cutoff)

    results = np.array([
        arg1_match_prf,
        arg2_match_prf,
        total_match_prf,
        entire_relation_match_prf,
        conn_relation_match_prf
    ])
    return results


def evaluate_args(arg1_alignment, arg2_alignment, partial_match_cutoff):
    """Evaluate argument matches"""
    total_arg1_gold, total_arg1_predicted, total_arg1_correct = \
        evaluate_arg_partial_match(arg1_alignment, 1, partial_match_cutoff)
    total_arg2_gold, total_arg2_predicted, total_arg2_correct = \
        evaluate_arg_partial_match(arg2_alignment, 2, partial_match_cutoff)
    arg1_prf = compute_prf(
        total_arg1_gold, total_arg1_predicted, total_arg1_correct)
    arg2_prf = compute_prf(
        total_arg2_gold, total_arg2_predicted, total_arg2_correct)
    rel_arg_prf = compute_prf(
        total_arg1_gold + total_arg2_gold,
        total_arg1_predicted + total_arg2_predicted,
        total_arg1_correct + total_arg2_correct)
    return arg1_prf, arg2_prf, rel_arg_prf


def evaluate_arg_partial_match(relation_pairs, position, partial_match_cutoff) -> Tuple[int, int, int]:
    """Evaluate the argument based on partial matching criterion
    We evaluate the argument as a whole.
    """
    assert position == 1 or position == 2
    total_correct = 0
    total_gold = 0
    total_predicted = 0
    for g_relation, p_relation in relation_pairs:
        assert g_relation is not None or p_relation is not None
        if g_relation is None:
            total_predicted += 1
        elif p_relation is None:
            total_gold += 1
        else:
            g_arg = g_relation.arg1 if position == 1 else g_relation.arg2
            p_arg = p_relation.arg1 if position == 1 else p_relation.arg2
            f1_score = compute_f1_span(g_arg, p_arg)
            if f1_score >= partial_match_cutoff:
                total_correct += 1
            total_predicted += 1
            total_gold += 1
    return total_gold, total_predicted, total_correct


def evaluate_rel_arg_whole_rel(relation_pairs, partial_match_cutoff):
    total_correct = 0.0
    total_gold = 0.0
    total_predicted = 0.0
    for g_relation, p_relation in relation_pairs:
        assert g_relation is not None or p_relation is not None
        if g_relation is None:
            total_predicted += 1
        elif p_relation is None:
            total_gold += 1
        else:
            g_arg1 = g_relation.arg1 if g_relation is not None else set()
            p_arg1 = p_relation.arg1 if p_relation is not None else set()
            arg1_f1_score = compute_f1_span(g_arg1, p_arg1)

            g_arg2 = g_relation.arg2 if g_relation is not None else set()
            p_arg2 = p_relation.arg2 if p_relation is not None else set()
            arg2_f1_score = compute_f1_span(g_arg2, p_arg2)
            if arg1_f1_score >= partial_match_cutoff and arg2_f1_score >= partial_match_cutoff:
                total_correct += 1
                total_predicted += 1
                total_gold += 1
    return compute_prf(total_gold, total_predicted, total_correct)


def evaluate_rel_conn_whole_rel(relation_pairs, partial_match_cutoff):
    """
    A predicted raw connective is considered iff
        1) the predicted raw connective includes the connective "head"
        2) the predicted raw connective tokens are the subset of predicted raw connective tokens
    """
    total_correct = 0.0
    total_gold = 0.0
    total_predicted = 0.0
    for g_relation, p_relation in relation_pairs:
        assert g_relation is not None or p_relation is not None
        if g_relation is None:
            total_predicted += 1
        elif p_relation is None:
            total_gold += 1
        else:
            g_conn = g_relation.conn if g_relation is not None else set()
            p_conn = p_relation.conn if p_relation is not None else set()
            # conn_f1_score = compute_f1_span(g_conn, p_conn)
            if g_conn == p_conn:
                conn_f1_score = 1
            elif not p_conn.issubset(g_conn):
                conn_f1_score = 0
            else:
                # g_conn_head = ConnHeadMapper().DEFAULT_MAPPING.get(' '.join(vocab.token(i) for i in g_conn), '<ukn>')
                # g_conn_head = set(vocab.id(t) for t in g_conn_head)
                # if g_conn_head.issubset(p_conn):
                #     conn_f1_score = 1
                # else:
                #     conn_f1_score = 0
                conn_f1_score = compute_f1_span(g_conn, p_conn)

            if conn_f1_score >= partial_match_cutoff:
                total_correct += 1
                total_predicted += 1
                total_gold += 1
    return compute_prf(total_gold, total_predicted, total_correct)


def compute_prf(total_gold: int, total_predicted: int, total_correct: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1
    Assume binary classification where we are only interested
    in the positive class. In our case, we look at argument extraction.
    """
    if total_predicted == 0:
        precision = 1.0
    else:
        precision = total_correct / total_predicted
    if total_gold == 0:
        recall = 1.0
    else:
        recall = total_correct / total_gold
    f1_score = 2.0 * (precision * recall) / (precision + recall) \
        if precision + recall != 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1_score, 4)
