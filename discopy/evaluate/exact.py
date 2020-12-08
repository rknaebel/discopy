import logging

import numpy as np

from discopy.utils import Relation

logger = logging.getLogger('discopy')


# def print_results(results, mode):
#     logger.info('==========================================================')
#     logger.info('Evaluation for {} discourse relations:'.format(mode))
#     logger.info('==========================================================')
#     logger.info('Conn extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[0])))
#     logger.info('Arg1 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[1])))
#     logger.info('Arg2 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[2])))
#     logger.info('Concat(Arg1, Arg2) extractor: P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[3])))
#     logger.info('Sense:                        P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[4])))


def evaluate_conll_document(gold_conll_list, pred_conll_list, threshold=0.9):
    gold_list = [Relation.from_conll(r) for r in gold_conll_list]
    pred_list = [Relation.from_conll(r) for r in pred_conll_list]
    connective_cm = evaluate_connectives(gold_list, pred_list, threshold)
    arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, pred_list, threshold)
    sense_cm, alignment = evaluate_sense(gold_list, pred_list, threshold)
    return (
        round(compute_prf(*connective_cm)[2], 2),
        round(compute_prf(*arg1_cm)[2], 2),
        round(compute_prf(*arg2_cm)[2], 2),
        round(compute_prf(*rel_arg_cm)[2], 2),
        round(compute_prf(*sense_cm)[2], 2),
    )


# def evaluate_explicit_arguments(gold_relations: dict, predicted_relations: dict, threshold=0.9):
#     results = []
#     for doc_id in gold_relations.keys():
#         gold_list = [r for r in gold_relations.get(doc_id, []) if r.is_explicit()]
#         predicted_list = [r for r in predicted_relations.get(doc_id, []) if r.is_explicit()]
#
#         connective_cm = evaluate_connectives(gold_list, predicted_list, threshold)
#         arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list, threshold)
#         sense_cm, alignment = evaluate_sense(gold_list, predicted_list, threshold)
#
#         results.append(
#             np.array([
#                 connective_cm,
#                 arg1_cm,
#                 arg2_cm,
#                 rel_arg_cm,
#                 sense_cm,
#             ])
#         )
#     results = np.stack(results).sum(axis=0)
#     logger.info('==========================================================')
#     logger.info('Evaluation for EXPLICIT discourse relations ({}):'.format(threshold))
#     logger.info('==========================================================')
#     logger.info('Conn extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[0])))
#     logger.info('Arg1 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[1])))
#     logger.info('Arg2 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[2])))
#     logger.info('Concat(Arg1, Arg2) extractor: P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[3])))
#     logger.info('==========================================================')
#     return results


# def evaluate_all(gold_relations: dict, predicted_relations: dict, threshold=0.9):
#     all_results = {}
#     results = []
#     for doc_id in gold_relations.keys():
#         all_results[doc_id] = {
#             'all': {},
#             'explicit': {},
#             'implicit': {}
#         }
#
#         gold_list = gold_relations.get(doc_id, [])
#         predicted_list = predicted_relations.get(doc_id, [])
#
#         connective_cm = evaluate_connectives(gold_list, predicted_list, threshold)
#         arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list, threshold)
#         sense_cm, alignment = evaluate_sense(gold_list, predicted_list, threshold)
#
#         results.append(
#             np.array([
#                 connective_cm,
#                 arg1_cm,
#                 arg2_cm,
#                 rel_arg_cm,
#                 sense_cm,
#             ]))
#
#         all_results[doc_id]['all'] = {
#             'DocID': doc_id,
#             'Conn': compute_prf(*results[-1][0]),
#             'Arg1': compute_prf(*results[-1][1]),
#             'Arg2': compute_prf(*results[-1][2]),
#             'Arg1+Arg2': compute_prf(*results[-1][3]),
#             'Sense': compute_prf(*results[-1][4]),
#             'Alignment': alignment,
#         }
#
#     results = np.stack(results).sum(axis=0)
#     print_results(results, 'ALL')
#
#     results = []
#     for doc_id in gold_relations.keys():
#         gold_list = [r for r in gold_relations.get(doc_id, []) if r.is_explicit()]
#         predicted_list = [r for r in predicted_relations.get(doc_id, []) if r.is_explicit()]
#
#         connective_cm = evaluate_connectives(gold_list, predicted_list, threshold)
#         arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list, threshold)
#         sense_cm, alignment = evaluate_sense(gold_list, predicted_list, threshold)
#
#         results.append(
#             np.array([
#                 connective_cm,
#                 arg1_cm,
#                 arg2_cm,
#                 rel_arg_cm,
#                 sense_cm,
#             ])
#         )
#
#         all_results[doc_id]['explicit'] = {
#             'DocID': doc_id,
#             'Conn': compute_prf(*results[-1][0]),
#             'Arg1': compute_prf(*results[-1][1]),
#             'Arg2': compute_prf(*results[-1][2]),
#             'Arg1+Arg2': compute_prf(*results[-1][3]),
#             'Sense': compute_prf(*results[-1][4]),
#             'Alignment': alignment,
#         }
#
#     results = np.stack(results).sum(axis=0)
#     print_results(results, 'EXPLICIT')
#
#     results = []
#     for doc_id in gold_relations.keys():
#         gold_list = [r for r in gold_relations.get(doc_id, []) if not r.is_explicit()]
#         predicted_list = [r for r in predicted_relations.get(doc_id, []) if not r.is_explicit()]
#
#         connective_cm = evaluate_connectives(gold_list, predicted_list, threshold)
#         arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list, threshold)
#         sense_cm, alignment = evaluate_sense(gold_list, predicted_list, threshold)
#
#         results.append(
#             np.array([
#                 connective_cm,
#                 arg1_cm,
#                 arg2_cm,
#                 rel_arg_cm,
#                 sense_cm,
#             ])
#         )
#
#         all_results[doc_id]['implicit'] = {
#             'DocID': doc_id,
#             'Conn': compute_prf(*results[-1][0]),
#             'Arg1': compute_prf(*results[-1][1]),
#             'Arg2': compute_prf(*results[-1][2]),
#             'Arg1+Arg2': compute_prf(*results[-1][3]),
#             'Sense': compute_prf(*results[-1][4]),
#             'Alignment': alignment,
#         }
#     results = np.stack(results).sum(axis=0)
#     print_results(results, 'NON-EXPLICIT')
#     logger.info('==========================================================')
#
#     return all_results


def evaluate_argument_extractor(gold_list, predicted_list, threshold=0.9):
    gold_arg1 = [r.arg1 for r in gold_list]
    predicted_arg1 = [r.arg1 for r in predicted_list]
    arg1_cm = compute_confusion_counts(gold_arg1, predicted_arg1, compute_span_f1, threshold)

    gold_arg2 = [r.arg2 for r in gold_list]
    predicted_arg2 = [r.arg2 for r in predicted_list]
    arg2_cm = compute_confusion_counts(gold_arg2, predicted_arg2, compute_span_f1, threshold)

    gold_arg12 = [arg1 | arg2 for arg1, arg2 in zip(gold_arg1, gold_arg2)]
    predicted_arg12 = [arg1 | arg2 for arg1, arg2 in zip(predicted_arg1, predicted_arg2)]
    rel_arg_cm = compute_confusion_counts(gold_arg12, predicted_arg12, compute_span_f1, threshold)
    return arg1_cm, arg2_cm, rel_arg_cm


def evaluate_connectives(gold_list, predicted_list, threshold=0.9):
    explicit_gold_list = [r.conn for r in gold_list if r.is_explicit()]
    explicit_predicted_list = [r.conn for r in predicted_list if r.is_explicit()]
    connective_cm = compute_confusion_counts(
        explicit_gold_list, explicit_predicted_list, compute_span_f1, threshold)
    return connective_cm


def compute_span_f1(g_index_set: set, p_index_set: set) -> float:
    correct = len(g_index_set.intersection(p_index_set))
    if correct == 0:
        return 0.0
    precision = float(correct) / len(p_index_set)
    recall = float(correct) / len(g_index_set)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_sense(gold_list, predicted_list, threshold=0.9):
    tp = fp = fn = 0
    gold_to_predicted_map = _link_gold_predicted(gold_list, predicted_list, threshold)
    for gi, gr in enumerate(gold_list):
        if gi in gold_to_predicted_map:
            # TODO check change
            # if any(g.startswith(p) for g in gr.senses for p in predicted_list[gold_to_predicted_map[gi]].senses):
            if any(g in predicted_list[gold_to_predicted_map[gi]].senses for g in gr.senses):
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return np.array([tp, fp, fn]), gold_to_predicted_map


def compute_confusion_counts(gold_list, predicted_list, matching_fn, threshold=0.9):
    tp = fp = 0
    unmatched = np.ones(len(predicted_list), dtype=bool)
    for gold_span in gold_list:
        for i, predicted_span in enumerate(predicted_list):
            if unmatched[i] and matching_fn(gold_span, predicted_span) > threshold:
                tp += 1
                unmatched[i] = 0
                break
        else:
            fp += 1
    # Predicted span that does not match with any
    fn = unmatched.sum()

    return np.array([tp, fp, fn])


def compute_prf(tp, fp, fn):
    if tp + fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def _link_gold_predicted(gold_list, predicted_list, threshold=0.9):
    """
    Link gold relations to the predicted relations that fits best
    based on the almost exact matching criterion
    """
    gold_to_predicted_map = {}

    for gi, gr in enumerate(gold_list):
        for pi, pr in enumerate(predicted_list):
            if compute_span_f1(gr.arg1 | gr.arg2, pr.arg1 | pr.arg2) > threshold:
                gold_to_predicted_map[gi] = pi
    return gold_to_predicted_map
