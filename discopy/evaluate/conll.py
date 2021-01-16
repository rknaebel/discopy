import logging
from typing import List

import numpy as np

from discopy.data.doc import ParsedDocument
from discopy.data.token import TokenSpan
from discopy.data.relation import Relation

logger = logging.getLogger('discopy')


def print_results(results):
    """
    Args:
        results:
    """
    logger.info('==========================================================')
    logger.info('Evaluation for discourse relations:')
    logger.info('==========================================================')
    logger.info('Conn extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*results['conn']))
    logger.info('Arg1 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*results['arg1']))
    logger.info('Arg2 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*results['arg2']))
    logger.info('Concat(Arg1, Arg2) extractor: P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*results['arg12']))
    logger.info('Sense:                        P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*results['sense']))


def score_doc(gold_doc: ParsedDocument, pred_doc: ParsedDocument, threshold=0.9):
    """
    Args:
        gold_doc (ParsedDocument):
        pred_doc (ParsedDocument):
        threshold:
    """
    gold_list = gold_doc.relations
    pred_list = pred_doc.relations
    connective_cm = evaluate_connectives(gold_list, pred_list, threshold)
    arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, pred_list, threshold)
    sense_cm, alignment = evaluate_sense(gold_list, pred_list, threshold)
    return {
        'conn': connective_cm,
        'arg1': arg1_cm,
        'arg2': arg2_cm,
        'arg12': rel_arg_cm,
        'sense': sense_cm,
    }


def evaluate_docs(gold_docs: List[ParsedDocument], pred_docs: List[ParsedDocument], threshold=0.9):
    """
    Args:
        gold_docs:
        pred_docs:
        threshold:
    """
    results = {
        'conn': [],
        'arg1': [],
        'arg2': [],
        'arg12': [],
        'sense': [],
    }
    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        result = score_doc(gold_doc, pred_doc, threshold)
        results['conn'].append(result['conn'])
        results['arg1'].append(result['arg1'])
        results['arg2'].append(result['arg2'])
        results['arg12'].append(result['arg12'])
        results['sense'].append(result['sense'])
    return {
        'conn': compute_prf(*np.sum(np.stack(results['conn']), axis=0)),
        'arg1': compute_prf(*np.sum(np.stack(results['arg1']), axis=0)),
        'arg2': compute_prf(*np.sum(np.stack(results['arg2']), axis=0)),
        'arg12': compute_prf(*np.sum(np.stack(results['arg12']), axis=0)),
        'sense': compute_prf(*np.sum(np.stack(results['sense']), axis=0)),
    }


def evaluate_docs_average(gold_docs: List[ParsedDocument], pred_docs: List[ParsedDocument], threshold=0.9):
    """
    Args:
        gold_docs:
        pred_docs:
        threshold:
    """
    results = {
        'conn': [],
        'arg1': [],
        'arg2': [],
        'arg12': [],
        'sense': [],
    }
    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        result = score_doc(gold_doc, pred_doc, threshold)
        results['conn'].append(compute_prf(*result['conn']))
        results['arg1'].append(compute_prf(*result['arg1']))
        results['arg2'].append(compute_prf(*result['arg2']))
        results['arg12'].append(compute_prf(*result['arg12']))
        results['sense'].append(compute_prf(*result['sense']))
    return {
        'conn': np.mean(np.stack(results['conn']), axis=0),
        'arg1': np.mean(np.stack(results['arg1']), axis=0),
        'arg2': np.mean(np.stack(results['arg2']), axis=0),
        'arg12': np.mean(np.stack(results['arg12']), axis=0),
        'sense': np.mean(np.stack(results['sense']), axis=0),
    }


def evaluate_argument_extractor(gold_list: List[Relation], predicted_list: List[Relation], threshold=0.9):
    """
    Args:
        gold_list:
        predicted_list:
        threshold:
    """
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


def evaluate_connectives(gold_list: List[Relation], predicted_list: List[Relation], threshold=0.9):
    """
    Args:
        gold_list:
        predicted_list:
        threshold:
    """
    explicit_gold_list = [r.conn for r in gold_list if r.is_explicit()]
    explicit_predicted_list = [r.conn for r in predicted_list if r.is_explicit()]
    connective_cm = compute_confusion_counts(
        explicit_gold_list, explicit_predicted_list, compute_span_f1, threshold)
    return connective_cm


def compute_span_f1(g_index_set: TokenSpan, p_index_set: TokenSpan) -> float:
    """
    Args:
        g_index_set (TokenSpan):
        p_index_set (TokenSpan):
    """
    correct = g_index_set.overlap(p_index_set)
    if correct == 0:
        return 0.0
    precision = float(correct) / len(p_index_set)
    recall = float(correct) / len(g_index_set)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_sense(gold_list: List[Relation], predicted_list: List[Relation], threshold=0.9):
    """
    Args:
        gold_list:
        predicted_list:
        threshold:
    """
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


def compute_confusion_counts(gold_list: List[TokenSpan], predicted_list: List[TokenSpan], matching_fn, threshold=0.9):
    """
    Args:
        gold_list:
        predicted_list:
        matching_fn:
        threshold:
    """
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
    """
    Args:
        tp:
        fp:
        fn:
    """
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


def _link_gold_predicted(gold_list: List[Relation], predicted_list: List[Relation], threshold=0.9):
    """Link gold relations to the predicted relations that fits best based on
    the almost exact matching criterion

    Args:
        gold_list:
        predicted_list:
        threshold:
    """
    gold_to_predicted_map = {}

    for gi, gr in enumerate(gold_list):
        for pi, pr in enumerate(predicted_list):
            if compute_span_f1(gr.arg1 | gr.arg2, pr.arg1 | pr.arg2) > threshold:
                gold_to_predicted_map[gi] = pi
    return gold_to_predicted_map
