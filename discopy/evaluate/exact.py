import numpy as np
import sklearn


def print_results(res_mean, mode):
    print('')
    print('================================================')
    print('Evaluation for {} discourse relations:'.format(mode))
    print('================================================')
    print('Conn extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*res_mean[0]))
    print('Arg1 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*res_mean[1]))
    print('Arg2 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*res_mean[2]))
    print('Concat(Arg1, Arg2) extractor: P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*res_mean[3]))
    print('Sense:                        P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*res_mean[4]))


def evaluate_all(gold_relations: dict, predicted_relations: dict):
    results = []
    for doc_id in gold_relations.keys():
        gold_list = gold_relations[doc_id]
        predicted_list = predicted_relations[doc_id]

        connective_cm = evaluate_connectives(gold_list, predicted_list)
        arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list)
        sense_cm = evaluate_sense(gold_list, predicted_list)

        results.append(
            np.array([
                connective_cm,
                arg1_cm,
                arg2_cm,
                rel_arg_cm,
                sense_cm,
            ])
        )

    results = np.stack(results)
    res_mean, res_std = results.mean(0), results.std(0)
    print_results(res_mean, 'ALL')

    results = []
    for doc_id in gold_relations.keys():
        gold_list = [r for r in gold_relations[doc_id] if r.is_explicit()]
        predicted_list = [r for r in predicted_relations[doc_id] if r.is_explicit()]

        connective_cm = evaluate_connectives(gold_list, predicted_list)
        arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list)
        sense_cm = evaluate_sense(gold_list, predicted_list)

        results.append(
            np.array([
                connective_cm,
                arg1_cm,
                arg2_cm,
                rel_arg_cm,
                sense_cm,
            ])
        )

    results = np.stack(results)
    res_mean, res_std = results.mean(0), results.std(0)
    print_results(res_mean, 'EXPLICIT')

    results = []
    for doc_id in gold_relations.keys():
        gold_list = [r for r in gold_relations[doc_id] if not r.is_explicit()]
        predicted_list = [r for r in predicted_relations[doc_id] if not r.is_explicit()]

        connective_cm = evaluate_connectives(gold_list, predicted_list)
        arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list)
        sense_cm = evaluate_sense(gold_list, predicted_list)

        results.append(
            np.array([
                connective_cm,
                arg1_cm,
                arg2_cm,
                rel_arg_cm,
                sense_cm,
            ])
        )

    results = np.stack(results)
    res_mean, res_std = results.mean(0), results.std(0)
    print_results(res_mean, 'NON-EXPLICIT')


def evaluate_argument_extractor(gold_list, predicted_list):
    """Evaluate argument extractor at Arg1, Arg2, and relation level

    """
    gold_arg1 = [r.arg1 for r in gold_list]
    predicted_arg1 = [r.arg1 for r in predicted_list]
    arg1_cm = compute_binary_eval_metric(gold_arg1, predicted_arg1, span_exact_matching)

    gold_arg2 = [r.arg2 for r in gold_list]
    predicted_arg2 = [r.arg2 for r in predicted_list]
    arg2_cm = compute_binary_eval_metric(gold_arg2, predicted_arg2, span_exact_matching)

    gold_arg12 = [arg1 | arg2 for arg1, arg2 in zip(gold_arg1, gold_arg2)]
    predicted_arg12 = [arg1 | arg2 for arg1, arg2 in zip(predicted_arg1, predicted_arg2)]
    rel_arg_cm = compute_binary_eval_metric(gold_arg12, predicted_arg12, span_exact_matching)
    return arg1_cm, arg2_cm, rel_arg_cm


def evaluate_connectives(gold_list, predicted_list):
    """Evaluate connective recognition accuracy for explicit discourse relations

    """
    explicit_gold_list = [r.conn for r in gold_list if r.is_explicit()]
    explicit_predicted_list = [r.conn for r in predicted_list if r.is_explicit()]
    connective_cm = compute_binary_eval_metric(
        explicit_gold_list, explicit_predicted_list, connective_head_matching)
    return connective_cm


def spans_exact_matching(gold_spans, predicted_spans):
    """Matching two lists of spans

    Input:
        gold_doc_id_spans : (DocID , a list of lists of tuples of token addresses)
        predicted_doc_id_spans : (DocID , a list of lists of token indices)

    Returns:
        True if the spans match exactly
    """
    exact_match = True

    for gold_span, predicted_span in zip(gold_spans, predicted_spans):
        exact_match = span_exact_matching(gold_span, predicted_span) and exact_match
    return exact_match


def span_exact_matching(gold_span, predicted_span):
    """Matching two spans

    Input:
        gold_span : a list of tuples :(DocID, list of tuples of token addresses)
        predicted_span : a list of tuples :(DocID, list of token indices)

    Returns:
        True if the spans match exactly
    """
    return gold_span == predicted_span


# TODO changed because of different implementation of relations that stores only indices instead of words
# use F1 overlapping score instead of the original score, which leads to minor changes in the final result
# misses connective head mapping
def connective_head_matching(gold_connective: set, predicted_connective):
    """Matching connectives

    Input:
        gold_connective : indices of gold connectives
        predicted_connective : indices of predicted connectives

    A predicted raw connective is considered iff
        1) the predicted raw connective includes the connective "head"
        2) the predicted raw connective tokens are the subset of predicted raw connective tokens

    For example:
        connective_head_matching('two weeks after', 'weeks after')  --> True
        connective_head_matching('two weeks after', 'two weeks')  --> False not covering head
        connective_head_matching('just because', 'because')  --> True
        connective_head_matching('just because', 'simply because')  --> False not subset
        connective_head_matching('just because', 'since')  --> False
    """

    def compute_f1_span(g_index_set: set, p_index_set: set) -> float:
        """Compute F1 score for a given pair of token list"""
        correct = float(len(g_index_set.intersection(p_index_set)))
        if correct == 0.0:
            return 0.0
        precision = correct / len(p_index_set)
        recall = correct / len(g_index_set)
        return 2 * (precision * recall) / (precision + recall)

    if gold_connective == predicted_connective:
        return True
    else:
        return compute_f1_span(gold_connective, predicted_connective) > 0.7


def evaluate_sense(gold_list, predicted_list):
    """Evaluate sense classifier

    The label ConfusionMatrix.NEGATIVE_CLASS is for the relations
    that are missed by the system
    because the arguments don't match any of the gold relations.
    """
    y_true, y_pred = [], []
    gold_to_predicted_map, _ = _link_gold_predicted(gold_list, predicted_list)
    for gi, gr in enumerate(gold_list):
        if gi in gold_to_predicted_map:
            senses = [s for s in predicted_list[gold_to_predicted_map[gi]].senses if s in gr.senses]
            if senses:
                y_true.append(senses[0])
                y_pred.append(senses[0])
            else:
                y_true.append(gr.senses[0])
                y_pred.append(predicted_list[gold_to_predicted_map[gi]].senses[0])
        else:
            y_true.append(gr.senses[0])
            y_pred.append('NO_MATCH')

    if not y_true:
        return 0.0, 0.0, 0.0
    else:
        precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(np.array(y_true),
                                                                                         np.array(y_pred),
                                                                                         average='macro')
    return precision, recall, f1


def compute_binary_eval_metric(gold_list, predicted_list, matching_fn):
    """Compute binary evaluation metric

    """
    tp = fp = fn = 0
    matched_predicted = [False for _ in predicted_list]
    for gold_span in gold_list:
        found_match = False
        for i, predicted_span in enumerate(predicted_list):
            if matching_fn(gold_span, predicted_span) and not matched_predicted[i]:
                tp += 1
                matched_predicted[i] = True
                found_match = True
                break
        if not found_match:
            fp += 1
    # Predicted span that does not match with any
    for matched in matched_predicted:
        if not matched:
            fn += 1

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def _link_gold_predicted(gold_list, predicted_list):
    """Link gold standard relations to the predicted relations

    A pair of relations are linked when the arg1 and the arg2 match exactly.
    We do this because we want to evaluate sense classification later.

    Returns:
        A tuple of two dictionaries:
        1) mapping from gold relation index to predicted relation index
        2) mapping from predicted relation index to gold relation index
    """
    gold_to_predicted_map = {}
    predicted_to_gold_map = {}

    for gi, gr in enumerate(gold_list):
        for pi, pr in enumerate(predicted_list):
            if span_exact_matching(gr.arg1 | gr.arg2, pr.arg1 | pr.arg2):
                gold_to_predicted_map[gi] = pi
                predicted_to_gold_map[pi] = gi
    return gold_to_predicted_map, predicted_to_gold_map
