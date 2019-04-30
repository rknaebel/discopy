import numpy as np


def print_results(results, mode):
    print('================================================')
    print('Evaluation for {} discourse relations:'.format(mode))
    print('================================================')
    print('Conn extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[0])))
    print('Arg1 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[1])))
    print('Arg2 extractor:               P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[2])))
    print('Concat(Arg1, Arg2) extractor: P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[3])))
    print('Sense:                        P {:<06.4} R {:<06.4} F1 {:<06.4}'.format(*compute_prf(*results[4])))


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
    results = np.stack(results).sum(axis=0)
    print_results(results, 'ALL')

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
    results = np.stack(results).sum(axis=0)
    print_results(results, 'EXPLICIT')

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
    results = np.stack(results).sum(axis=0)
    print_results(results, 'NON-EXPLICIT')


def evaluate_argument_extractor(gold_list, predicted_list):
    gold_arg1 = [r.arg1 for r in gold_list]
    predicted_arg1 = [r.arg1 for r in predicted_list]
    arg1_cm = compute_confusion_counts(gold_arg1, predicted_arg1, span_almost_exact_matching)

    gold_arg2 = [r.arg2 for r in gold_list]
    predicted_arg2 = [r.arg2 for r in predicted_list]
    arg2_cm = compute_confusion_counts(gold_arg2, predicted_arg2, span_almost_exact_matching)

    gold_arg12 = [arg1 | arg2 for arg1, arg2 in zip(gold_arg1, gold_arg2)]
    predicted_arg12 = [arg1 | arg2 for arg1, arg2 in zip(predicted_arg1, predicted_arg2)]
    rel_arg_cm = compute_confusion_counts(gold_arg12, predicted_arg12, span_almost_exact_matching)
    return arg1_cm, arg2_cm, rel_arg_cm


def evaluate_connectives(gold_list, predicted_list):
    explicit_gold_list = [r.conn for r in gold_list if r.is_explicit()]
    explicit_predicted_list = [r.conn for r in predicted_list if r.is_explicit()]
    connective_cm = compute_confusion_counts(
        explicit_gold_list, explicit_predicted_list, span_almost_exact_matching)
    return connective_cm


def span_almost_exact_matching(gold_span, predicted_span):
    return compute_span_f1(gold_span, predicted_span) > 0.95


def compute_span_f1(g_index_set: set, p_index_set: set) -> float:
    correct = len(g_index_set.intersection(p_index_set))
    if correct == 0:
        return 0.0
    precision = float(correct) / len(p_index_set)
    recall = float(correct) / len(g_index_set)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_sense(gold_list, predicted_list):
    tp = fp = fn = 0
    gold_to_predicted_map = _link_gold_predicted(gold_list, predicted_list)
    for gi, gr in enumerate(gold_list):
        if gi in gold_to_predicted_map:
            if any(g in predicted_list[gold_to_predicted_map[gi]].senses for g in gr.senses):
                tp += 1
            else:
                fp += 1
        else:
            fn += 1

    return np.array([tp, fp, fn])


def compute_confusion_counts(gold_list, predicted_list, matching_fn):
    tp = fp = 0
    unmatched = np.ones(len(predicted_list), dtype=bool)
    for gold_span in gold_list:
        for i, predicted_span in enumerate(predicted_list):
            if matching_fn(gold_span, predicted_span) and unmatched[i]:
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


def _link_gold_predicted(gold_list, predicted_list):
    """
    Link gold relations to the predicted relations that fits best
    based on the almost exact matching criterion
    """
    gold_to_predicted_map = {}

    for gi, gr in enumerate(gold_list):
        for pi, pr in enumerate(predicted_list):
            if span_almost_exact_matching(gr.arg1 | gr.arg2, pr.arg1 | pr.arg2):
                gold_to_predicted_map[gi] = pi
    return gold_to_predicted_map
