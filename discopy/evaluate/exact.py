#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The Official CONLL 2016 Shared Task Scorer

"""
import json

import numpy as np


class ConfusionMatrix(object):
    """Confusion matrix for evaluating a classifier
    For more information on confusion matrix en.wikipedia.org/wiki/Confusion_matrix
    """

    INIT_NUM_CLASSES = 100
    NEGATIVE_CLASS = '__NEGATIVE_CLASS__'

    def __init__(self, alphabet=None):
        if alphabet is None:
            self.alphabet = Alphabet()
            self.matrix = np.zeros((self.INIT_NUM_CLASSES, self.INIT_NUM_CLASSES))
        else:
            self.alphabet = alphabet
            num_classes = alphabet.size()
            self.matrix = np.zeros((num_classes, num_classes))

    def __iadd__(self, other):
        self.matrix += other.matrix
        return self

    def add(self, prediction, true_answer):
        """Add one data point to the confusion matrix
        If prediction is an integer, we assume that it's a legitimate index
        on the confusion matrix.
        If prediction is a string, then we will do the look up to
        map to the integer index for the confusion matrix.
        """
        if type(prediction) == int and type(true_answer) == int:
            self.matrix[prediction, true_answer] += 1
        else:
            self.alphabet.add(prediction)
            self.alphabet.add(true_answer)
            prediction_index = self.alphabet.get_index(prediction)
            true_answer_index = self.alphabet.get_index(true_answer)
            self.matrix[prediction_index, true_answer_index] += 1
            # XXX: this will fail if the prediction_index is greater than
            # the initial capacity. I should grow the matrix if this crashes

    def add_list(self, predictions, true_answers):
        """Add a list of data point to the confusion matrix
        A list can be a list of integers.
        If prediction is an integer, we assume that it's a legitimate index
        on the confusion matrix.
        A list can be a list of strings.
        If prediction is a string, then we will do the look up to
        map to the integer index for the confusion matrix.
        """
        for p, t in zip(predictions, true_answers):
            self.add(p, t)

    def get_prf_for_i(self, i):
        """Compute precision, recall, and f1 score for a given index."""

        if sum(self.matrix[i, :]) == 0:
            precision = 1.0
        else:
            precision = self.matrix[i, i] / sum(self.matrix[i, :])
        if sum(self.matrix[:, i]) == 0:
            recall = 1.0
        else:
            recall = self.matrix[i, i] / sum(self.matrix[:, i])
        if precision + recall != 0.0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        return (precision, recall, f1)

    def get_prf_for_all(self):
        """Compute precision, recall, and f1 score for all indexes."""

        precision = np.zeros(self.alphabet.size())
        recall = np.zeros(self.alphabet.size())
        f1 = np.zeros(self.alphabet.size())

        # compute precision, recall, and f1
        for i in range(self.alphabet.size()):
            precision[i], recall[i], f1[i] = self.get_prf_for_i(i)

        return precision, recall, f1

    def get_prf(self, class_name):
        """Compute precision, recall, and f1 score for a given class. """
        i = self.alphabet.get_index(class_name)
        return self.get_prf_for_i(i)

    def compute_micro_average_f1(self):
        total_correct = 0.0
        for i in range(self.alphabet.size()):
            total_correct += self.matrix[i, i]
        negative_index = self.alphabet.get_index(self.NEGATIVE_CLASS)
        total_predicted = np.sum([x for i, x in enumerate(self.matrix.sum(1)) \
                                  if negative_index == -1 or i != negative_index])
        total_gold = np.sum([x for i, x in enumerate(self.matrix.sum(0)) \
                             if negative_index == -1 or i != negative_index])

        if total_predicted == 0:
            precision = 1.0
        else:
            precision = total_correct / total_predicted
        if total_gold == 0:
            recall = 1.0
        else:
            recall = total_correct / total_gold
        if precision + recall != 0.0:
            f1_score = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        return round(precision, 4), round(recall, 4), round(f1_score, 4)

    def compute_average_f1(self):
        precision, recall, f1 = self.get_prf_for_all()
        return np.mean(f1)

    def compute_average_prf(self):
        precision, recall, f1 = self.get_prf_for_all()
        return (round(np.mean(precision), 4),
                round(np.mean(recall), 4),
                round(np.mean(f1), 4))

    def print_matrix(self):
        num_classes = self.alphabet.size()
        # header for the confusion matrix
        header = [' '] + [self.alphabet.get_label(i) for i in range(num_classes)]
        rows = []
        # putting labels to the first column of rhw matrix
        for i in range(num_classes):
            row = [self.alphabet.get_label(i)] + [str(self.matrix[i, j]) for j in range(num_classes)]
            rows.append(row)
        print("row = predicted, column = truth")
        print(matrix_to_string(rows, header))

    def print_summary(self):

        precision = np.zeros(self.alphabet.size())
        recall = np.zeros(self.alphabet.size())
        f1 = np.zeros(self.alphabet.size())

        max_len = 0
        for i in range(self.alphabet.size()):
            label = self.alphabet.get_label(i)
            if label != self.NEGATIVE_CLASS and len(label) > max_len:
                max_len = len(label)

        lines = []
        correct = 0.0
        # compute precision, recall, and f1
        for i in range(self.alphabet.size()):
            precision[i], recall[i], f1[i] = self.get_prf_for_i(i)
            correct += self.matrix[i, i]
            label = self.alphabet.get_label(i)
            if label != self.NEGATIVE_CLASS:
                space = ' ' * (max_len - len(label) + 1)
                lines.append('%s%s precision %1.4f\trecall %1.4f\tF1 %1.4f' % \
                             (label, space, precision[i], recall[i], f1[i]))
        precision, recall, f1 = self.compute_micro_average_f1()
        space = ' ' * (max_len - 14 + 1)
        lines.append('*Micro-Average%s precision %1.4f\trecall %1.4f\tF1 %1.4f' % (
            space, np.mean(precision), np.mean(recall), np.mean(f1)))
        lines.sort()
        print('\n'.join(lines))

    def print_out(self):
        """Printing out confusion matrix along with Macro-F1 score"""
        self.print_matrix()
        self.print_summary()


class Alphabet(object):
    """Two way map for label and label index
    It is an essentially a code book for labels or features
    This class makes it convenient for us to use numpy.array
    instead of dictionary because it allows us to use index instead of
    label string. The implemention of classifiers uses label index space
    instead of label string space.
    """

    def __init__(self):
        self._index_to_label = {}
        self._label_to_index = {}
        self.num_labels = 0
        self.growing = True

    def __len__(self):
        return self.size()

    def __eq__(self, other):
        return self._index_to_label == other._index_to_label and \
               self._label_to_index == other._label_to_index and \
               self.num_labels == other.num_labels

    def size(self):
        return self.num_labels

    def has_label(self, label):
        return label in self._label_to_index

    def get_label(self, index):
        """Get label from index"""
        if index >= self.num_labels:
            raise KeyError("There are %d labels but the index is %d" % (self.num_labels, index))
        return self._index_to_label[index]

    def get_index(self, label):
        """Get index from label"""
        if not self.has_label(label):
            if self.growing:
                self.add(label)
            else:
                return -1
        return self._label_to_index[label]

    def add(self, label):
        """Add an index for the label if it's a new label"""
        if label not in self._label_to_index:
            if not self.growing:
                raise ValueError(
                    'Alphabet is not set to grow i.e. accepting new labels')
            self._label_to_index[label] = self.num_labels
            self._index_to_label[self.num_labels] = label
            self.num_labels += 1

    def json_dumps(self):
        return json.dumps(self.to_dict())

    @classmethod
    def json_loads(cls, json_string):
        json_dict = json.loads(json_string)
        return Alphabet.from_dict(json_dict)

    def to_dict(self):
        return {
            '_label_to_index': self._label_to_index
        }

    @classmethod
    def from_dict(cls, alphabet_dictionary):
        """Create an Alphabet from dictionary
        alphabet_dictionary is a dictionary with only one field
        _label_to_index which is a map from label to index
        and should be created with to_dict method above.
        """
        alphabet = cls()
        alphabet._label_to_index = alphabet_dictionary['_label_to_index']
        alphabet._index_to_label = {}
        for label, index in alphabet._label_to_index.items():
            alphabet._index_to_label[index] = label
        # making sure that the dimension agrees
        assert (len(alphabet._index_to_label) == len(alphabet._label_to_index))
        alphabet.num_labels = len(alphabet._index_to_label)
        return alphabet


def evaluate_all(gold_relations: dict, predicted_relations: dict):
    results = []
    for doc_id in gold_relations.keys():
        gold_list = gold_relations[doc_id]
        predicted_list = predicted_relations[doc_id]

        connective_cm = evaluate_connectives(gold_list, predicted_list)
        arg1_cm, arg2_cm, rel_arg_cm = evaluate_argument_extractor(gold_list, predicted_list)

        results.append(
            np.array([
                connective_cm.get_prf('yes'),
                arg1_cm.get_prf('yes'),
                arg2_cm.get_prf('yes'),
                rel_arg_cm.get_prf('yes'),
            ])
        )

    results = np.stack(results)
    res_mean, res_std = results.mean(0), results.std(0)

    print('')
    print('================================================')
    print('Evaluation for {} discourse relations:'.format('ALL'))
    print('================================================')
    print('Conn extractor:               P {:06.4} R {:06.4} F1 {:06.4}'.format(*res_mean[0]))
    print('Arg1 extractor:               P {:06.4} R {:06.4} F1 {:06.4}'.format(*res_mean[1]))
    print('Arg2 extractor:               P {:06.4} R {:06.4} F1 {:06.4}'.format(*res_mean[2]))
    print('Concat(Arg1, Arg2) extractor: P {:06.4} R {:06.4} F1 {:06.4}'.format(*res_mean[3]))


    # print('Explicit connectives         : Precision %1.4f Recall %1.4f F1 %1.4f' % connective_cm.get_prf('yes'))
    # print('Arg 1 extractor              : Precision %1.4f Recall %1.4f F1 %1.4f' % arg1_cm.get_prf('yes'))
    # print('Arg 2 extractor              : Precision %1.4f Recall %1.4f F1 %1.4f' % arg2_cm.get_prf('yes'))
    # print('Arg1 Arg2 extractor combined : Precision %1.4f Recall %1.4f F1 %1.4f' % rel_arg_cm.get_prf('yes'))
    # print('Sense classification--------------')
    # sense_cm = evaluate_sense(gold_list, predicted_list)
    # sense_cm.print_summary()
    # print('Overall parser performance --------------')
    # precision, recall, f1 = sense_cm.compute_micro_average_f1()
    # print('Precision %1.4f Recall %1.4f F1 %1.4f' % (precision, recall, f1))

    # results = np.array([
    #     arg1_cm.get_prf('yes'),
    #     arg2_cm.get_prf('yes'),
    #     rel_arg_cm.get_prf('yes'),
    #     [0, 0, 0],
    #     connective_cm.get_prf('yes')
    # ])
    # return results


def matrix_to_string(matrix, header=None):
    """
    Return a pretty, aligned string representation of a nxm matrix.
    This representation can be used to print any tabular data, such as
    database results. It works by scanning the lengths of each element
    in each column, and determining the format string dynamically.
    the implementation is adapted from here
    mybravenewworld.wordpress.com/2010/09/19/print-tabular-data-nicely-using-python/
    Args:
        matrix - Matrix representation (list with n rows of m elements).
        header -  Optional tuple or list with header elements to be displayed.
    Returns:
        nicely formatted matrix string
    """

    if isinstance(header, list):
        header = tuple(header)
    lengths = []
    if header:
        lengths = [len(column) for column in header]

    # finding the max length of each column
    for row in matrix:
        for column in row:
            i = row.index(column)
            column = str(column)
            column_length = len(column)
            try:
                max_length = lengths[i]
                if column_length > max_length:
                    lengths[i] = column_length
            except IndexError:
                lengths.append(column_length)

    # use the lengths to derive a formatting string
    lengths = tuple(lengths)
    format_string = ""
    for length in lengths:
        format_string += "%-" + str(length) + "s "
    format_string += "\n"

    # applying formatting string to get matrix string
    matrix_str = ""
    if header:
        matrix_str += format_string % header
    for row in matrix:
        matrix_str += format_string % tuple(row)

    return matrix_str


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


# def evaluate_sense(gold_list, predicted_list):
#     """Evaluate sense classifier
#
#     The label ConfusionMatrix.NEGATIVE_CLASS is for the relations
#     that are missed by the system
#     because the arguments don't match any of the gold relations.
#     """
#     sense_alphabet = Alphabet()
#     for relation in gold_list:
#         for sense in relation.senses:
#             sense_alphabet.add(sense)
#
#     sense_alphabet.add(ConfusionMatrix.NEGATIVE_CLASS)
#
#     sense_cm = ConfusionMatrix(sense_alphabet)
#     gold_to_predicted_map, predicted_to_gold_map = \
#         _link_gold_predicted(gold_list, predicted_list, spans_exact_matching)
#
#     for i, gold_relation in enumerate(gold_list):
#         gold_sense = gold_relation.senses
#         if i in gold_to_predicted_map:
#             predicted_sense = gold_to_predicted_map[i].senses
#             if predicted_sense in gold_relation.senses:
#                 sense_cm.add(predicted_sense, predicted_sense)
#             else:
#                 if not sense_cm.alphabet.has_label(predicted_sense):
#                     predicted_sense = ConfusionMatrix.NEGATIVE_CLASS
#                 sense_cm.add(predicted_sense, gold_sense)
#         else:
#             sense_cm.add(ConfusionMatrix.NEGATIVE_CLASS, gold_sense)
#
#     for i, predicted_relation in enumerate(predicted_list):
#         if i not in predicted_to_gold_map:
#             predicted_sense = predicted_relation['Sense'][0]
#             if not sense_cm.alphabet.has_label(predicted_sense):
#                 predicted_sense = ConfusionMatrix.NEGATIVE_CLASS
#             sense_cm.add(predicted_sense, ConfusionMatrix.NEGATIVE_CLASS)
#     return sense_cm


def compute_binary_eval_metric(gold_list, predicted_list, matching_fn):
    """Compute binary evaluation metric

    """
    binary_alphabet = Alphabet()
    binary_alphabet.add('yes')
    binary_alphabet.add('no')
    cm = ConfusionMatrix(binary_alphabet)
    matched_predicted = [False for _ in predicted_list]
    for gold_span in gold_list:
        found_match = False
        for i, predicted_span in enumerate(predicted_list):
            if matching_fn(gold_span, predicted_span) and not matched_predicted[i]:
                cm.add('yes', 'yes')
                matched_predicted[i] = True
                found_match = True
                break
        if not found_match:
            cm.add('no', 'yes')
    # Predicted span that does not match with any
    for matched in matched_predicted:
        if not matched:
            cm.add('yes', 'no')
    return cm


def _link_gold_predicted(gold_list, predicted_list, matching_fn):
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
    gold_arg12_list = [(x['DocID'], (x['Arg1']['TokenList'], x['Arg2']['TokenList']))
                       for x in gold_list]
    predicted_arg12_list = [(x['DocID'], (x['Arg1']['TokenList'], x['Arg2']['TokenList']))
                            for x in predicted_list]
    for gi, gold_span in enumerate(gold_arg12_list):
        for pi, predicted_span in enumerate(predicted_arg12_list):
            if matching_fn(gold_span, predicted_span):
                gold_to_predicted_map[gi] = predicted_list[pi]
                predicted_to_gold_map[pi] = gold_list[gi]
    return gold_to_predicted_map, predicted_to_gold_map
