
import numpy as np
from collections import defaultdict
from label_embedding import TypeHierarchy
from sklearn import metrics


def compute_matches(set_a, set_b):
    count = 0
    for item in set_a:
        if item in set_b:
            count += 1
    return count


def get_if_perfect_match(set_a, set_b):
    if len(set_a) == len(set_b):
        for item in set_a:
            if item not in set_b:
                return False
        return True
    return False


def compute_f1(precision, recall):
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(prediction, ground_truth, is_all=True):
    """
    evaluate the prediction with the assessment on strict accuracy, Micro-F1, Macro-F1
    """
    assert len(prediction) == len(ground_truth)
    if not isinstance(prediction[0], list):            
        prediction = [[x] for x in prediction]
    count = len(prediction)
    total_macro_precision = 0.0
    total_macro_recall = 0.0
    perfect_match = 0
    total_matches = 0
    total_ground_types = 0
    total_predicted_types = 0
    for i in range(len(ground_truth)):
        p = set(prediction[i])
        if is_all:
            g = set(np.where(ground_truth[i] == 1)[0])
        else:
            g = set(ground_truth)
        matches = compute_matches(p, g)
        total_matches += matches
        total_ground_types += len(g)
        total_predicted_types += len(p)
        if len(p) > 0:
            total_macro_precision += float(matches) / float(len(p))
        if len(g) > 0:
            total_macro_recall += float(matches) / float(len(g))
        if get_if_perfect_match(g, p):
            perfect_match += 1

    strict_accuracy = 0.0
    if count > 0:
        strict_accuracy = float(perfect_match) / float(count)

    micro_precision = 0.0
    if total_predicted_types > 0.0:
        micro_precision = float(total_matches) / float(total_predicted_types)
    micro_recall = 0.0
    if total_ground_types > 0.0:
        micro_recall = float(total_matches) / float(total_ground_types)
    micro_f1 = compute_f1(micro_precision, micro_recall)

    macro_precision = 0.0
    macro_recall = 0.0
    if count > 0:
        macro_precision = float(total_macro_precision) / float(count)
        macro_recall = float(total_macro_recall) / float(count)
    macro_f1 = compute_f1(macro_precision, macro_recall)

    return strict_accuracy, micro_f1,  macro_f1, micro_precision, micro_recall, macro_precision, macro_recall

    # print("=========Performance==========")
    # print("Strict Accuracy:{:.5f}" .format(strict_accuracy))
    # print("---------------")
    # print("Micro Precision:{:.5f}" .format(micro_precision))
    # print("Micro Recall:{:.5f}" .format(micro_recall))
    # print("Micro F1:{:.5f}".format(micro_f1))
    # print("---------------")
    # print("Macro Precision:{:.5f}".format(macro_precision))
    # print("Macro Recall:{:.5f}".format(macro_recall))
    # print("Macro F1:{:.5f}".format(macro_f1))
    # print("==============================")


def load_raw_labels(super_type_file, prediction):
    """
    given a child label in the prediction, include its parent label into the prediction.
    """
    type_hierarchy = TypeHierarchy(super_type_file)
    pred_update = []
    for i in range(len(prediction)):
        id = prediction[i]
        pred_update.append(type_hierarchy.get_type_path(id))
    return pred_update


def evaluation_level(prediction, ground_truth):
    """
    give the regular evaluation matrix, such as: precision, recall, f1, accuracy.
    """
    g = []
    for i in range(len(ground_truth)):
        g.append(list(np.where(ground_truth[i] == 1)[0])[0])
    ground_truth = g
    accuracy = metrics.accuracy_score(ground_truth, prediction)

    precision_micro = metrics.precision_score(ground_truth, prediction, average='micro')
    precision_macro = metrics.precision_score(ground_truth, prediction, average='macro')

    recall_micro = metrics.recall_score(ground_truth, prediction, average="micro")
    recall_macro = metrics.recall_score(ground_truth, prediction, average="macro")

    f1_micro = metrics.f1_score(ground_truth, prediction, average="micro")
    f1_macro = metrics.f1_score(ground_truth, prediction, average="macro")

    metrics.classification_report(ground_truth, prediction)     # report for all evaluation aforementioned
    metrics.cohen_kappa_score(ground_truth, prediction)         # >0.8 classification has good quality

    return accuracy, f1_micro, f1_macro, precision_micro, recall_micro, precision_macro, recall_macro


