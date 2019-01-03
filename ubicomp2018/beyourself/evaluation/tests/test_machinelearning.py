import pandas as pd
from beyourself.evaluation.ml import metrics_evaluate
import numpy as np


def test_case_1():
    labels = pd.DataFrame(data = np.array([1, 1, 1, 2, 2, 2]))
    pred = pd.DataFrame(data = np.array([1, 1, 2, 2, 1, 1]))
    metrics = metrics_evaluate(labels=labels, pred=pred, target_class=1)

    print(metrics)


def test_case_2():
    labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 2, 1, 1])
    metrics = metrics_evaluate(labels=labels, pred=pred, target_class=1)

    print(metrics)


def test_case_3():
    labels = [1, 1, 1, 2, 2, 2, 3, 3]
    pred = [1, 1, 2, 2, 1, 1, 3, 3]
    metrics = metrics_evaluate(labels=labels, pred=pred, target_class=1)

    print(metrics['f1_pos'])
    print(metrics['confusion_matrix'])