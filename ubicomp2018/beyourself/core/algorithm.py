from __future__ import division
import logging
import pandas as pd
import numpy as np
from intervaltree import Interval, IntervalTree
from beyourself.settings import ABSOLUTE_TIME_FORMAT
from datetime import datetime, timedelta
from beyourself.core.util import *
import numbers


logger = logging.getLogger(__name__)


def _intersect(interval_a, interval_b):
    """
    Calculate the intersection between two tuples
    Represented by datetime objects

    Parameters
    ----------
    interval_a, interval_b: tuple of datetime objects, or tuples of int, or Interval object

    Returns
    -------
    out: intersecting tuple
    """

    if interval_b[0] > interval_a[1] or interval_a[0] > interval_b[1]:
        raise ValueError("Intervals do not intersect")

    return max(interval_a[0], interval_b[0]), min(interval_a[1], interval_b[1])


def _get_overlap(a, b):
    """
    Given two pairs a and b,
    find the intersection between them
    could be either number or datetime objects
    """

    tmp = min(a[1], b[1]) - max(a[0], b[0])
    
    if isinstance(tmp, timedelta):
        zero_value = timedelta(seconds=0)
    else:
        zero_value = 0
    
    return max(zero_value, tmp)


def _get_sum(segments):
 
    diffs = [(s[1] - s[0]) for s in segments]

    print(type(diffs[0]))

    if isinstance(diffs[0], timedelta):
        logger.info("datetime object")
        out = timedelta(seconds=0)
        for diff in diffs:
            out += diff
        return out
    
    elif isinstance(diffs[0], numbers.Integral):
        logger.info("integer")
        out = 0
        for diff in diffs:
            out += diff
        return out

    else:
        raise ValueError("Invalid type of segments")


def interval_intersect_interval(**kwargs):
    """
    Efficient algorithm to find which intervals intersect

    Handles both unix timestamp or datetime object

    Return:
    -------

    prediction_gt:
        array with same size as prediction,
        will be 1 if there's an overlapping label
        0 if not
    recall:
        recall percentage of labels
    overlap:
        how much overlap between label and prediction
    """

    gt = kwargs['groundtruth']
    pred = kwargs['prediction']

    # calculate recall
    tree = IntervalTree()
    for segment in pred:
        tree.add(Interval(segment[0],segment[1]))

    recall_gt = []
    for segment in gt:
        overlap = tree.search(segment[0], segment[1])
        if len(overlap) != 0:
            recall_gt.append(1)
        else:
            recall_gt.append(0)

    recall = np.mean(recall_gt)

    # calculate precision
    tree = IntervalTree()
    for segment in gt:
        tree.add(Interval(segment[0],segment[1]))

    prediction_gt = []
    for segment in pred:
        overlap = tree.search(segment[0], segment[1])
        if len(overlap) != 0:
            prediction_gt.append(1)
        else:
            prediction_gt.append(0)

    result = {'prediction_gt': prediction_gt,
              'recall_gt': recall_gt,
              'recall': recall,
              'precision': np.mean(prediction_gt)}

    return result


def point_intersect_interval(points, df_interval):
    """
    Expect both points and df_interval to be datetime object
    """

    # store index of intervals as value of the interval
    tree = IntervalTree()
    for i in range(df_interval.shape[0]):
        tree[df_interval['start'].iloc[i]:df_interval['end'].iloc[i]] = i

    points_gt = np.zeros_like(points).astype(bool)
    interval_gt = [False] * df_interval.shape[0]

    for i in range(len(points)):
        intersection = tree.search(points[i])
        if len(intersection) == 0:
            points_gt[i] = False
        else:
            points_gt[i] = True
    
            for segment in intersection:
                interval_gt[segment.data] = True

    results = {'points_gt': points_gt,
               'interval_gt': interval_gt}
    return results


def get_overlapping_intervals(ranges_a, ranges_b):
    """
    Return a list of overlapping intervals
    """
    if len(ranges_a) < len(ranges_b):
        longer = ranges_b
        shorter = ranges_a
    else:
        longer = ranges_a
        shorter = ranges_b

    tree = IntervalTree()
    for s in longer:
        tree.add(Interval(s[0], s[1]))

    overlap = []
    for seg in shorter:
        overlap += [_intersect(s, seg) for s in tree.search(seg[0], seg[1])]

    return sorted(overlap)


def _assert_non_overlap(ranges):
    ranges = sorted(ranges)
    for i in range(len(ranges) - 1):
        if ranges[i][1] > ranges[i+1][0]:
            raise ValueError("Intervals are overlapping: {}, {}".format(ranges[i][1], ranges[i+1][0]))