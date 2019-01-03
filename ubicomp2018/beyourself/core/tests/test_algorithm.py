from beyourself.core.algorithm import *
from beyourself.core.algorithm import _intersect
from numpy.testing import assert_equal
import pandas as pd
from datetime import datetime


# def test_interval_intersect():

#     gt = [(1,2),(2,3),(5,6)]

#     pred = [(1.5,2.5)]

#     print(interval_intersect_interval(prediction=pred, groundtruth=gt))


# def test_interval_intersect_datetime():

#     gt = [ (datetime(2013, 6, 9, 11, 13),\
#             datetime(2013, 6, 9, 11, 19)),\
#            (datetime(2013, 6, 9, 11, 30),\
#             datetime(2013, 6, 9, 12, 19))]

#     pred = [(datetime(2013, 6, 9, 11, 15),\
#             datetime(2013, 6, 9, 11, 18))]


#     print(interval_intersect_interval(prediction=pred, groundtruth=gt))


# def test_point_intersect():
#     start = [0, 4, 5, 7]
#     end = [2, 4.5, 6, 10]
#
#     df = pd.DataFrame({'start': start, 'end': end}, columns=['start', 'end'])
#
#     print(df)
#
#     points = np.array([1, 3, 5, 7, 9])
#     print(points)
#
#     print(point_intersect_interval(points, df))


def test_intersect():
    start = (0, 1)
    end = (0.5, 2)

    overlap = _intersect(start, end)

    assert overlap[0] == 0.5
    assert overlap[1] == 1


def test_overlapping_interval():
    start = [[0, 1], [2, 3], [4, 5], [6, 7]]
    end = [[0.5, 1], [2.5, 3], [4.5, 5], [6.5, 7]]

    print(get_overlapping_intervals(start, end))

    start = [[0, 1], [2, 3], [4, 5], [6, 7]]
    end = [[0.5, 1], [2.5, 3.5], [4.5, 5], [6.5, 7]]

    print(get_overlapping_intervals(start, end))