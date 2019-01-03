import os
import logging
import json
from datetime import timedelta
import numpy as np
import pandas as pd
from numpy.random import randint
from beyourself.settings import MAX_LENGTH_SEC, BATCH_SIZE, N_CHUNK, WIN, N_SENSOR, CLEAN_FOLDER, MAX_LENGTH
from beyourself.core.util import datetime_from_str, humanstr_withtimezone_to_datetime_df, datetime_to_epoch
from beyourself.core.algorithm import get_overlapping_intervals, _assert_non_overlap
from beyourself.data.sensordata import get_necklace
import pickle
import random

logger = logging.getLogger(__name__)


def _random_datetime(start, end):
    """
    Generate a single random datetime between `start` and `end`
    """

    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))


def _random_datetime_interval(start: object, end: object, duration: object) -> object:
    if end - start < duration:
        logger.warning("Duration is too large")
        return start, end

    a = _random_datetime(start, end - duration)
    b = a + duration

    return a, b


class SequentialTrainChewingData:

    def __init__(self, json_path, key_list):

        self.validation_folds = []
        self.key_list = key_list
        self.subj = 'P120'

        # Ranges of validation sets
        with open(json_path) as f:
            data = json.load(f)
            for k, v in sorted(data.items()):
                self.validation_folds.append((datetime_from_str(v['start']),
                                              datetime_from_str(v['end'])))
        _assert_non_overlap(self.validation_folds)

        self.toggleNegativeSequence = False

        # Ranges where data is continuous
        total_continuous_ranges = pickle.load(
            open(os.path.join(CLEAN_FOLDER, '{}/label/continuous.pickle'.format(self.subj)), 'rb'))
        self.continuous_ranges = get_overlapping_intervals(total_continuous_ranges, self.validation_folds)

        # Ranges of chewing ground truth
        df_chewing = pd.read_csv(os.path.join(CLEAN_FOLDER, '{}/label/chewing.csv'.format(self.subj)))
        df_chewing = humanstr_withtimezone_to_datetime_df(df_chewing, ['start', 'end'])
        total_gt_ranges = list(zip(df_chewing['start'].tolist(),
                                   df_chewing['end'].tolist()))
        self.gt_ranges = get_overlapping_intervals(total_gt_ranges, self.validation_folds)

        # for querying chewing
        total_gt_extended_ranges = list(zip((df_chewing['start'] - timedelta(seconds=0.5*MAX_LENGTH_SEC)).tolist(),
                                   (df_chewing['end'] + timedelta(seconds=0.5*MAX_LENGTH_SEC)).tolist()))
        query_gt_ranges = get_overlapping_intervals(total_gt_extended_ranges, self.validation_folds)
        self.query_gt_ranges = get_overlapping_intervals(query_gt_ranges, self.continuous_ranges)

    def __iter__(self):
        return self

    def __next__(self):

        batch_x = np.zeros((BATCH_SIZE, N_CHUNK, WIN * N_SENSOR))
        batch_y = np.zeros((BATCH_SIZE, N_CHUNK))

        min_duration = timedelta(seconds=MAX_LENGTH_SEC)

        if self.toggleNegativeSequence:
            logger.info("Generate non-chewing sequence")
            # make sure that data contains at least one MAX_LENGTH sequence
            while True:
                fold = self.continuous_ranges[random.randint(0, len(self.continuous_ranges) - 1)]
                if fold[1] - fold[0] > min_duration:
                    break
        else:
            logger.info("Generate chewing sequence")
            # make sure that data contains at least one MAX_LENGTH sequence
            while True:
                fold = self.gt_ranges[random.randint(0, len(self.gt_ranges) - 1)]
                if fold[1] - fold[0] > min_duration:
                    break

        # get random data into memory
        random_interval = _random_datetime_interval(fold[0], fold[1], timedelta(seconds=4*MAX_LENGTH_SEC))
        df = get_necklace(self.subj, datetime_to_epoch(random_interval[0]), datetime_to_epoch(random_interval[1]))

        # get point-wise label
        df['label_chewing'] = np.zeros((df.shape[0],))
        for seg in self.gt_ranges:
            df['label_chewing'][(df.index >= seg[0]) & (df.index <= seg[1])] = 1
        label_chewing = df['label_chewing'].as_matrix()

        for b in range(BATCH_SIZE):
            start = randint(0, df.shape[0] - MAX_LENGTH)

            # slicing data
            x = df[self.key_list].iloc[start:start + MAX_LENGTH].as_matrix()
            b_x = np.reshape(x, (N_CHUNK, WIN * N_SENSOR))
            batch_x[b, :, :] = b_x

            # downsample label
            query_chewing = label_chewing[start:start + MAX_LENGTH]
            batch_y[b, :] = query_chewing[(WIN // 2)::WIN]

        self.toggleNegativeSequence = not self.toggleNegativeSequence

        return batch_x.astype(float), batch_y.astype(int)


class SequentialTestChewingData(object):

    def __init__(self, json_path, key_list):

        self.validation_folds = []
        self.key_list = key_list
        self.subj = "P120"

        with open(json_path) as f:
            data = json.load(f)
            for k, v in sorted(data.items()):
                self.validation_folds.append((datetime_from_str(v['start']),
                                              datetime_from_str(v['end'])))
        # Ranges of chewing ground truth
        df_chewing = pd.read_csv(os.path.join(CLEAN_FOLDER, '{}/label/chewing.csv'.format(self.subj)))
        df_chewing = humanstr_withtimezone_to_datetime_df(df_chewing, ['start', 'end'])
        total_gt_ranges = list(zip(df_chewing['start'].tolist(),
                                   df_chewing['end'].tolist()))
        self.gt_ranges = get_overlapping_intervals(total_gt_ranges, self.validation_folds)

        # Ranges where data is continuous
        total_continuous_ranges = pickle.load(
            open(os.path.join(CLEAN_FOLDER, '{}/label/continuous.pickle'.format(self.subj)), 'rb'))
        continuous_ranges = get_overlapping_intervals(total_continuous_ranges, self.validation_folds)

        self.test_folds = []
        for fold in sorted(continuous_ranges):
            previous = fold[0]
            while True:
                current = previous + timedelta(seconds=BATCH_SIZE * MAX_LENGTH_SEC)
                if current > fold[1]:
                    break
                self.test_folds.append((previous, current))
                previous = current

            # if previous < fold[1]:
            #     self.test_folds.append((previous, fold[1]))
        logger.info("Length of test folds: {}".format(len(self.test_folds)))

        self.counter = 0

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter > len(self.test_folds) - 1:
            raise StopIteration

        fold = self.test_folds[self.counter]

        print(fold[1] - fold[0])

        self.counter = self.counter + 1

        # slice continuous folds and chewing folds
        df = get_necklace("P120", datetime_to_epoch(fold[0]), datetime_to_epoch(fold[1]))

        df['label_chewing'] = np.zeros((df.shape[0],))
        for c in self.gt_ranges:
            df['label_chewing'][(df.index >= c[0]) & (df.index <= c[1])] = 1
        label_chewing = df['label_chewing'].as_matrix()

        N = df.shape[0]
        N_even = ((N - 1) // (BATCH_SIZE * WIN)) * BATCH_SIZE * WIN
        logger.debug("Number of samples before truncation: {} and after: {}".format(N, N_even))

        # slicing
        df = df.iloc[:N_even]
        label_chewing = label_chewing[:N_even]

        batch_x = df[self.key_list].as_matrix()
        batch_x = np.reshape(batch_x, (BATCH_SIZE, -1, WIN * N_SENSOR))

        # majority voting
        label_chewing = np.reshape(label_chewing, (-1, WIN))
        label_sum_chewing = np.sum(label_chewing, axis=1)
        batch_y_chewing = label_sum_chewing > WIN // 2
        batch_y_chewing = np.reshape(batch_y_chewing, (BATCH_SIZE, -1))

        return batch_x.astype(float), batch_y_chewing.astype(int)
