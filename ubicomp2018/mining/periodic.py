from __future__ import division
import numpy as np
import matlab.engine
import logging
from beyourself.core.util import assert_monotonic, assert_vector, epoch_to_datetime


logger = logging.getLogger(__name__)


def moving_variance_threshold(df, window_minutes=5, threshold=0):
    '''
    Remove static time of usage based on the moving variance
    of ambient light and quaternion
    '''
    logger.info("N sample before removal: {}".format(df.shape[0]))

    df = df.copy()
  

    var_ambient = df['ambient'].rolling(center=True, window=20*60*window_minutes).std()
    var_lf = df['leanForward'].rolling(center=True, window=20*60*window_minutes).std()

    out = df.loc[(var_ambient + var_lf) > threshold]

    logger.info("N sample after removal: {}".format(out.shape[0]))

    return out



def peak_detection(signal, min_prominence=0.05):
    '''
    Prominence based
    '''
    logger.info("matlab findpeaks() ...")
    matlab_engine = matlab.engine.start_matlab()
    mat_signal = matlab.double(list(signal))
    _, peaks_index = matlab_engine.findpeaks(mat_signal, 'MinPeakProminence', min_prominence, 'Threshold',2, nargout=2)
    peaks_index = np.asarray(peaks_index)[0].astype(int) - 1
    matlab_engine.exit()

    logger.info("matlab done")
    logger.info("Number of peaks: {}".format(len(peaks_index)))


    return peaks_index


def get_periodic_stat(a):
    '''
    Given a sequence, find its periodicity statistics

    Returns
    -------

    stat: dictionary
        pmin
        pmax
        eps
        length

    '''

    assert_vector(a)
    assert_monotonic(a)

    stat = {}

    diff = []
    for i in range(len(a) - 1):
        diff.append(a[i + 1] - a[i])

    stat['pmin'] = min(diff)
    stat['pmax'] = max(diff)
    stat['eps'] = stat['pmax'] / stat['pmin'] - 1
    stat['start'] = epoch_to_datetime(a[0])
    stat['end'] = epoch_to_datetime(a[-1])
    stat['length'] = len(a)

    return stat


def periodic_subsequence(peaks_index, peaks_time, min_length=5, max_length=100, eps=0.15, alpha=0.1, low=500, high=1000):
    '''
    Find periodic subsequences from an array of timestamp, and values

    Parameters
    ----------

    time: list of timestamps
    value: list of sensor value
    peak_neighbor: size of the neighborhood
    min_length: minimum length of the subsequence
    eps: periodicity
    low: lower bound for p_min
    high: upper bound for p_max

    Returns
    -------

    subsequences: a list of numpy vector
        each vector is one subsequence
        contains the index of periodic peaks

    '''

    assert_vector(peaks_time)
    assert_monotonic(peaks_time)

    subs_index = relative_error_periodic_subsequence(
        peaks_time, eps, alpha, low, high, min_length, max_length)

    subsequences = []
    for s in subs_index:
        tmp = [peaks_index[i] for i in s]
        subsequences.append(np.array(tmp))

    return subsequences


def relative_error_periodic_subsequence(a, eps, alpha, low, high, min_length, max_length):
    '''
    Approximation algorithm that find eps-periodic subsequences
    '''

    assert_vector(a)
    assert_monotonic(a)

    subsequences = []

    n_steps = np.ceil(np.log(high / low) /
                      np.log(1 + eps)).astype(int)
    for i in range(n_steps):
        pmin = low * np.power((1 + eps), i)
        pmax = pmin * (1 + eps) * (1+alpha)

        if pmax > high:
            break

        logger.info("pmin {:0.2f} and pmax {:0.2f}".format(pmin, pmax))

        seqs = absolute_error_periodic_subsequence(a, pmin, pmax)
        seqs = [np.array(s) for s in seqs if len(s) > min_length and len(s) < max_length]

        subsequences += seqs

    # sort subsequences by its start time
    start = [seq[0] for seq in subsequences]

    subsequences = [seq for _, seq in
                    sorted(zip(start, subsequences), key=lambda pair:pair[0])]

    return subsequences


def absolute_error_periodic_subsequence(a, pmin, pmax):
    '''
    Return longest subsequences that is periodic
    Dynamic programming approach

    Parameters
    ----------

    a: list of increasing numbers

    '''

    assert_vector(a)
    assert_monotonic(a)

    N = len(a)

    traceback = {}
    for i in range(N):
        traceback[i] = []

    for i in range(1, N):

        valid = []
        for j in range(i - 1, -1, -1):
            if a[i] - a[j] > pmax:
                break
            if a[i] - a[j] >= pmin:
                valid.append(j)

        valid = list(reversed(valid))

        # now find valid predecessor for i
        for j in valid:

            if not traceback[j]:
                L = 2
            else:
                L = traceback[j][0]['L'] + 1

            predecessor = {'prev': j, 'L': L}

            tobe_kept = []
            for k in range(len(traceback[i])):
                if traceback[i][k]['L'] >= predecessor['L']:
                    tobe_kept.append(k)

            traceback[i] = [traceback[i][k] for k in tobe_kept]
            traceback[i].append(predecessor)

        # logger.debug(traceback[i])

    subsequences = []
    sequence = []
    i = N - 1
    while i >= 0:
        if traceback[i]:
            sequence.append(i)
            i = traceback[i][0]['prev']
        else:
            if len(sequence) > 0:
                sequence.append(i)
                reverse = list(reversed(sequence))
                subsequences.append(reverse)
                sequence = []
            i -= 1

    return list(reversed(subsequences))
