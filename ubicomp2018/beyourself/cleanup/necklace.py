from __future__ import division
from .quaternion import leanForward
from ..core.util import *
import re
import struct
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import scale
from .. import settings


logger = logging.getLogger(__name__)


def normalize_necklace(df):
    df = df.copy()

    # absolute values, just need to calibrate to 0 and 1
    df['leanForward'] = df['leanForward']/float(90)

    # energy kept intact, since it is absolute value


    # normalize proximity and ambient locally by hours
    prox_amb = df[['proximity', 'ambient']].as_matrix()

    scaled_prox_amb = scale(prox_amb)

    df['proximity'] = scaled_prox_amb[:,0]
    df['ambient'] = scaled_prox_amb[:,1]

    return df


def _hex_to_float(string):
    string = string.replace(' ', '0')
    assert(len(string) == 8)
    return struct.unpack('!f', bytes.fromhex(string))[0]


def _hex_to_int(string):
    string = string.replace(' ', '0')
    assert(len(string) == 4)
    return struct.unpack('!H', bytes.fromhex(string))[0]


def _parse_binary(binString, order_proximity_first=True):
    '''
    Parse the hex format 
    Returns:
    --------

    dict

    Time: in ms
    proximity
    ambient
    
    '''
    
    binString = binString.strip()
    chunk = re.split(',|S', binString)
    timeStamp = int(chunk[0])
    cal = int(binString[-2:])

    if order_proximity_first:
        proximity = _hex_to_int(chunk[1][:4])
        ambient = _hex_to_int(chunk[1][4:8])
    else:
        ambient = _hex_to_int(chunk[1][:4])
        proximity = _hex_to_int(chunk[1][4:8])

    aX = _hex_to_float(chunk[1][8:16])
    aY = _hex_to_float(chunk[1][16:24])
    aZ = _hex_to_float(chunk[1][24:32])

    qW = _hex_to_float(chunk[1][32:40])
    qX = _hex_to_float(chunk[1][40:48])
    qY = _hex_to_float(chunk[1][48:56])
    qZ = _hex_to_float(chunk[1][56:64])

    lf = leanForward((qW, qX, qY, qZ))

    result = {}
    result['Time'] = timeStamp * 1000
    result['proximity'] = proximity
    result['ambient'] = ambient
    result['aX'] = aX
    result['aY'] = aY
    result['aZ'] = aZ
    result['qW'] = qW
    result['qX'] = qX
    result['qY'] = qY
    result['qZ'] = qZ
    result['leanForward'] = lf
    result['cal'] = cal

    return result


def parse_necklace(rawfile, order_proximity_first, outlier_function):

    '''
    Parse the hex data of necklace
    Also split data into hours (if raw file span two hours)

    Parameters:
    -----------

    rawfile: string
        path to the original raw file

    order_proximity_first: boolean
        whether proximity goes first in the hex representation

    outlier_time_function: 
        a function that returns true for invalid timestamps

    
    Returns:
    --------

    Dict that maps filename (hour format) to the correct data string
    
    '''

    logger.info(rawfile)
    
    split_hour = {}

    with open(rawfile) as f:
        for line in f:
            sample = _parse_binary(line.strip(), order_proximity_first)
            timestamp = sample['Time']

            if not outlier_function(timestamp):
                dt = epoch_to_datetime(timestamp)

                hour = '{}_{:02d}.csv'.format(\
                    dt.strftime(settings.DATEFORMAT), dt.hour)

                if not hour in split_hour:
                    split_hour[hour] = ""
                else:
                    s = []
                    for h in settings.NECKLACE_HEADER:
                        s.append(str(sample[h]))
                    sample_str = ','.join(s) + '\n'

                    split_hour[hour] += sample_str

    return split_hour, settings.NECKLACE_HEADER


def duplicate_second_to_millisecond(duplicate_time):
    ''' Convert duplicate timestamp in seconds to milliseconds

    Count the number of duplicates,
    and divided evenly throughout the second

    For eg, [1 1 1 2 2] will become [1 1.33 1.66 2 2.5]

    Parameters:
        duplicate_time: Pandas Series of second timestamps

    Returns:
        increasing_time: numpy array of millisecond timestamp
    '''

    assert_vector(duplicate_time)

    logger.info("Converting duplicate sec time to ms")

    N = len(duplicate_time)

    increasing_time = np.copy(duplicate_time).astype(float)

    reliability_ts = []
    reliability_count = []

    count = 0
    for i in range(N - 1):
        if (duplicate_time[i + 1] == duplicate_time[i]):
            count += 1
        else:
            increasing_time[(i - count):(i + 1)] += 1000 * \
                np.arange(count + 1) / float(count + 1)
            count = 0

    if count > 0:
        increasing_time[(N - count - 1):N] += 1000 * \
            np.arange(count + 1) / float(count + 1)
        
    return increasing_time.astype(int)
