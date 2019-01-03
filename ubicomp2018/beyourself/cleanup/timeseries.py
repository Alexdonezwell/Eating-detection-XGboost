from ..core import batch_run, maybe_create_folder
from .. import settings
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np
from beyourself.core.util import epoch_to_datetime_df


logger = logging.getLogger(__name__)


def split_hour(raw_file, out_folder, parse_function):
    '''Remap samples into hour files

    Samples are guaranteed to be in the correct hour file
    Outlier will be removed (1970 or in the future), 
    determined by outlier_time function

    Also each sample will be parsed following the parse_function

    Parameters:

    raw_file: 
            absolute path of the original data file

    out_folder:
            output folder containing dates/hours

    parse_function:
            return a dict of filename map to data string

    '''

    logger.info("Mapping %s to hour file", raw_file)


    split_hour, header_list = parse_function(raw_file)

    for filename, data in sorted(split_hour.items()):

        hour_path = os.path.join(out_folder, filename)

        if not os.path.exists(hour_path):
            with open(hour_path, 'w') as f:
                f.write(','.join(header_list) + '\n')

        with open(hour_path,'a') as f:            
            f.write(data)


def get_reliability(df):
    '''
    Calculate reliability of a dataframe
    '''
    time_ms = df['Time'].as_matrix()
    time_sec = time_ms//1000

    time, count = np.unique(time_sec, return_counts=True)

    df_reliability = pd.DataFrame({ 'Time':1000*time,\
                                    'Count':count})

    return df_reliability


def resampling(df, sampling_freq=20, higher_freq=100, max_gap_sec=1):
    ''' Resample unevenly spaced timeseries data linearly by 
    first upsampling to a high frequency (short_rate) 
    then downsampling to the desired rate.

    Parameters
    ----------
        df:               dataFrame
        sampling_freq:    sampling frequency
        max_gap_sec:      if gap larger than this, interpolation will be avoided
    
    Return
    ------
        result:           dataFrame
    
    '''
    
    # find where we have gap larger than max_gap_sec
    # print(df.index)
    # diff = np.diff(df.index)

    # print(diff)
    idx = np.where(np.greater(np.diff(df.index), np.timedelta64(max_gap_sec, 's')))[0]
    start = df.index[idx].tolist()
    stop = df.index[idx + 1].tolist()
    big_gaps = list(zip(start, stop))

    # upsample to higher frequency
    df = df.resample('{}ms'.format(1000/higher_freq)).mean().interpolate()

    # downsample to desired frequency
    df = df.resample('{}ms'.format(1000/sampling_freq)).ffill()

    # remove data inside the gaps
    for start, stop in big_gaps:
        df[start:stop] = None
    df.dropna(inplace=True)

    return df
