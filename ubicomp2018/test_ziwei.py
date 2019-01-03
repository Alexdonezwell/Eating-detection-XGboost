import os
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np

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
    # print(np.diff(df.index).dtype)
    # print('======')
    # print((np.timedelta64(max_gap_sec, 's').astype('uint64') / 1e6).astype('uint64'))
    time64 = (np.timedelta64(max_gap_sec, 's').astype('uint64') / 1e6).astype('uint64')

    idx = np.where(np.greater(np.diff(df.index),time64))[0]
    start = df.index[idx].tolist()
    stop = df.index[idx + 1].tolist()
    big_gaps = list(zip(start, stop))

    # upsample to higher frequency
    # df = df.resample('{}ms'.format(1000/higher_freq)).mean().interpolate()
    df.index = pd.to_datetime(df.index,unit='ms')
    df = df.resample('10ms').mean().interpolate()

    # downsample to desired frequency
    df = df.resample('{}ms'.format(1000/sampling_freq)).ffill()

    # remove data inside the gaps
    for start, stop in big_gaps:
        df[start:stop] = None
    df.dropna(inplace=True)

    return df

df = pd.read_csv('/Users/apple/Desktop/Tomaz_implement/P000_necklace.csv')
df = df.set_index(["Time"])


print(df)
print(df.columns)
print('fine')
df = resampling(df, 20)
print(df)
print("ok")