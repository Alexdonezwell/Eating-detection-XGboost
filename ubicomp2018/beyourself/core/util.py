import os
import logging
import shutil
import numpy as np
from datetime import datetime, timedelta, time, date, timezone
from .. import settings
import pandas as pd


logger = logging.getLogger(__name__)


def epoch_to_datetime_df(df, column=['Time']):

    df = df.copy()
    
    for c in column:
        df[c] = pd.to_datetime(df[c], unit='ms')\
                        .dt.tz_localize('UTC' )\
                        .dt.tz_convert(settings.TIMEZONE)
    return df


def humanstr_withtimezone_to_datetime_df(df, column_list=['start','end']):
    '''
    convert human string to datetime (tz aware) object
    '''
    for c in column_list:
        df[c] = pd.to_datetime(df[c])
        df[c] = df[c].apply(lambda x: x.tz_localize('UTC').\
                                tz_convert(settings.TIMEZONE))
    
    return df


def humanstr_notimezone_to_datetime_df(df, column_list=['start','end']):
    '''
    convert human string to datetime (tz aware) object
    '''
    for c in column_list:
        df[c] = pd.to_datetime(df[c])
        df[c] = df[c].apply(lambda x: settings.TIMEZONE.localize(x))

    return df


def maybe_create_folder(f, deleteExisting=False):
    '''
    Create the folder

    Parameters:
            f: folder path. Could be nested path (so nested folders will be created)

            deleteExising: if True then the existing folder will be deleted.

    '''
    if os.path.exists(f):
        if deleteExisting:
            shutil.rmtree(f)
    else:
        os.makedirs(f)


def assert_monotonic(x):
    if not np.all(np.diff(x) >= 0):
        raise Exception("Not monotonic")


def assert_vector(x):
    if not type(x) is np.ndarray and x.ndim != 1:
        raise Exception("Not a vector")


def datetime_to_epoch(dt):
    '''
    Convert Python datetime object (timezone aware)
    to epoch unix time in millisecond
    '''
    # dt = dt.replace(tzinfo=timezone.utc).timestamp()
    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f%z")
    # if len(dt) == 24:
    #     dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S%z")
    # if len(dt) == 30:
    #     dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f%z")
    # print(dt)
    # print(type(dt))
    return int(1000 * dt.timestamp())

def datetime_to_epoch_ms_3bits(dt):
    
    dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
    return int(1000 * dt.timestamp())


def epoch_to_datetime(unixtime):
    '''
    Convert unix timestamp in millisecond
    to a Python datetime object (timezone aware)
    '''
    return datetime.fromtimestamp(unixtime/1000.0, settings.TIMEZONE)


def human_to_epoch(str):
    return datetime_to_epoch(datetime_from_str(str))


def epoch_to_human(unixtime):
    return datetime_to_str(epoch_to_datetime(unixtime))


def timedelta_from_str(relative_str):
    t = datetime.strptime(relative_str, "%H:%M:%S.%f")
    if t.microsecond == None :
        relative = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    else:
        relative = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

    return relative


def datetime_from_str(absolute_str):
    naive = datetime.strptime(absolute_str, settings.ABSOLUTE_TIME_FORMAT)
    aware = settings.TIMEZONE.localize(naive)
    return aware


def datetime_to_str(dt):
    return dt.strftime(settings.ABSOLUTE_TIME_FORMAT)[:23]


def sync_relative_time(relative, matching):
    '''
    Given a matching pair of LED_relative, LED_absolute
    calculate the corresponding absolute time for relative_start

    Parameters:
    
    relative: timedelta object or array
        relative time of an event that we want to find absolute time

    matching: a dict contains
        LED_relative: string
            relative time of a synced event

        LED_absolute: string
            absolute time of a synced event

        offset: int
            offset in millisecond to be added to absolute time

    Return:

    absolute_time: datetime object

    '''

    LED_relative = matching['relative']
    LED_absolute = matching['absolute']
   
    if not 'SYNC' in matching:
        offset = 0
    else:
        offset = int(matching['SYNC'])
        logger.info("Syncing offset is not zero")

    absolute_dt = datetime.strptime(LED_absolute, settings.ABSOLUTE_TIME_FORMAT)
    shift = absolute_dt - timedelta_from_str(LED_relative) + timedelta(microseconds=offset*1000)

    return relative + shift


def epoch_to_relative_str(ms):

    dt = (datetime.min + timedelta(microseconds=1000*ms)).time()
    return dt.strftime(settings.RELATIVE_TIME_FORMAT)[:-3]


def subtract_relative_time(relative_start, relative_end):
    '''
    Given two relative times
    calculate the difference between them
    '''

    start = timedelta_from_str(relative_start)
    end = timedelta_from_str(relative_end)

    time_obj = (datetime.min + (end - start)).time()
    return time_obj.strftime(settings.RELATIVE_TIME_FORMAT)[:-3]


def segment_plot(start, end, height):

    x = []
    y = []

    for s,e,h in zip(start, end, height):
        
        x.append(s)
        y.append(0)

        x.append(s)
        y.append(h)

        x.append(e)
        y.append(h)

        x.append(e)
        y.append(0)

    return x, y




