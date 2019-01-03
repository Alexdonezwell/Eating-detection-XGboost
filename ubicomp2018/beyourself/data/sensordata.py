import os
from .. import settings
import pandas as pd
from datetime import datetime, timedelta
import logging
from ..core.util import human_to_epoch
import numpy as np


logger = logging.getLogger(__name__)


def get_necklace(subj, start, end, reliability=0.2):
    folder = os.path.join(settings.CLEAN_FOLDER, subj + '/necklace')

    return _get_data(folder, start, end, reliability)
    
def get_watch(subj, start, end, reliability=0.2):
    logger.info("getting wathc data")
    folder = os.path.join(settings.CLEAN_FOLDER, subj + '/WRIST/ACCELEROMETER')

    return _get_data(folder, start, end, reliability)
    
def get_watch_timestr(subj, start_str, end_str, reliability=0.2):
    folder = os.path.join(settings.CLEAN_FOLDER, subj + '/WRIST/ACCELEROMETER')

    return _get_data_timestr(folder, start_str, end_str, reliability)


def get_necklace_timestr(subj, start_str, end_str, reliability=0.2):
    folder = os.path.join(settings.CLEAN_FOLDER, subj + '/necklace')

    return _get_data_timestr(folder, start_str, end_str, reliability)


def _get_data_timestr(folder, start_str, end_str, reliability):
    logger.debug("Querying from {} to {}".format(start_str, end_str))
    start = human_to_epoch(start_str)
    end = human_to_epoch(end_str)

    return _get_data(folder, start, end, reliability)


def _get_data(folder, start, end, reliability):
    '''
    Get data from the cleaned folder

    folder: path to the clean folder, following time-series structure
    start: unix time in ms
    end:   unix time in ms
    contiguous: whether need to return contiguous data or not
    reliability: lowest reliability score
    
    Returns:
    --------

    pandas dataframe

    '''

    logger.debug("Querying from {} to {}, lowest reliability {}".format(start, end, reliability))

    start_dt = datetime.fromtimestamp(start // 1000)
    end_dt = datetime.fromtimestamp(end // 1000)

    concat_list = []

    current = start_dt.replace(minute=0, second=0, microsecond=0)
    while (current < end_dt):
        path = os.path.join(folder, 'data/{}_{:02d}.csv'.format(
            current.strftime(settings.DATEFORMAT), current.hour))

        if os.path.isfile(path):
            logger.debug(path)
            df = pd.read_csv(path, index_col=False)
            df = df[(df.Time >= start) & (df.Time <= end)]

            concat_list.append(df)

        current += timedelta(hours=1)

    if len(concat_list) > 0:
        df_total = pd.concat(concat_list)
        df_total['Time'] = pd.to_datetime(df_total['Time'], unit='ms')\
                                .dt.tz_localize('UTC' )\
                                .dt.tz_convert(settings.TIMEZONE)
        df_total.set_index(['Time'],inplace=True)
        return df_total

    else:
        return pd.DataFrame(columns = settings.NECKLACE_HEADER)
