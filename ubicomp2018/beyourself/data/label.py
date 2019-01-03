import pandas as pd
import json
from .. import settings
from ..core.util import *


def read_missing_json(path):
    with open(path) as f:
        data = json.load(f)
    
    cut_pnt_list = []
    missing_dur_list = []

    for item in data:
        for key, value in item.items():
            cut_pnt_list.append(key)
            missing_dur_list.append(value)

    return cut_pnt_list, missing_dur_list


def read_json(path):
    with open(path) as f:
        data = json.load(f)

        start = []
        end = []
        label = []
        for item in data['array']:
            start.append(datetime_from_str(item['start']))
            end.append(datetime_from_str(item['end']))
            label.append(item['label'])

        df = pd.DataFrame({
                'start': start,
                'end': end,
                'label': label
            }, columns=['start','end','label'])

        return df


def write_json(df, type, absolute=False):
    pass


def read_ELAN(path):
    '''
    Read ELAN txt files into a pandas dataframe
    '''
    
    df = pd.read_table(path, header=None)
    df = df.iloc[:,[2,4,-1]]
    df.columns = ['start', 'end', 'label']

    df['start'] = pd.to_timedelta(df['start'])
    df['end']   = pd.to_timedelta(df['end'])
    
    return df


def read_SYNC(path):
    '''
    read SYNC point annotation
    and convert to pd dataframe
    storing durations
    '''

    with open(path) as f:
        data = json.load(f)
        N = len(data)
        marked = [False] * N

        keys = sorted(data.keys())
        values = [data[k] for k in keys]

        list_rows = []

        for i in range(N):
            value = values[i]

            if marked[i] == False:
                if value[-1] != '1' and value[-1] != '2':
                    row = {}
                    row['start'] = keys[i]
                    row['end'] = keys[i]
                    row['label'] = value

                else:
                    duration_end = value[:-1] + '2'

                    for j in range(i, N):
                        if marked[j] == False and values[j] == duration_end:
                            marked[j] = True

                            row = {}
                            row['start'] = keys[i]
                            row['end'] = keys[j]
                            row['label'] = value[:-1]
                            break

                list_rows.append(row)

        df = pd.DataFrame(list_rows, columns=['start','end','label'])

        df['start'] = pd.to_datetime(df['start'])\
                        .dt.tz_localize(settings.TIMEZONE)

        df['end'] = pd.to_datetime(df['end'])\
                        .dt.tz_localize(settings.TIMEZONE)

        return df


def write_SYNC(df, outpath):
    obj = {}

    for i in range(df.shape[0]):
        obj[datetime_to_str(df['start'][i])] = df['label'][i] + '1'
        obj[datetime_to_str(df['end'][i])] = df['label'][i] + '2'

    with open(outpath, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# read_json('feeding.json')

# df = read_ELAN('DVR___2017-07-23_11.25.32.AVI.txt')
# print(df)
# print(df.shape)

# write_SYNC(df, 'test.json')

