
# coding: utf-8

# In[18]:


from IPython.display import display, HTML
import pandas as pd

from mining.periodic import *
from beyourself.data import get_necklace, get_necklace_timestr
from beyourself.data.label import read_json, read_SYNC
from beyourself import settings
from beyourself.core.util import *
from beyourself.core.algorithm import *

from os.path import join
    
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import plotly.offline as pltly
import plotly.graph_objs as go


from datetime import timedelta, date


# In[ ]:


folderpath = '/Users/apple/Downloads/necklace_inlab/ubicomp2018/beyourself/CLEAN/Label_IS'


# In[ ]:


subj = "P120"
subj_folder = join(folderpath, subj)

df_sensor = pd.read_csv(join(subj_folder,'necklace.csv'))
print(df_sensor.shape)
print(df_sensor)

# Read chewing groundtruth
df_chewing = read_SYNC(join(subj_folder,'labelchewing.json'))
print(df_chewing[['start','end']])
chewing_intervals = list(zip(df_chewing['start'].tolist(), df_chewing['end'].tolist()))


# In[ ]:


# Run segmentation algorithm
time = df_sensor['Time'].as_matrix()
proximity = df_sensor['proximity'].as_matrix()

# be careful to pick prominence (check data is normalized or not)
peaks_index = peak_detection(proximity, min_prominence=10)
peaks_time = time[peaks_index]

subsequences = periodic_subsequence(peaks_index, peaks_time,                min_length=4,max_length=100, eps=0.1,alpha=0.45, low=400, high=1200)


plt.figure(figsize=(15,5))


time_obj = [datetime.fromtimestamp(t/1000) for t in time]
plt.plot(time_obj, proximity)


for i in peaks_index:
    plt.plot(time_obj[i], proximity[i], 'g*')


for seq in subsequences:
    for i in seq:
        plt.plot(time_obj[i], proximity[i], 'y*')

    # plot start end
    plt.plot(time_obj[seq[0]], proximity[seq[0]], 'r*')
    plt.plot(time_obj[seq[-1]], proximity[seq[-1]], 'b*')

plt.savefig(join("data/inlab", subj + ".png"))
plt.close()


# In[ ]:


segments = []
for index in subsequences:
    seq = time[index]
    segments.append(get_periodic_stat(seq))

df_segments = pd.DataFrame(segments, columns=['start','end','eps','pmin','pmax','length'])


seg_intervals = list(zip(df_segments['start'].tolist(), df_segments['end'].tolist()))

# calculating groundtruth
intersect = interval_intersect_interval(groundtruth=chewing_intervals,                                prediction=seg_intervals)

gt = intersect['prediction_gt']
recall = intersect['recall']

print("Recall: {}".format(recall))
df_segments['chewing_gt'] = pd.Series(gt)
    
display(df_segments)
df_segments.to_csv('segments_P120.csv',index=None)

# generate features

# resampling
df_sensor['Datetime'] = df_sensor['Time']
df_sensor['Datetime'] = pd.to_datetime(df_sensor['Datetime'], unit='ms')                    .dt.tz_localize('UTC' )                    .dt.tz_convert(settings.TIMEZONE)
df_sensor = df_sensor.set_index('Datetime')
df_sensor = df_sensor[~df.index.duplicated(keep='first')]
df_sensor = df_sensor.resample('50ms').ffill()
df_sensor = df_sensor.dropna()


for i in range(df_segments.shape[0]):
    start_unixt = datetime_to_epoch(df_segments['start'][i])
    end_unixt = datetime_to_epoch(df_segments['end'][i])

    win_start = [start_unixt - 2000, start_unixt - 3000]
    win_end = [end_unixt + 2000, start_unixt + 1000]
    window_list = ["-2s_start_2s_end","-2s_start_2sstart"]

    min_start = min(win_start)
    max_end = max(win_end)

    try:
        multi_win_f_list = []
        for s, e, win_header in zip(win_start, win_end, window_list):

            df_window = df[(df['Time'] >= s) & (df['Time'] <= e)]
            f = get_feature(df_window, FFTSAMPLE)

            f.columns = ["{}_{}".format(t, win_header) for t in f.columns.values]
            multi_win_f_list.append(f)

        if len(multi_win_f_list) == 2:
            multi_win_f = pd.concat(multi_win_f_list, axis=1)

        multi_win_f['pmin'] = df_segments['pmin'][i]
        multi_win_f['pmax'] = df_segments['pmax'][i]
        multi_win_f['eps'] = df_segments['eps'][i]
        multi_win_f['length'] = df_segments['length'][i]
        multi_win_f['hour'] = df_segments['start'][i].hour
        multi_win_f['label'] = df_segments['chewing_gt'][i]
        multi_win_f['date_exp'] = df_segments['date_exp'][i]

        list_total_feature.append(multi_win_f)

    except ValueError as e:
        print(e)
        print("Skipping row {} due to error".format(i))


print(time.time() - start_time)


feature = pd.concat(list_total_feature)

feature.to_csv('inlab_feature_' + subj + '.csv', index=None)

