import pandas as pd
# from mining.periodic import *
from beyourself.data import get_necklace, get_necklace_timestr
from beyourself.data.label import read_json, read_SYNC
from beyourself import settings
from beyourself.core.util import *
from beyourself.core.algorithm import *
# from beyourself.cleanup.timeseries import resampleing
from os.path import join
from datetime import datetime, timedelta
from mining.feature import get_feature
import inspect    
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

import matplotlib.pyplot as plt
from datetime import timedelta, date

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
    # time64 = (np.timedelta64(max_gap_sec, 's').astype('uint64') / 1e9).astype('uint64')
    # print(np.timedelta64(max_gap_sec, 's'))
    # print(type(np.timedelta64(max_gap_sec, 's')))
    # print(type(np.diff(df.index)))
    idx = np.where(np.greater(np.diff(df.index), 1000))[0]
    # idx = np.where(np.greater(np.diff(df.index),time64))[0]
    start = df.index[idx].tolist()
    stop = df.index[idx + 1].tolist()
    big_gaps = list(zip(start, stop))

    # upsample to higher frequency
    # df = df.resample('{}ms'.format(1000/higher_freq)).mean().interpolate()
    
    df.index = pd.to_datetime(df.index,unit='ms')
    df = df.resample('10ms').mean().interpolate()

    # downsample to desired frequency
    df = df.resample('{}ms'.format(1000/sampling_freq)).ffill()

    # df.index = datetime_to_epoch_ms_3bits(str(df.index))
    # remove data inside the gaps
    for start, stop in big_gaps:
        df[start:stop] = None
    df.dropna(inplace=True)

    return df



folderpath = '/Users/apple/Downloads/necklace_inlab/ubicomp2018/beyourself/CLEAN/Label_IS/Middle_data'

outfolder = '/Users/apple/Downloads/necklace_inlab/ubicomp2018/beyourself/CLEAN/Label_IS'


subj_list = ["P000", "P106", "P108", "P110", "P111", "P113", "P116", "P119", "P120", "P121"]

# subj_list = ["P113"]

# subj_list = ["P113", "P121"]

for subj in subj_list:

    # print("================================================")
    # print(subj)

    # subj_folder = join(folderpath, subj)

    # df = pd.read_csv(join(subj_folder,'necklace.csv'))
    # print(df.shape)

    # # Read chewing groundtruth
    # df_chewing = read_SYNC(join(subj_folder,'labelchewing.json'))
    # print(df_chewing[['start','end']])
    # chewing_intervals = list(zip(df_chewing['start'].tolist(), df_chewing['end'].tolist()))


    # print(type(df_chewing['start'][0]))


    # # Run segmentation algorithm
    # time = df['Time'].as_matrix()
    # proximity = df['proximity'].as_matrix()

    # # be careful to pick prominence (check data is normalized or not)
    # peaks_index = peak_detection(proximity, min_prominence=3)
    # peaks_time = time[peaks_index]

    # subsequences = periodic_subsequence(peaks_index, peaks_time,\
    #                 min_length=4,max_length=100, eps=0.1,alpha=0.45, low=400, high=1200)


    # # Plotting the segmentation results
    # # plt.figure(figsize=(15,5))
    # # time_obj = [datetime.fromtimestamp(t/1000) for t in time]
    # # plt.plot(time_obj, proximity)

    # # for i in peaks_index:
    # #     plt.plot(time_obj[i], proximity[i], 'g*')

    # # for seq in subsequences:
    # #     for i in seq:
    # #         plt.plot(time_obj[i], proximity[i], 'y*')

    # #     # plot start end
    # #     plt.plot(time_obj[seq[0]], proximity[seq[0]], 'r*')
    # #     plt.plot(time_obj[seq[-1]], proximity[seq[-1]], 'b*')

    # # axes = plt.gca()
    # # axes.set_ylim([2000, 16000])

    # # plt.show()
    # # plt.savefig(join("data/inlab", subj + ".png"))
    # # plt.close()


    # segments = []
    # for index in subsequences:
    #     seq = time[index]
    #     segments.append(get_periodic_stat(seq))

    # df_segments = pd.DataFrame(segments, columns=['start','end','eps','pmin','pmax','length'])

    # df_segments = df_segments.drop_duplicates(keep=False).reset_index(drop=True)

    # # df_segments.to_csv("data/segment_{}".format(subj),index=None)



    # seg_intervals = list(zip(df_segments['start'].tolist(), df_segments['end'].tolist()))

    # # calculating groundtruth
    # intersect = interval_intersect_interval(groundtruth=chewing_intervals,\
    #                                 prediction=seg_intervals)

    # gt = intersect['prediction_gt']
    # recall = intersect['recall']

    # print("Recall: {}".format(recall))
    # df_segments['chewing_gt'] = pd.Series(gt)
    # subj_folder = join(folderpath, subj)


    df = pd.read_csv(join(folderpath,str(subj) + '_before_resampling_toZiwei.csv'))
    df_segments = pd.read_csv(join(folderpath,str(subj)+'_segments.csv'))



    

    df = df.set_index(["Time"])
    df['Time'] = df.index
    print(df)
    print('======')

    df = resampling(df, 20)
    print(df)
    print('------')    

    # # generate features
    # # resampling
    # df['Datetime'] = df['Time']
    # df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')\
    #                     .dt.tz_localize('UTC' )\
    #                     .dt.tz_convert(settings.TIMEZONE)
    # df = df.set_index('Datetime')
    # df = df[~df.index.duplicated(keep='first')]
    # df = df.resample('50ms').ffill()
    # df = df.dropna()

    FFTSAMPLE = 80


    list_total_feature = []

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
                print(df_window)
                print('++++')
                df_window = df_window.drop(columns="Time")
                print(df_window)
                print('+-+-')
                df_window = df_window.set_index(['Unnamed: 0'])

                f = get_feature(df_window, FFTSAMPLE)

                f.columns = ["{}_{}".format(t, win_header) for t in f.columns.values]
                multi_win_f_list.append(f)

            if len(multi_win_f_list) == 2:
                multi_win_f = pd.concat(multi_win_f_list, axis=1)

            multi_win_f['pmin'] = df_segments['pmin'][i]
            multi_win_f['pmax'] = df_segments['pmax'][i]
            multi_win_f['eps'] = df_segments['eps'][i]
            multi_win_f['length'] = df_segments['length'][i]
            multi_win_f['label'] = df_segments['chewing_gt'][i]
            multi_win_f['start'] = df_segments['start'][i]
            multi_win_f['end'] = df_segments['end'][i]

            list_total_feature.append(multi_win_f)

        except ValueError as e:
            print(e)
            print("Skipping row {} due to error".format(i))



    feature = pd.concat(list_total_feature)

    feature.to_csv(join(outfolder, 'inlab_feature_' + subj + '.csv'), index=None)