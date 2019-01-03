import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from datetime import datetime, timedelta, time, date
from collections import namedtuple
from beyourself import settings
from beyourself.core.util import maybe_create_folder
from scipy.signal import boxcar
import time
import re
from six import string_types
import sys


def lprint(logfile, *argv): # for python version 3

    """ 
    Function description: 
    ----------
        Save output to log files and print on the screen.

    Function description: 
    ----------
        var = 1
        lprint('log.txt', var)
        lprint('log.txt','Python',' code')

    Parameters
    ----------
        logfile:                 the log file path and file name.
        argv:                    what should 
        
    Return
    ------
        none

    Author
    ------
    Shibo(shibozhang2015@u.northwestern.edu)
    """

    # argument check
    if len(argv) == 0:
        print('Err: wrong usage of func lprint().')
        sys.exit()

    argAll = argv[0] if isinstance(argv[0], str) else str(argv[0])
    for arg in argv[1:]:
        argAll = argAll + (arg if isinstance(arg, str) else str(arg))
    
    print(argAll)

    with open(logfile, 'a') as out:
        out.write(argAll + '\n')


def list_folder_in_directory(mypath):
    return [f for f in os.listdir(mypath) if not os.path.isfile(os.path.join(mypath, f))]


def list_files_in_directory(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]


def group_miss_seconds(targets, preds):
    for pred in preds:
        targets = subtract_time(targets, pred)
    
    ttl_sec = 0
    for target in targets:
        ttl_sec += (target[1]-target[0]).total_seconds()

    return ttl_sec


def subtract_time(targets, pred):
    Range = namedtuple('Range', ['start', 'end'])
    dt_s2 = pred[0]
    dt_e2 = pred[1]
    r2 = Range(start=dt_s2, end=dt_e2)
    new_targets = []

    for target in targets:
        dt_s1 = target[0]
        dt_e1 = target[1]
        # print(dt_s1)
        # print(dt_e1)

        r1 = Range(start=dt_s1, end=dt_e1)
        # print(r1)
        latest_start = max(r1.start, r2.start)
        earliest_end = min(r1.end, r2.end)
        overlap_sec = (earliest_end - latest_start).total_seconds()

        if overlap_sec>0:
            if dt_s2>dt_s1 and dt_s2<dt_e1:
                new_targets.append([dt_s1, dt_s2])
            if dt_e2>dt_s1 and dt_e2<dt_e1:
                new_targets.append([dt_e2, dt_e1])
        else:
            new_targets.append(target)

    return new_targets


def group_overlap_seconds(group1, group2):
    overlap_sec = 0
    for interval1 in group1:
        for interval2 in group2:
            overlap_sec += overlap_seconds(interval1, interval2)

    return overlap_sec


def overlap_seconds(interval1, interval2):
    dt_s1 = interval1[0]
    dt_e1 = interval1[1]
    dt_s2 = interval2[0]
    dt_e2 = interval2[1]

    Range = namedtuple('Range', ['start', 'end'])
    r1 = Range(start=dt_s1, end=dt_e1)
    r2 = Range(start=dt_s2, end=dt_e2)
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)
    overlap_sec = (earliest_end - latest_start).total_seconds()
    overlap_sec = max(0, overlap_sec)

    return overlap_sec


def df_to_datetime_tz_aware(in_df, column_list):
    df = in_df.copy()

    for column in column_list:
        if len(df): # if empty df, continue
            d = df[column].iloc[0]
            # if type is string 
            if isinstance(d, string_types):#if "import datetime" then "isinstance(x, datetime.date)"
                if check_end_with_timezone(d):
                    # if datetime string end with time zone
                    lprint(log_file,'Column '+column+' time zone contained')
                    # df[column] = pd.to_datetime(df[column],utc=True).apply(lambda x: x.tz_convert(settings.TIMEZONE))
                    df[column] = pd.to_datetime(df[column])
                    df[column] = df[column].apply(lambda x: x.tz_localize('UTC').\
                                tz_convert(settings.TIMEZONE))
                else:
                    # if no time zone contained
                    lprint(log_file,'Column '+column+' no time zone contained')
                    df[column] = pd.to_datetime(df[column]).apply(lambda x: x.tz_localize(settings.TIMEZONE))
            
            # if type is datetime.date
            elif isinstance(d, date):
                if d.tzinfo is not None and d.tzinfo.utcoffset(d) is not None:
                    # if datetime is tz naive
                    lprint(log_file,'Column '+column+" time zone naive")
                    df[column] = df[column].apply(lambda x: x.tz_convert(settings.TIMEZONE))
                else:
                    # if datetime is tz aware
                    lprint(log_file,'Column '+column+" time zone aware")
    return df


def check_end_with_timezone(s):# TODO: limit re to -2numbers:2number
    m = re.search(r'-\d+:\d+$', s)
    return True if m else False


def pointwise2headtail(pw):
    # pw means pointwise_rpr
    diff = np.concatenate((pw[:],np.array([0]))) - np.concatenate((np.array([0]),pw[:]))
    ind_head = np.where(diff == 1)[0]
    ind_tail = np.where(diff == -1)[0]-1
    # lprint(log_file,len(ind_tail))
    # lprint(log_file,len(ind_head))

    headtail_rpr = np.vstack((ind_head, ind_tail)).T;

    return headtail_rpr


# def markClassPeriod(df, output_column, starttimes, endtimes):
#     for st, end, label in zip(starttimes,endtimes,labels):
#         df[output_column].iloc[(df.index >= st) & (df.index < end)] = label
#     # for st, end, pred in zip(starttimes,endtimes,preds):
#     #     df['pred'].iloc[(df.index >= st) & (df.index < end)] = pred
        
#     return df


def pw_fusion(df, output_column, starttimes, endtimes, preds):
    for st, end, pred in zip(starttimes,endtimes,preds):
        if pred == 1:
            indices = df.index[(df.index >= st) & (df.index < end)].tolist()
            for ind in indices:
                df[output_column][ind] = df[output_column][ind] + 1

    return df


# # This padding is one option, otherwise you can pad before smoothing
# def rolling_ave(df1, winsize = 200):
#     df = df1.copy()
#     df = df.rolling(window=winsize, center=True).mean()
#     df = df.fillna(method = 'backfill')
#     df = df.fillna(method = 'pad')

#     return df


def smooth_boxcar(data, selected_columns, winsize):
    """Boxcar smoothing of data

    Parameters
    ----------
    data:                   dataframe
    selected_columns:       list of keys, stating which columns will be smoothed
    winsize:                number of samples of rectangle window

    Return
    ------
    smoothed:               dataFrame

    """
    smoothed = data.copy(deep=True)

    for col_header in selected_columns:
        column = smoothed[col_header].as_matrix()

        # padding data
        # when winsize is even, int(winsize/2) is bigger than int((winsize-1)/2) by 1
        # when winsize is odd, int(winsize/2) is the same as int((winsize-1)/2)
        pad_head = [column[0]] * int((winsize - 1) / 2)
        pad_tail = [column[-1]] * int(winsize / 2)
        signal = np.r_[pad_head, column, pad_tail]

        window = boxcar(winsize)

        smoothed[col_header] = np.convolve(
            window / window.sum(), signal, mode='valid')
            
    return smoothed


def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta


def cut(b, thre = 1):
    a = np.empty_like (b)
    a[:] = b
    a[a<thre] = 0
    return a


def filter_back(flt_farr, score):
    ind = np.where(score == 0)
    
    flt_farr[ind] = 0
    return flt_farr


def merge(array, min_dist):
    """fill in the small gaps shorter than min_dist threshold
    """
    indices=[]

    # save the indices when the number changes
    for i in range(len(array)):
        if i==len(array)-1:
            indices.append(i)
            break
        else:
            if array[i]!=array[i+1]:
                indices.append(i)
            else:
                continue
    segments=[]
    start=0

    # break input array down to pieces and save in list 'segments'
    for i in (indices):
        segments.append(array[start:i+1])
        start=i+1

    for (i,sequence) in enumerate(segments):
        # if this is an all-0 segments ([0,0,....,0])
        if np.mean(sequence)==0: 
            # if this all-zero segment is shorter then min_min_dist threshold
            if len(sequence)<=min_dist:
                # fill in this short all-zero segment, that is, fill in this gap
                segments[i]=1-segments[i]

    # concat the segments to one piece of the orginal length
    new_array=np.concatenate(segments)
    return new_array
    


run_time_start = time.time()

inlab = 0

# days = ['1','2','3','4','5','6','7','8','9','10','11','12','13']


# subj = str(sys.argv[1])
# min_dist = int(sys.argv[2])
# threshold = float(sys.argv[3])
# log_file = str(sys.argv[4])

min_dist = 1200
threshold = 0.5
log_file = "fusion.log"


lprint(log_file,"This is the name of the script: ", sys.argv[0])
lprint(log_file,"Number of arguments: ", len(sys.argv))
lprint(log_file,"The arguments are: " , str(sys.argv))


lprint(log_file,"=====================================================================================")
lprint(log_file,"")
lprint(log_file,"")
lprint(log_file,"")
lprint(log_file,"save meal prediction from chewing detection")
lprint(log_file,"")
lprint(log_file,"")
lprint(log_file,"")
lprint(log_file,"=====================================================================================")



subj_list = ['P103','P105','P107','P108','P110','P114','P116','P118','P120','P121']


for subj in subj_list:

    files = list_files_in_directory(os.path.join('../data/wild/',subj,'personalized'))
    dur_df_list = []

    # for file in files:
    for day in range(14):

        filepath = "../data/wild/{}/personalized/prediction_day_{}.csv".format(subj, day)

        if not os.path.isfile(filepath):
            continue
        
        maybe_create_folder(os.path.join('Meal_prediction', subj))

        lprint(log_file,filepath)
        raw_df = pd.read_csv(filepath)


        # TIME ZONE CONTAINED AUTO CHECK CODE:
        raw_df = df_to_datetime_tz_aware(raw_df, ['start','end'])

        starttimes = raw_df['start'].values
        endtimes = raw_df['end'].values
        preds = raw_df['prediction'].values

        earliest = raw_df['start'].min()
        latest = raw_df['end'].max()


        dt = pd.date_range(start=earliest, end=latest, freq='50ms')
        
        df = pd.DataFrame()
        df['time']=dt
        df['score'] = 0


        lprint(log_file,"Elapsed time {}".format(time.time() - run_time_start))

        df = df.set_index('time')
        df = pw_fusion(df, 'score', starttimes, endtimes, preds) # one subj one day take 27 seconds

        df.to_csv('../data/wild/{}/personalized/pw_fusion_day{}_chew_pred_dur_mindist{}_thre{}.csv'.format(subj, day, min_dist, threshold, index=None))     
        lprint(log_file,"Elapsed time {}".format(time.time() - run_time_start))

        t = df.index.tolist()
        t = np.array(t)

        # pred = df['pred'].as_matrix()
        score = df['score'].as_matrix()
        df1 = smooth_boxcar(df, ['score'], min_dist) # one subj one day take 14 seconds

        lprint(log_file,"Elapsed time {}".format(time.time() - run_time_start))

        farr = df1['score'].as_matrix()
        # df1.to_csv("../data/wild/{}/personalized/tmp1_day{}_chew_pred_dur_mindist{}_thre{}.csv".format(subj, day, min_dist, threshold),index=None)

        # ===========================================================
        # one subj one day take 18 seconds
        flt_farr = cut(farr, thre = threshold)
        flt_farr = filter_back(flt_farr, score)
        flt_farr[np.where(flt_farr > 0)] = 1
        flt_farr = merge(flt_farr, min_dist)
        # ===========================================================

        lprint(log_file,"Elapsed time {}".format(time.time() - run_time_start))


        # # comment out the following to save plot
        # f, axarr = plt.subplots(4, sharex=True, figsize=(15,10))
        # label = df['label'].as_matrix()
        # axarr[0].plot(t, label)
        # axarr[1].plot(t, score)
        # axarr[2].plot(t, farr)
        # axarr[3].plot(t, flt_farr)

        maybe_create_folder(os.path.join("../data/wild/{}/personalized/".format(subj), 'Meal_prediction'))

        # plt.savefig(os.path.join('Meal_prediction', subj, 'meal_mindist'+str(min_dist)+'thre'+str(threshold)+'.png'))
        # lprint(log_file,"Elapsed time {}".format(time.time() - run_time_start))


        # np.savetxt("../data/wild/{}/personalized/day{}_meal_pw_mindist{}_thre{}.txt".format(subj, day, min_dist, threshold), flt_farr)

        ht = pointwise2headtail(flt_farr)
        m_pred = t[ht]

        # NOTE: Pandas cannot create DataFrame from Numpy Array of TimeStamps #13287. from: https://github.com/pandas-dev/pandas/issues/13287
        dur_df = pd.DataFrame.from_records(data = m_pred, columns = ['start','end'])
        lprint(log_file,dur_df)

        dur_df_list.append(dur_df)

        # dur_df.to_csv("../data/wild/{}/personalized/day{}_chew_pred_dur_mindist{}_thre{}.csv".\
                        # format(subj,day,min_dist,threshold), index = None)

        lprint(log_file,"Elapsed time {}".format(time.time() - run_time_start))


    lprint(log_file,"=====================================================================================")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"calculate event-based metrics based on meal ground truth and meal prediction")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"=====================================================================================")


    dur_df_total = pd.concat(dur_df_list)



    #### METRICS
    # read in ground truth
    directory = os.path.join('data',subj,'label')
    in_file = os.path.join(settings.CLEAN_FOLDER, subj, 'label/chewing.csv')
    gt_df = pd.read_csv(in_file)

    gt_df = df_to_datetime_tz_aware(gt_df, ['start', 'end'])

    gt = []
    for (s,e) in zip(gt_df['start'].tolist(),gt_df['end'].tolist()):
        gt.append([s,e])


    for df in dur_df_list:

        m_pred = []
        for (s,e) in zip(df['start'].tolist(),df['end'].tolist()):
            m_pred.append([s,e])
        # lprint(log_file,gt)
        # lprint(log_file,'m_pred:')
        # lprint(log_file,m_pred)

        overlap_second = group_overlap_seconds(gt, m_pred)
        lprint(log_file,"overlap_second:")
        lprint(log_file,overlap_second)
        # miss_second = group_miss_seconds(gt, m_pred)
        # lprint(log_file,"miss_second:")
        # lprint(log_file,miss_second)


    df = dur_df_total

    m_pred = []
    for (s,e) in zip(df['start'].tolist(),df['end'].tolist()):
        m_pred.append([s,e])

    overlap_second = group_overlap_seconds(gt, m_pred)
    lprint("=======final ==============")
    lprint(subj)
    lprint(log_file,"overlap seconds:")
    lprint(log_file,overlap_second)

    miss_second = group_miss_seconds(targets=gt, preds=m_pred)
    lprint(log_file,"miss seconds:")
    lprint(log_file,miss_second)

    fa_second = group_miss_seconds(targets=m_pred, preds=gt)
    lprint(log_file,"false alarm seconds:")
    lprint(log_file,fa_second)

    f1 = float(overlap_second)*2/(2*overlap_second+miss_second+fa_second)
    lprint(log_file,'f-score:')
    lprint(log_file,f1)
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"")
    lprint(log_file,"")

