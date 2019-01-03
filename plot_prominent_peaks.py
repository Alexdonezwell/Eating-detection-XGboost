import pandas as pd
import numpy as np
from mining.periodic import peak_detection 
from beyourself.data import get_necklace_timestr
import matplotlib.dates as mdates
from scipy.signal import argrelextrema
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import datetime

from beyourself.core.util import datetime_to_epoch

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import matplotlib
matplotlib.rcParams.update({'font.size': 20})



data = get_necklace_timestr('P120',"2017-08-26 08:18:15.000","2017-08-26 08:19:00.000",reliability=0.01)

# data = pd.read_csv('data/test_fft.csv')

proximity = data['proximity'].as_matrix()
proximity = proximity - 2000
time = [datetime_to_epoch(t) for t in data.index]
time = np.array(time)


peaks_index = peak_detection(proximity, min_prominence=6)
peaks_index = np.array(peaks_index)
peaks_time = time[peaks_index]

other_peaks_index = argrelextrema(proximity, np.greater, order=2, mode='clip')[0]
other_peaks_time = time[peaks_index]

# segments = periodic_subsequence(peaks_index, peaks_time, min_length=4,max_length=100, eps=0.1, alpha=0.45,low=400,high=1200)

time_obj = [datetime.datetime.fromtimestamp(t/1000) for t in time]

plt.figure(figsize=(10,5))


# plt.plot(time_obj, original_prox)

plt.plot(time_obj, proximity)


prominence_time = [time_obj[i] for i in peaks_index]
prominence_y = [proximity[i] for i in peaks_index]

plt.scatter(prominence_time, prominence_y, marker='^',c='r', label='prominence-based peaks')


other_time = [time_obj[i] for i in other_peaks_index]
other_y = [0] * len(other_time)

plt.scatter(other_time, other_y, marker='*', c='b', label='local maxima peaks')



# for seq in segments:
# 	for i in seq:
# 		# print(time_obj[i])
# 		plt.plot(time_obj[i], proximity[i], 'y*')

# 	# plot start end
# 	plt.plot(time_obj[seq[0]], proximity[seq[0]], 'r*')
# 	plt.plot(time_obj[seq[-1]], proximity[seq[-1]], 'b*')


# label = pd.read_csv('labels.csv')

# label['start'] = pd.to_datetime(label['start'])
# label['end'] = pd.to_datetime(label['end'])

# for i in range(label.shape[0]):
# 	if label['label'][i] == "b":
# 		plt.plot(label['start'][i], 5000, 'k*')
# 	else:
# 		plt.plot(label['start'][i], 5000, 'r*')
# 		plt.plot(label['end'][i], 5000, 'b*')
# print(segments)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.SecondLocator(interval=15))   #to get a tick every 15 minutes
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))     #optional formatting 

plt.yticks(np.arange(0, 7000, 1500))

plt.legend()
plt.tight_layout()
plt.savefig('peaks.pdf')
# plt.show()
