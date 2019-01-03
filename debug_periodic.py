import pandas as pd
import numpy as np
from mining.periodic import *
from mining.preprocessing import *
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import datetime

from beyourself.data import *

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


data = get_necklace_timestr('P120',"2017-08-27 11:56:30.000","2017-08-27 11:57:15.000",reliability=0.01)

# data = pd.read_csv('data/test_fft.csv')

proximity = data['proximity'].as_matrix()
proximity = proximity - 2000
time = data['Time'].as_matrix()


peaks_index = peak_detection(proximity, min_prominence=5)
peaks_time = time[peaks_index]

other_peaks_index = argrelextrema(proximity, np.greater, order=3, mode='clip')
other_peaks_time = time[peaks_index]

# segments = periodic_subsequence(peaks_index, peaks_time, min_length=4,max_length=100, eps=0.1, alpha=0.45,low=400,high=1200)

time_obj = [datetime.datetime.fromtimestamp(t/1000) for t in time]

plt.figure(figsize=(15,5))


# plt.plot(time_obj, original_prox)

plt.plot(time_obj, proximity)


for i in peaks_index:
	plt.plot(time_obj[i], proximity[i], 'r*')

for i in peaks_index:
	plt.plot(time_obj[i], 0, 'g*')


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

plt.show()
