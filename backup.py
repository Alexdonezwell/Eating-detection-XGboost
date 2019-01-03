from mining.feature import get_feature
import pandas as pd
from beyourself.data import get_necklace
from beyourself import settings
from beyourself.core.util import *
from beyourself.core.ml import undersample

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import time


df_segments = pd.read_csv('segments_P120.csv')

df_segments = humanstr_to_datetime_df(df_segments,['start','end'])


df_segments = undersample(df_segments, 'chewing_gt',0,5)


print(df_segments.shape)
print(np.mean(df_segments['chewing_gt'].as_matrix()))


list_total_feature = []

N = df_segments.shape[0]

start_time = time.time()

for i in range(N):
	if i % 1000 == 1:
		print("Sample {}, elapsed {}".format(i, time.time() - start_time))
		feature = pd.concat(list_total_feature)
		feature.to_csv('feature_P120.csv', index=None)


	start = datetime_to_epoch(df_segments['start'][i])
	end = datetime_to_epoch(df_segments['end'][i])

	try:
		df = get_necklace('P120', start, end, reliability=0.01)
		f = get_feature(df)

		f['start'] = df_segments['start'][i]
		f['end'] = df_segments['end'][i]
		f['pmin'] = df_segments['pmin'][i]
		f['pmax'] = df_segments['pmax'][i]
		f['eps'] = df_segments['eps'][i]
		f['length'] = df_segments['length'][i]
		f['hour'] = df_segments['start'][i].hour
		f['label'] = df_segments['chewing_gt'][i]
		f['date_exp'] = df_segments['date_exp'][i]

		list_total_feature.append(f)

	except ValueError:
		print("Error generating feature, skipping")


	

feature = pd.concat(list_total_feature)
feature.to_csv('feature_P120.csv', index=None)

