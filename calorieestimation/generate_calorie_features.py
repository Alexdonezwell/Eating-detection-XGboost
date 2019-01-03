from beyourself.data import get_necklace_timestr, get_necklace
from beyourself.data.label import read_SYNC
from beyourself.core.util import datetime_to_epoch, humanstr_withtimezone_to_datetime_df
from beyourself import settings
from mining.calorieesimator import get_chewing_rate
import logging
import numpy as np
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)



subj = "P103"

valid_meals = [ 'm0627_1',
				'm0628_1',
				'm0628_2',
				'm0629_1',
				'm0630_1',
				'm0701_2']


# subj = 'P105'
# valid_meals = [ 'm0624_1',
# 				'm0624_2',
# 				'm0626_1',
# 				'm0626_2',
# 				'm0702_1',
# 				'm0702_2',
# 				'm0703_2']


# subj = 'P107'

# valid_meals = [ 'm0712_1',
# 				'm0717_1',
# 				'm0718_1']

# subj = 'P108'
# valid_meals = [ 'm0804_1',
# 				'm0804_2',
# 				'm0805_2',
# 				'm0806_1',
# 				'm0806_2',
# 				'm0807_1_a',
# 				'm0808_1',
# 				'm0809_1',
# 				'm0810_2',
# 				'm0811_1',
# 				'm0812_1',
# 				'm0812_2',
# 				'm0813_1',
# 				'm0814_1',
# 				'm0815_1']

# subj = 'P110'
# valid_meals = [ 'm0805_2',
# 				'm0806_1']


# subj = 'P114'

# valid_meals = [ 'm0809_1_2',
# 				'm0810_1',
# 				'm0810_2',
# 				'm0811_1',
# 				'm0811_2',
# 				'm0814_2']


# subj = 'P116'
# valid_meals = [ 'm0812_1',
# 				'm0812_2_a',
# 				'm0813_4',
# 				'm0821_2',
# 				'm0821_3']


# subj = 'P118'
# valid_meals = [ 'm0819_1',
# 				'm0821_1',
# 				'm0821_2_1',
# 				'm0821_2_2',
# 				'm0822_1',
# 				'm0828_2',
# 				'm0829_1',
# 				'm0829_2',
# 				'm0830_1',
# 				'm0830_2',
# 				'm0831_1']

# subj = 'P120'
# valid_meals = [ 'm0824_1',
# 				'm0824_2',
# 				'm0826_1',
# 				'm0826_4',
# 				'm0827_1',
# 				'm0827_2',
# 				'm0827_3',
# 				'm0828_2',
# 				'm0829_1',
# 				'm0829_2',
# 				'm0830_1',
# 				'm0830_2',
# 				'm0831_1',
# 				'm0902_1',
# 				'm0902_2',
# 				'm0902_3',
# 				'm0903_1',
# 				'm0903_3',
# 				'm0904_3',
# 				'm0905_1',
# 				'm0905_2']


chewing_rate_list = []
chewing_rate_std_list = []

for m in valid_meals:

	# get chewing groundtruth
	gt = read_SYNC("{}/{}/visualize/SYNC/{}/labelchewing.json".format(settings.CLEAN_FOLDER, subj, m))
	
	print(gt)

	rate_list = []
	rate_std_list = []
	for i in range(gt.shape[0]):
		df = get_necklace(subj, datetime_to_epoch(gt['start'].iloc[i]),
								datetime_to_epoch(gt['end'].iloc[i]))

		print(df.shape)

		if df.empty or df.shape[0]<3:
			print("EMPTY DATAFRAME, DOUBLE CHECK!!!")
			continue

		rate, rate_std = get_chewing_rate(df)

		rate_list.append(rate)
		rate_std_list.append(rate_std)

	chewing_rate_list.append(np.mean(np.array(rate_list)))
	chewing_rate_std_list.append(np.mean(np.array(rate_std_list)))


print(chewing_rate_list)

print(chewing_rate_std_list)

df_result = pd.DataFrame({	'meal':valid_meals, 
							'chewing_rate':chewing_rate_list,
							'chewing_rate_std': chewing_rate_std_list},
							columns=['meal', 'chewing_rate', 'chewing_rate_std'])

df_result.to_csv("mealfeatures/{}.csv".format(subj),index=None)
	# get features for this part of data
