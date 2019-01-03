import pandas as pd
import os

subj_list = ["P103","P108","P110","P118","P120"]

list_total = []

for subj in subj_list:
	print(subj)
	df = pd.read_csv("../features/LOSOwild/{}.csv".format(subj))

	print(df.shape)

	if subj == 'P120':
		# removal of bad days
		df = df[(df['date_exp']!=8)&(df['date_exp']!=12)]

		df.drop('date_exp',inplace=True, axis=1)

		print(df.shape)


	df['subj'] = subj

	list_total.append(df)


total_df = pd.concat(list_total)

total_df.to_csv("../features/LOSOwild/total.csv",index=None)


