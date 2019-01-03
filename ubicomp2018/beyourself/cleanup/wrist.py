import pandas as pd
import os


def concatenate(arg):

	path = arg['path']
	outpath = arg['outpath']

	total_list = []
	for f in os.listdir(path):
		if not f.startswith("."):
			df = pd.read_csv(os.pat.join(path, f),index_col=False)
			total_list.append(df)

	total_df = pd.concat(total_list)

	total_df.to_csv(outpath, index=False)

