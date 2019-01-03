import pandas as pd
import os
from scipy.stats import zscore
from os.path import join


def standardize_zscore(data, selected_columns):
    """normalize the whole dataset to make the system cleaner
    only normalize for columns belong to columns list
    Parameters
    ----------
        data:               dataFrame
        selected_columns:   list of columns which will be standardized
    Return
    ------
        dataZ           dataFrame
    """

    dataZ = data.copy()

    for col_header in selected_columns:
        dataZ[col_header] = zscore(dataZ[col_header])

    return dataZ



subj_list = ["P000", "P106", "P108", "P110", "P111", "P113", "P116", "P119", "P120", "P121"]
list_total = []


folder = '/Users/apple/Downloads/necklace_inlab/ubicomp2018/beyourself/CLEAN/Label_IS'


for subj in subj_list:
	df = pd.read_csv(join(folder, "inlab_feature_{}.csv".format(subj)))
	df['subj'] = subj

	list_total.append(df)


total_df = pd.concat(list_total)


# normalize data across all participants

selected_columns = list(total_df.keys())[:-7]
total_df = standardize_zscore(total_df, selected_columns)

total_df.to_csv(join(folder,'inlab_feature.csv'), index=None)
