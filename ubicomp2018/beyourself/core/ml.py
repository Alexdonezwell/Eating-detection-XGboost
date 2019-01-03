import numpy as np
import logging


logger = logging.getLogger(__name__)



def undersample(df, column_label, target, ratio=5):
	'''
	Undersample a dataframe to avoid imbalance problem

	Parameters
	----------

	df: pandas dataframe 
		data to be undersampled

	column_label: str
		which column is the label

	target: int
		which class will be undersampled

	'''

	label = df[column_label].as_matrix()

	if not target in np.unique(label):
		raise ValueError("Target is outside the range")


	fat_class = np.where(label==target)[0]
	skinny_class = np.where(label!=target)[0]

	N = len(fat_class)

	logger.info("Number of the redundant samples: {}".format(N))
	rand_index = np.random.randint(N, size=int(N//ratio))

	reduced = fat_class[rand_index]

	undersampled = np.sort(np.concatenate([reduced, skinny_class]))

	return df.iloc[undersampled].reset_index(drop=True)
