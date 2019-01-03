import pandas as pd
import os
import scipy.signal as signal


def smooth(df, columns):
	data = df.copy()
	for c in columns:
		data[c] = signal.savgol_filter(data[c], 9, 3)
	return data