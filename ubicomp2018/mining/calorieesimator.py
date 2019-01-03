import numpy as np
from mining.periodic import peak_detection 


def get_chewing_rate(df):
	proximity = df.proximity.as_matrix()

	peaks_index = peak_detection(proximity, min_prominence=2)
	chewing_duration = np.diff(df.index[peaks_index]).astype(int)

	chewing_rate = 1e9/np.mean(chewing_duration)
	chewing_rate_std = 1e9/np.std(chewing_duration)
	num_chewing = len(peaks_index)

	return chewing_rate, chewing_rate_std, num_chewing