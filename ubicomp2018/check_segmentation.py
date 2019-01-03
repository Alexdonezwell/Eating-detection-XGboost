import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
from beyourself import settings
from beyourself.core.util import humanstr_withtimezone_to_datetime_df

from beyourself.core.algorithm import interval_intersect_interval

subj = "P118"


df_segments = pd.read_csv('../data/wild/{}/segments_{}.csv'.format(subj, subj))
df_segments = humanstr_withtimezone_to_datetime_df(df_segments, ['start', 'end'])
interval_segments = list(zip(df_segments['start'].tolist(), df_segments['end'].tolist()))


df_chewing = pd.read_csv(os.path.join(settings.CLEAN_FOLDER, '{}/label/chewing.csv'.format(subj)))
df_chewing = humanstr_withtimezone_to_datetime_df(df_chewing, ['start', 'end'])
interval_chewing = list(zip(df_chewing['start'].tolist(), df_chewing['end'].tolist()))



intersect = interval_intersect_interval(groundtruth=interval_chewing,
										prediction=interval_segments)


index = np.where(np.array(intersect['recall_gt']) == 0)

print(index)

for i in index:
	print(df_chewing['start'].iloc[i])

print(intersect['recall'])

print(intersect['recall_gt'])

print(intersect['precision'])

