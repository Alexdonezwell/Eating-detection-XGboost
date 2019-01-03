import os
import sys
import json
import logging
from beyourself import settings
from beyourself.core.util import datetime_from_str, datetime_to_epoch, human_to_epoch
from beyourself.data.label import read_SYNC
from beyourself.data import get_necklace_timestr, get_necklace

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mining.periodic import peak_detection 
from mining.calorieesimator import get_chewing_rate
import pandas as pd
import numpy as np


subj = "P121"
meal = "m0827_1"

# calculate number of prominent chewing per bite
# save the count to a text file
# for plotting and analysis later

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def bite_time_format(t):
    '''
    convert SYNC json to matching human readable format
    '''
    if len(t) > 23: 
        t = t[:23]
    else:
        if len(t) == 19:
            t = t + '.'
        t = t + '0' * (23 - len(t))   

    return t


def read_label_bite(label_bite_path):

    with open(label_bite_path) as f:
        json_data = json.load(f)
        bite_epoch = [human_to_epoch(bite_time_format(b)) 
                        for b in sorted(json_data.keys())]

        bite_chewing_count = [json_data[k] for k in sorted(json_data.keys())]
    
        return json_data, bite_epoch, bite_chewing_count


label_bite_path = "{}/{}/visualize/SYNC/{}/labelbites.json".format(settings.CLEAN_FOLDER, subj, meal)


label_bite_dict, bite_epoch, bite_chewing_count = read_label_bite(label_bite_path)


# create bite ranges (skip the range starting at 'e')
bite_ranges = []
b_previous = None

tmp = list(zip(bite_epoch, bite_epoch[1:]))
for interval, c in zip(tmp, bite_chewing_count):
    if not 'e' in c:
        bite_ranges.append(interval)

    
df = get_necklace(subj, bite_ranges[0][0], bite_ranges[-1][1])

peaks_index = peak_detection(df['proximity'], min_prominence=2)
peaks_time = df.index[peaks_index]

peaks_epoch = np.array([datetime_to_epoch(p) for p in peaks_time])


# calculating number of prominent peaks per bite
for b1, b2 in bite_ranges:
    print(b1)
    print(b2)
    count = len(np.where((peaks_epoch >= b1) & (peaks_epoch < b2))[0])

    print(count)





