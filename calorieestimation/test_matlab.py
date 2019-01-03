import sys, os
from beyourself.data import get_necklace_timestr

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from mining.periodic import peak_detection
from beyourself import settings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

df = get_necklace_timestr('P120', "2017-08-27 11:56:30.000", "2017-08-27 11:58:00.000")

proximity = df.proximity.as_matrix()

peaks_index = peak_detection(proximity, min_prominence=2)