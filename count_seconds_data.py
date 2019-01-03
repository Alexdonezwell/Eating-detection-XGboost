from beyourself import settings
import os
from beyourself.core.util import datetime_from_str
import pandas as pd
from datetime import timedelta
import subprocess


def getLength(input_video):
    duration = subprocess.check_output(['ffprobe', '-i', input_video, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")])
    return float(duration)

subj = ['P103',
        'P105',
        'P107',
        'P108',
        'P110',
        'P114',
        'P116',
        'P118',
        'P120',
        'P121']


# subj = ['P114',
#         'P116',
#         'P118',
#         'P120',
#         'P121']


# total = 0

# for s in subj:
#     folder = settings.RAW_FOLDER + '/VIDEO/{}/'.format(s)

#     subj_time = 0

#     for root, subdirs, files in os.walk(folder):
#         for f in files:
#             if f.endswith("mov"):
#                 fullpath = os.path.join(root, f)

#                 # print(fullpath)
#                 try:
#                     length = getLength(fullpath)

#                     subj_time += length

#                 except:
#                     print("skipping {}".format(f))

#     print(s, subj_time/3600)
#     total += subj_time

# print(total/3600)

# # count
# total = 0

# for s in subj:
#   folder = os.path.join(settings.CLEAN_FOLDER, s + '/necklace/data')

#   total_sum = 0
#   for f in os.listdir(folder):
#       with open(os.path.join(folder, f)) as txtfile:
#           size = sum(1 for _ in txtfile)
#           total_sum += size

#   print(s, total_sum/50/3600)
#   total += total_sum
# print(total)


# ===================== count unique days ============================
total = 0

for s in subj:
  folder = os.path.join(settings.CLEAN_FOLDER, s + '/necklace/data')

  unique_days = set()

  total_sum = 0
  for f in os.listdir(folder):    
    day = f[3:5]
    unique_days.add(day)

  print(s, len(unique_days))

  total += len(unique_days)

print(total)





# # ================ raw data (before removing 1970) ===================
# print("getting all available data")
# total = 0
# for s in subj:
#     folder = os.path.join("/Volumes/ANDREY/BeYourself/BeYourself/RAW/NECKLACE", s)

#     total_sum = 0
#     for path, subdir, files in os.walk(folder):
#         for f in files:
#             if f == 'start.log' or f == 'testSpeed.txt':
#                 continue

#             # print(os.path.join(path, f))
#             try:
#                 with open(os.path.join(path, f)) as txtfile:
#                     size = sum(1 for _ in txtfile)
#                     total_sum += size
#             except:
#                 pass

#     print(s, total_sum/50/3600)

#     total += total_sum

# print(total/50/3600)



# for s in subj:

#   df = pd.read_csv(os.path.join(settings.CLEAN_FOLDER, s+'/label/inclusion.csv'))

#   sum_time = timedelta(seconds=0)

#   for i in range(df.shape[0]):
#       timedelta = datetime_from_str(df['end'].iloc[i]) - datetime_from_str(df['start'].iloc[i])
#       sum_time += timedelta

#   print(sum_time.seconds/3600)
