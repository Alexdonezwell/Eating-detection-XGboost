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


# P120 debugging
total = 0
folder = settings.RAW_FOLDER + '/VIDEO/P120/'
for d in os.listdir(folder):
    if d.startswith('.'):
        continue
    folder = settings.RAW_FOLDER + '/VIDEO/P120/{}/'.format(d)

    subj_time = 0

    for root, subdirs, files in os.walk(folder):
        for f in files:
            if f.endswith("mov"):
                fullpath = os.path.join(root, f)

                # print(fullpath)
                try:
                    length = getLength(fullpath)

                    subj_time += length

                except:
                    print("skipping {}".format(f))

    total += subj_time

    print("================")
    print("Day: {}".format(d))
    print("total_hour: {}".format(subj_time/3600))






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


# # ===================== count unique days ============================
# total = 0

# for s in subj:
#   folder = os.path.join(settings.CLEAN_FOLDER, s + '/necklace/data')

#   unique_days = set()

#   total_sum = 0
#   for f in os.listdir(folder):    
#     day = f[3:5]
#     unique_days.add(day)

#   print(s, len(unique_days))

#   total += len(unique_days)

# print(total)





# # ================ NUMBER OF ROWS IN RAW DATA ===================
# print("======================================")
# print("Counting necklace data from raw folder")
# total = 0
# for s in subj:
#     folder = "//Volumes/ANDREY/BeYourself/BeYourself/RAW/NECKLACE/{}".format(s)

#     if not os.path.isdir(folder):
#         raise ValueError("Folder does not exist")

#     n_rows = 0
#     # finding all csv files in this folder
#     # and sum up the number of rows
#     for path, subdir, files in os.walk(folder):
#         for f in files:
#             if f.endswith('csv'):
#               try:
#                   with open(os.path.join(path, f)) as txtfile:
#                       size = sum(1 for _ in txtfile)
#                       n_rows += size
#               except:
#                   pass

#     print(s, n_rows/50/3600)

#     total += n_rows

# print("Total sum: {}".format(total/50/3600))


# # ================ NUMBER OF ROWS IN TEMP DATA ===================
# # AFTER RUNNING microsecond interpolation
# print("======================================")
# print("Counting necklace data from temporary folder")
# total = 0
# for s in subj:
#     folder = "/Volumes/BeYourself/BeYourself/TEMP/finaltemp/{}".format(s)

#     if not os.path.isdir(folder):
#         raise ValueError("Folder does not exist")

#     n_rows = 0
#     # finding all csv files in this folder
#     # and sum up the number of rows
#     for path, subdir, files in os.walk(folder):
#         for f in files:
#             if f.endswith('csv'):
#               try:
#                   with open(os.path.join(path, f)) as txtfile:
#                       size = sum(1 for _ in txtfile)
#                       n_rows += size
#               except:
#                   pass

#     print(s, n_rows/50/3600)

#     total += n_rows

# print("Total from temp folder: {}".format(total/50/3600))


# for s in subj:

#   df = pd.read_csv(os.path.join(settings.CLEAN_FOLDER, s+'/label/inclusion.csv'))

#   sum_time = timedelta(seconds=0)

#   for i in range(df.shape[0]):
#       timedelta = datetime_from_str(df['end'].iloc[i]) - datetime_from_str(df['start'].iloc[i])
#       sum_time += timedelta

#   print(sum_time.seconds/3600)
