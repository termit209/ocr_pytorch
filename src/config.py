import glob
import os
from sklearn import preprocessing
import numpy as np
import collections
import os
import pandas as pd

df_part1 = pd.read_csv("/content/handwritten_rus/LABELED/assignments_from_pool_601263__05-12-2020.tsv", delimiter='\t' )
df_part2 = pd.read_csv("/content/handwritten_rus/LABELED/assignments_from_pool_615470__05-12-2020.tsv", delimiter='\t' )
df = df_part1.append(df_part2, ignore_index=True)
df = df.dropna(subset=['INPUT:image', 'OUTPUT:output'])

labels_dict = {}
for image_name in os.listdir("/content/HANDWRITTEN/werner_1000_part_1_181120"):
  label = df.loc[df["INPUT:image"] == "".join(("/HANDWRITTEN/werner_1000_part_1_181120/", image_name)), "OUTPUT:output"]
  label = label.tolist()
  if label != []: 
    labels_dict.update({'/content/HANDWRITTEN/werner_1000_part_1_181120/' + image_name:label[0]})

for image_name in os.listdir("/content/HANDWRITTEN/werner_1001_2000_part_2_241120"):
  label = df.loc[df["INPUT:image"] == "".join(("/HANDWRITTEN/werner_1001_2000_part_2_241120/", image_name)), "OUTPUT:output"]
  label = label.tolist()
  if label != []: 
    labels_dict.update({"/content/HANDWRITTEN/werner_1001_2000_part_2_241120/" + image_name:label[0]})
labels_dict


import glob
import os
from sklearn import preprocessing
import numpy as np


#DATASET_PATH = "/content/HANDWRITTEN/werner_1000_part_1_181120/"
BATCH_SIZE = 32
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
NUM_WORKERS = 4
EPOCHS = 201
DEVICE = 'cuda'
MAX_LENGHT = 32
PRINT_STATISTIC = True

LABELS_DICT_KEYS = list(labels_dict.keys())
IMAGES=LABELS_DICT_KEYS
# to look like '6bnnm'
LABELS_NAMES =list(labels_dict.values())
# to look like ['g', 'p', 'x', 'n', 'g']
LABELS_NAMES = [[_ for _ in x] for x in LABELS_NAMES]
LABELS_NAMES_FLAT = [_ for sublist in LABELS_NAMES for _ in sublist]
labels_encoded = preprocessing.LabelEncoder()
labels_encoded.fit(LABELS_NAMES_FLAT)
# print(labels_encoded.classes_)
# keep 0 for unknown
LABELS_ENCODED = np.array([labels_encoded.transform(x) for x in LABELS_NAMES]) +1
#print(LABELS_ENCODED)
#print(len(labels_encoded.classes_))

# some statistic
if PRINT_STATISTIC:
  print('size dataset: %d' %len(LABELS_NAMES))
  print('number symbols: %d' %len(labels_encoded.classes_))

  lenght_labels = np.array([len(LABELS_NAME) for LABELS_NAME in LABELS_NAMES])
  print('longest labels size: ', np.sort(lenght_labels)[-5:])
  print('shorted labels size: ', np.sort(lenght_labels)[0:5])
  print('mean label size: ', sum(lenght_labels) / len(lenght_labels))
  print('number lenght bigger than MAX_LENGHT: ', sum(lenght_labels > MAX_LENGHT))

  cymbol_counter = collections.Counter(LABELS_NAMES_FLAT)
  print('most frequent symbols: ', cymbol_counter.most_common(10))
  print('least frequent symbols: ', cymbol_counter.most_common()[:-10:-1] )