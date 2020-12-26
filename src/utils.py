import glob
import os
import numpy as np
from sklearn import preprocessing

from config import Config


def get_labels_encode(config: Config):
    images = glob.glob(os.path.join(config.dataset_path, "*.png"))
    labels_names = [x.split('/')[-1][:-4] for x in images]
    labels_names = [[_ for _ in x] for x in labels_names]
    labels_names_flat = [_ for sublist in labels_names for _ in sublist]
    labels_encoded = preprocessing.LabelEncoder()
    labels_encoded.fit(labels_names_flat)
    return np.array([labels_encoded.transform(x) for x in labels_names]) + 1

