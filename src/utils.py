import glob
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd

from config import Config


def get_labels_encode(labels_names):
    labels_names = [[_ for _ in x] for x in labels_names]
    labels_names_flat = [_ for sublist in labels_names for _ in sublist]
    labels_encoded = preprocessing.LabelEncoder()
    labels_encoded.fit(labels_names_flat)
    return np.array([labels_encoded.transform(x) for x in labels_names]) + 1

def create_label_dict(image_folders, labels_list):
    df = pd.read_csv(f"/content/{labels_list[0]}", delimiter='\t')
    for item in labels_list[1:]:
        df = df.append(pd.read_csv(f"/content/{item}", delimiter='\t'), ignore_index=True)
    df = df.dropna(subset=['INPUT:image', 'OUTPUT:output'])
    
    labels_dict = {}
    for folder_name in image_folders:
        for image_name in os.listdir(f'/content/{folder_name}'):
            label = df.loc[df["INPUT:image"] == "".join((f'/{folder_name}/', image_name)), "OUTPUT:output"]
            label = label.tolist()
            if label != []:
                labels_dict.update({f'/content/{folder_name}/' + image_name: label[0]})
    return labels_dict
