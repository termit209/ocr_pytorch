import os
import pandas as pd

labels_list = ['handwritten_rus/LABELED/assignments_from_pool_601263__05-12-2020.tsv', # Werner
               'handwritten_rus/LABELED/assignments_from_pool_615470__05-12-2020.tsv', # Werner
               'handwritten_rus/LABELED/assignments_from_pool_613899__05-12-2020.tsv', # Kovshov
               'handwritten_rus/LABELED/assignments_from_pool_613887__05-12-2020.tsv'  # Stepanischeva
               ]

df = pd.read_csv(f"/content/{labels_list[0]}", delimiter='\t')
for item in labels_list[1:]:
    df = df.append(pd.read_csv(f"/content/{item}", delimiter='\t'), ignore_index=True)

df = df.dropna(subset=['INPUT:image', 'OUTPUT:output'])


def update_dict(l_dict, folder_name):
    for image_name in os.listdir(f'/content/{folder_name}'):
        label = df.loc[df["INPUT:image"] == "".join((f'/{folder_name}/', image_name)), "OUTPUT:output"]
        label = label.tolist()
        if label != []:
            l_dict.update({f'/content/{folder_name}/' + image_name: label[0]})


labels_dict = {}
image_folders = ['HANDWRITTEN/werner_1000_part_1_181120',
                 'HANDWRITTEN/werner_1001_2000_part_2_241120',
                 'HANDWRITTEN/kovshov_4461_part_1_261120',
                 'HANDWRITTEN/arina_580_part_1_231120',
                 ]
for folder_name in image_folders:
    update_dict(labels_dict, folder_name)