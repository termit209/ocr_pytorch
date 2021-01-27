import torch 
import numpy as np
import cv2
from albumentations import *


class OcrDataset: 
    """
    Handle images for dataloader.
    Apply resize and augmentations.
    Put attention to resize PIL: first goes width then height
    """
    def __init__(self, image_path, labels, resize=None, random_flag=False):
        self.image_path = image_path
        self.labels = labels
        self.resize = resize
        self.augmentations = Compose([
           Normalize(always_apply=True),
            OneOf([
               RandomBrightness(limit=0.8, p=0.7),
               RandomContrast(limit=0.8, p=0.7),
               RandomBrightnessContrast(p=0.5),
               RandomGamma(gamma_limit=(80, 300), p=0.7)
           ]),
               OneOf([
                   ToSepia(p=1),
                    ToGray(p=1),
               ]),
           ])

        self.label_size = [item.shape[0] for item in self.labels]
        self.max_lenght = max(self.label_size)

    def __len__(self):
        return len(self.image_path)
        
    def __getitem__(self, item):
        image = cv2.imread(self.image_path[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_lenght = self.label_size[item]
        labels = self.labels[item]
        label_pad = np.zeros(self.max_lenght)
        label_pad[:input_lenght] = labels
        if self.resize is not None:
            image = cv2.resize(image, (self.resize[1], self.resize[0]))
        image = np.array(image)
        augmented_image = self.augmentations(image=image)
        image = augmented_image["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(label_pad, dtype=torch.long),
            "label_size": torch.tensor(input_lenght, dtype=torch.long)
            }
