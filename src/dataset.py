import torch 
import numpy as np
import cv2
from albumentations import *
from random import randint


class OcrDataset: 
    """
    Handle images for dataloader.
    Apply resize and augmentations.
    Put attention to resize PIL: first goes width then height
    """
    def __init__(self, image_path, labels, resize=None, random_flag=True, use_augment=True):
        self.image_path = image_path
        self.labels = labels
        self.resize = resize
        self.use_augment = use_augment
        self.random_flag = random_flag
        if self.use_augment:
        #self.augmentations = Compose([Normalize(always_apply=True)])
            self.augmentations = Compose([Normalize(always_apply=True),
                                          OneOf([
                                                 RandomBrightness(limit=0.8, p=0.7),
                                                 RandomContrast(limit=0.8, p=0.7),
                                                 RandomBrightnessContrast(p=0.5),]),
                                          OneOf([
                                                 ToSepia(p=1),
                                                 ToGray(p=1),]),])
        else:
            self.augmentations = Compose([Normalize(always_apply=True)])
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
            if self.use_padding:
                image = self.make_padding(image, self.resize, self.random_flag)
            else:
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
    
    def make_padding(self, image, resize, random_flag):
        desired_height, desired_width = resize[0], resize[1]
        proportion_declared = desired_width / desired_height
        image_proportion = image.shape[1] / image.shape[0]
        new_im = np.zeros((desired_height, desired_width, 3))
        h, w = image.shape[:2]
        if w < desired_width and h < desired_height:
            if not random_flag:
                h_begin = desired_height // 2 - h // 2
                w_begin = desired_width // 2 - w // 2
            else:
                h_begin = randint(0, desired_height - h)
                w_begin = randint(0, desired_width - w)
            new_im[h_begin:h_begin + h, w_begin:w_begin + w, :] = image
        else:
            if proportion_declared >= image_proportion:
                image_copy = cv2.resize(image, (int(desired_height / h * w), desired_height))
                h_begin = 0
                w_begin = desired_width // 2 - image_copy.shape[1] // 2
            else:
                image_copy = cv2.resize(image, (desired_width, int(desired_width / w * h)))
                h_begin = desired_height // 2 - image_copy.shape[0] // 2
                w_begin = 0    
            new_im[h_begin:h_begin + image_copy.shape[0], 
              w_begin:w_begin + image_copy.shape[1], :] = image_copy
        return new_im
