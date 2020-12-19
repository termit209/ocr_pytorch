import torch 
import numpy as np
from random import randint
from albumentations import Compose, Normalize, MedianBlur, GaussianBlur, \
    MotionBlur, GaussNoise, MultiplicativeNoise, Cutout, \
    CoarseDropout, GridDistortion, ElasticTransform, \
    RandomBrightness, RandomContrast, RandomBrightnessContrast, \
    ShiftScaleRotate, IAAPiecewiseAffine

from PIL import Image


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
        self.random_flag = random_flag
        self.augmentations = Compose(
            [
            Normalize(always_apply=True),
            # MedianBlur(blur_limit=5, p=0.2),
            # GaussianBlur(p=0.2),
            # MotionBlur(p=0.2),
            # GaussNoise(var_limit=5. / 255., p=0.2),
            # MultiplicativeNoise(p=0.2),
            # Cutout(num_holes=8, max_h_size=10, max_w_size=10,p=0.2),
            # CoarseDropout(max_holes=8, max_height=10, max_width=10,p=0.2),
            # GridDistortion(p=0.2),
            # ElasticTransform(sigma=50, alpha=1, alpha_affine=10, p=0.2),
            # RandomBrightness(p=0.2),
            # RandomContrast(p=0.2),
            # RandomBrightnessContrast(p=0.2),
            # IAAPiecewiseAffine(p=0.2),
            # ShiftScaleRotate(shift_limit=0.1,
            #                     scale_limit=0.1,
            #                     rotate_limit=30,
            #                     p=0.2)             
            ]                                    
                                            
                                            )
        
    def __len__(self):
        return len(self.image_path)
        
    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        labels = self.labels[item]
        
        if self.resize is not None:
            image = self.make_padding(image, self.resize, self.random_flag)
            
        
        image = np.array(image)
        augmented_image = self.augmentations(image=image)
        image = augmented_image["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        
        return {
            "images": torch.tensor(image, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def make_padding(self, image, resize, random_flag):
        desired_height, desired_width = resize[0], resize[1]
        proportion_declared = desired_width / desired_height
        image_proportion = image.size[0] / image.size[1]
        new_im = Image.new("RGB", (desired_width, desired_height))
        if image.size[0] < desired_width and image.size[1] < desired_height:
            if random_flag:
                new_im.paste(image, (desired_width // 2 - image.size[0] // 2,
                                     desired_height // 2 - image.size[1] // 2))
            else:
                new_im.paste(image, (randint(0, desired_width - image.size[0]),
                                     randint(0, desired_height - image.size[1])))
        else:
            if proportion_declared >= image_proportion:
                image = image.resize((int(desired_height / image.size[1] * image.size[0]),
                                      desired_height), resample=Image.BILINEAR)
                new_im.paste(image, ((desired_width - image.size[0]) // 2, 0))
            else:
                image = image.resize((desired_width,
                                      int(desired_width / image.size[0] * image.size[1])),
                                     resample=Image.BILINEAR)
                new_im.paste(image, (0, (desired_height - image.size[1]) // 2))
        return new_im


class SynthCollator(object):
    
    def __call__(self, batch):
        label_size = [item['labels'].shape for item in batch]

        max_label_lenght = max(label_size) #torch.Size(np.array(max(labels)) + 7) #max(labels)
        for item in batch:
          target = torch.zeros(max_label_lenght)
          source_lenght = list(item['labels'].shape)[0]
          target[:source_lenght]=item['labels']
          item['labels']=target
        imgs = [item['images'] for item in batch]
        labels = [item['labels'] for item in batch]
        imgs=torch.stack(imgs)
        labels = torch.stack(labels)
        item={'images':imgs,'labels': labels, 'label_size':torch.flatten(torch.tensor(label_size))}
        return item
