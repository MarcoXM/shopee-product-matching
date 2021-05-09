import pandas as pd 
import numpy as np 
import albumentations as A
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from albumentations.pytorch.transforms import ToTensorV2



def get_transforms(img_size=256, trans_type = 'train'):

    if trans_type == 'train':
        return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        A.Resize(img_size, img_size),
        A.Cutout(max_h_size=int(img_size * 0.4), max_w_size=int(img_size * 0.4), num_holes=1, p=0.5),
        A.Normalize(),
        ToTensorV2(p=1.0),
    ])  
      #      A.Compose([
       #     A.Resize(height=img_size, width=img_size, p=1),
        #    A.RandomSizedCrop(min_max_height=(int(img_size * 0.8), int(img_size * 0.8)), height=img_size, width=img_size, p=0.5),
         #   A.RandomRotate90(p=0.5),
          #  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
           # A.HorizontalFlip(p=0.5),
           # A.VerticalFlip(p=0.5),
           # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
           # A.Normalize(),
           # ToTensorV2(p=1.0),                  
        #], p=1.0)


    return  A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(),
                ToTensorV2(p=1.0)
            ])


class ShopeeDataset(Dataset):
    def __init__(self,
                 csv,
                 split = None,
                 mode = 'train',
                transforms = get_transforms(img_size=256, trans_type = 'train'),
                tokenizer = None):

        self.df = csv.reset_index()
        self.split = split
        self.mode = mode  ## traning mode or test 
        self.transform = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = row.title
        
        image = cv2.imread(row.filepath)
        image = image[:,:,::-1]
        image = image.astype(np.float32)
        # print(image.shape)
        res0 = self.transform(image=image)
        image = res0['image']

        input_ids = torch.rand(1, 1)
        attention_mask = torch.rand(1, 1)
        if self.tokenizer:
            text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
            input_ids = text['input_ids'][0]
            attention_mask = text['attention_mask'][0]
        
        if self.mode == 'test':
            return {
                "images": image,
                "input_ids": input_ids,
                "attention_mask" : attention_mask,
            }
        else:
            target = torch.tensor(row.label_group)
            return {
                "images": image,
                "input_ids": input_ids,
                "attention_mask" : attention_mask,
                "target" : target,
            }
