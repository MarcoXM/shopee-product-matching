import pandas as pd 
import numpy as np 
import albumentations as A
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2



def get_transforms(img_size=256):
    return  A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize()
            ])


class LandmarkDataset(Dataset):
    def __init__(self,
                 csv,
                 split,
                 mode,
                transforms = get_transforms(img_size=256),
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
        image = image[:, :, ::-1]
        
        res0 = self.transform(image=image)
        image0 = res0['image'].astype(np.float32)
        image = image0.transpose(2, 0, 1)        

        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        if self.mode == 'test':
            return torch.tensor(image), input_ids, attention_mask
        else:
            return torch.tensor(image), input_ids, attention_mask, torch.tensor(row.label_group)