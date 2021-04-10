import sys
sys.path = [
    '../geffnet-20200820'  
] + sys.path
import os
import numpy as np
import pandas as pd
import gc
import csv
import cv2
import matplotlib.pyplot as plt
# import cudf, cuml, cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors
from misc import getMetric, combine_for_cv, combine_for_sub, seed_everything
from sklearn.preprocessing import LabelEncoder
from dataset import ShopeeDataset, get_transforms
import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from model import Enet_Arcface_FINAL,ShopeeNet
from loss import fetch_loss, ShopeeScheduler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from config import model_params,scheduler_params
import geffnet
from engine import train_fn, eval_fn
import transformers
import math
from tqdm import tqdm
from time import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"


DIM = (512,512)

NUM_WORKERS = 4
TRAIN_BATCH_SIZE = model_params['batch_size']
VALID_BATCH_SIZE = 16
EPOCHS = 30

model_name = model_params['model_name'] #efficientnet_b0-b7

seed_everything(224)

log_name = f"training_log_{model_name}_{model_params['loss_module']}.txt"

if os.path.isfile(log_name):
    os.remove(log_name)

with open("training_log.txt", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fold','epoch', 'loss', 'val_loss'])


def main(fold):
    COMPUTE_CV = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    data = pd.read_csv('../train_fold.csv')
    data['filepath'] = data['image'].apply(lambda x: os.path.join('../', 'train_images', x))

    target_encoder = LabelEncoder()

    data['label_group'] = target_encoder.fit_transform(data['label_group'])
    
    train = data[data['fold']!=fold].reset_index(drop=True)
    valid = data[data['fold']==fold].reset_index(drop=True)
    # Defining DataSet
    train_dataset = ShopeeDataset(
        csv=train,
        transforms=get_transforms(img_size=DIM[0], trans_type = 'train'),
        mode = 'train',
    )
        
    valid_dataset = ShopeeDataset(
        csv=valid,
        transforms=get_transforms(img_size=DIM[0], trans_type = 'valid'),
        mode = 'train',
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=True,
        drop_last=True,
        num_workers=NUM_WORKERS
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    
    # Defining Model for specific fold
    model = ShopeeNet(**model_params)
    model.to(DEVICE)

    criterion = fetch_loss()
    criterion.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr = scheduler_params['lr_start'])
    
    #Defining LR SCheduler
    scheduler = ShopeeScheduler(optimizer,**scheduler_params)
        
    # THE ENGINE LOOP
    best_loss = 2 << 13

    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model,criterion, optimizer, DEVICE,epoch_th=epoch,scheduler=scheduler)
        valid_loss = eval_fn(valid_loader, model, criterion,DEVICE)


        print('Fold {} | Epoch {}/{} | Training | Loss: {:.4f} | Valid | Loss: {:.4f}'.format(
                fold, epoch + 1, EPOCHS , train_loss['loss'].avg, valid_loss['loss'].avg ))
        with open(log_name, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([fold, epoch + 1, train_loss['loss'].avg, valid_loss['loss'].avg])
        
        if valid_loss['loss'].avg < best_loss:
            best_loss = valid_loss['loss'].avg
            torch.save(model.state_dict(),os.path.join("./models",model_name,f'fold_{fold}_model_{model_params["model_name"]}_IMG_SIZE_{DIM[0]}_{model_params["loss_module"]}.bin'))
            print('best model found for epoch {}'.format(epoch))


    

if __name__ == "__main__":

    if not os.path.isdir(os.path.join("./models",model_name)):
        os.makedirs(os.path.join("./models",model_name))

    for i in range(5):
        main(i)