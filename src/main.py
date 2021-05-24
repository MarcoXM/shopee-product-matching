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
from misc import getMetric, combine_for_cv, combine_for_sub, seed_everything, GradualWarmupSchedulerV2
from sklearn.preprocessing import LabelEncoder
from dataset import ShopeeDataset, get_transforms
import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from model import ShopeeNetV3,ShopeeNet, ShopeeNetV2
from loss import ShopeeScheduler,ArcFaceLossAdaptiveMargin
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from config import model_params,scheduler_params,batch_size
import geffnet
from engine import train_fn, eval_fn
import transformers
import math
from tqdm import tqdm
from time import time


os.environ["TOKENIZERS_PARALLELISM"] = "false"


DIM = (512,512)

NUM_WORKERS = 4
TRAIN_BATCH_SIZE = batch_size
VALID_BATCH_SIZE = 8
EPOCHS = 20


model_version = 'V1'
model_name = model_params['model_name'] #tf_efficientnet_b0-b7

seed_everything(224)

log_name = f"{model_version}_training_log_{model_name}_{model_params['loss_module']}.txt"

if os.path.isfile(log_name):
    os.remove(log_name)

with open(log_name, 'w') as csvfile:
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
    # get adaptive margin
    tmp = np.sqrt(1 / np.sqrt(data['label_group'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05
    
    # Defining Model for specific fold
    if model_version == "V1":
        model = ShopeeNet(**model_params)
    elif model_version == "V2":
        model = ShopeeNetV2(**model_params)
    else:
        model = ShopeeNetV3(**model_params)
    model.to(DEVICE)
    def fetch_loss(loss_type = None):
        if loss_type is None:
            loss = nn.CrossEntropyLoss()
        elif loss_type == 'arcface':
            loss = ArcFaceLossAdaptiveMargin(margins=margins, out_dim = model_params['n_classes'], s=80)
        return loss


    criterion = fetch_loss()
    criterion.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr = scheduler_params['lr_start'])
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, EPOCHS)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)


    #Defining LR SChe
    scheduler = None
        
    # THE ENGINE LOOP
    best_loss = 2 << 13

    for epoch in range(EPOCHS):
        scheduler_warmup.step(epoch - 1)
        train_loss = train_fn(train_loader, model,criterion, optimizer, DEVICE,epoch_th=epoch,scheduler=scheduler)
        valid_loss = eval_fn(valid_loader, model, criterion,DEVICE)


        print('Fold {} | Epoch {}/{} | Training | Loss: {:.4f} | Valid | Loss: {:.4f}'.format(
                fold, epoch + 1, EPOCHS , train_loss['loss'].avg, valid_loss['loss'].avg ))
        with open(log_name, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([fold, epoch + 1, train_loss['loss'].avg, valid_loss['loss'].avg])
        
        if valid_loss['loss'].avg < best_loss:
            best_loss = valid_loss['loss'].avg
            torch.save(model.state_dict(),os.path.join("./models",model_name,f'{model_version}_fold_{fold}_model_{model_params["model_name"]}_IMG_SIZE_{DIM[0]}_{model_params["loss_module"]}.bin'))
            print('best model found for epoch {}'.format(epoch))


    

if __name__ == "__main__":

    if not os.path.isdir(os.path.join("./models",model_name)):
        os.makedirs(os.path.join("./models",model_name))

    for i in range(5):
        main(i)
