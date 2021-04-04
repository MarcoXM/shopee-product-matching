import sys
sys.path = [
    '../geffnet-20200820'  
] + sys.path
import os
import numpy as np
import pandas as pd
import gc
import cv2
import matplotlib.pyplot as plt
# import cudf, cuml, cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml.neighbors import NearestNeighbors
from misc import getMetric

from dataset import LandmarkDataset, get_transforms
import albumentations
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from model import Enet_Arcface_FINAL

import geffnet
import transformers
import math
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["OMP_NUM_THREADS"] = str(1)
def main():
    COMPUTE_CV = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



    test = pd.read_csv('../test.csv')
    if len(test)>3: 
        COMPUTE_CV = False
    else: 
        print('this submission notebook will compute CV score, but commit notebook will not')

    train = pd.read_csv('../train.csv')
    tmp = train.groupby('label_group').posting_id.agg('unique').to_dict()
    train['target'] = train.label_group.map(tmp)
    print('train shape is', train.shape )


    tmp = train.groupby('image_phash').posting_id.agg('unique').to_dict()
    train['oof'] = train.image_phash.map(tmp)


    train['f1'] = train.apply(getMetric('oof'),axis=1)
    print('CV score for baseline =',train.f1.mean())



    if COMPUTE_CV:
        test = pd.read_csv('../train_fold.csv')
    #     test = test[test.fold==0]
        # test_gf = cudf.DataFrame(test)
        print('Using train as test to compute CV (since commit notebook). Shape is', test.shape )
    else:
        test = pd.read_csv('../test.csv')
        # test_gf = cudf.read_csv('../test.csv')
        print('Test shape is', test.shape )



    tokenizer = transformers.AutoTokenizer.from_pretrained('../input/bert-base-uncased') 

    if not COMPUTE_CV: 
        df_sub = pd.read_csv('../test.csv')

        df_test = df_sub.copy()
        df_test['filepath'] = df_test['image'].apply(lambda x: os.path.join('../', 'test_images', x))

        dataset_test = LandmarkDataset(df_test, 'test', 'test', transforms=get_transforms(img_size=256), tokenizer=tokenizer)
        test_loader = DataLoader(dataset_test, batch_size=16, num_workers=0)

        print(len(dataset_test),dataset_test[0])
    else:
        df_sub = test

        df_test = df_sub.copy()
        df_test['filepath'] = df_test['image'].apply(lambda x: os.path.join('../', 'train_images', x))

        dataset_test = LandmarkDataset(df_test, 'test', 'test', transforms=get_transforms(img_size=256), tokenizer=tokenizer)
        test_loader = DataLoader(dataset_test, batch_size=16, num_workers=4)

        print(len(dataset_test),dataset_test[0][0].shape)


    


    model = Enet_Arcface_FINAL('tf_efficientnet_b0_ns', out_dim=11014).to(device=DEVICE)
    # model = load_model(model, WGT)

    embeds = []

    with torch.no_grad():
        for img, input_ids, attention_mask in tqdm(test_loader): 
            img, input_ids, attention_mask = img.to(device=DEVICE), input_ids.to(device=DEVICE), attention_mask.to(device=DEVICE)
            feat, _ = model(img, input_ids, attention_mask)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    _ = gc.collect()
    image_embeddings = np.concatenate(embeds)
    print('image embeddings shape',image_embeddings.shape)

if __name__ == "__main__":
    main()