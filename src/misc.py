import numpy as np 
import os 
import torch
import random

def combine_for_sub(row):
    x = np.concatenate([row.preds, row.preds2, row.preds3])
    return ' '.join( np.unique(x) )

def combine_for_cv(row):
    x = np.concatenate([row.preds, row.preds2, row.preds3])
    return np.unique(x)

def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def fetch_scheduler(optimizer):
    if SCHEDULER =='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True, eps=eps)
    elif SCHEDULER =='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
    elif SCHEDULER =='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
    return scheduler


def seed_everything(seed=224):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

