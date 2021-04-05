from misc import AverageMeter
from tqdm import tqdm
import torch


def train_fn(dataloader,model,criterion,optimizer,device,epoch_th, scheduler = None):
    model.train()
    loss_score = AverageMeter()

    tk0 = tqdm(enumerate(dataloader), total=len(dataloader))
    for step,data in tk0:
        


        images = data['images'].to(device)
        targets = data['target'].to(device)
        batch_size = images.shape[0]

        optimizer.zero_grad()

        output = model(images,targets)
        
        loss = criterion(output,targets)
        
        loss.backward()
        optimizer.step()
        
        loss_score.update(loss.detach().item(), batch_size)
        tk0.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch_th, LR = optimizer.param_groups[0]['lr'])
        
    if scheduler is not None:
            scheduler.step()

        
    return {
        "loss" : loss_score
    }


def eval_fn(data_loader,model,criterion,device):
    
    loss_score = AverageMeter()
    
    model.eval()
    tk0 = tqdm(enumerate(data_loader), total=len(data_loader))
    
    with torch.no_grad():
        for step,data  in tk0:
            

            images = data['images'].to(device)
            targets = data['target'].to(device)
            batch_size = images.shape[0]

            images = images.to(device)
            targets = targets.to(device)

            output = model(images,targets)

            loss = criterion(output,targets)
            
            loss_score.update(loss.detach().item(), batch_size)
            tk0.set_postfix(Eval_Loss=loss_score.avg)
            
    return {
        "loss" : loss_score
    }