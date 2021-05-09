import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import transformers
import geffnet
import math 
from loss import AdaCos, ArcMarginProduct,AddMarginProduct, fetch_loss, ArcMarginProduct_subcenter
from torch.optim.lr_scheduler import _LRScheduler
from config import *
import timm
from torch.nn.utils import spectral_norm

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k  ## k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine 
    
sigmoid = torch.nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

    
 
    
class ShopeeNetV3(nn.Module):
    def __init__(self, n_classes,
                       model_name='tf_efficientnet_b4',
                       use_fc=False,
                       fc_dim=512,
                       dropout=0.0,
                       margin=0.50,
                       loss_module='softmax',
                       s=30.0,
                       ls_eps=0.0,
                       theta_zero=0.785,
                       pretrained=True):
        super(ShopeeNetV3, self).__init__()
     #   self.bert = transformers.AutoModel.from_pretrained('../input/bert-base-uncased')
        self.enet = timm.create_model(model_name, pretrained=True)
        self.feat = nn.Linear(self.enet.classifier.in_features, 512) # self.bert.config.hidden_size
        self.swish = Swish_module()
        self.metric_classify = ArcMarginProduct_subcenter(512, n_classes)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)
 
    def forward(self, x, targets ):  #input_ids, attention_mask

        x = self.extract(x)
    #    text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        # x = torch.cat([x, text], 1)
        logits_m= self.metric_classify(self.swish(self.feat(x)))
        return logits_m


class Attn(nn.Module):
    def __init__(self,channels,reduction_attn = 8, reduction_sc = 2): # Ratio should be 4: 1
        super(Attn,self).__init__()
        self.channels_attn = channels//reduction_attn # reduce cal c
        self.channels_sc = channels//reduction_sc # reduce cal 4c
        
        #ATTN consisted of query,key and value, structure design
        
        self.qconv = spectral_norm(nn.Conv2d(channels,self.channels_attn,kernel_size = 1,bias=False)) # 1 x 1 filter
        self.kconv = spectral_norm(nn.Conv2d(channels,self.channels_attn,kernel_size = 1,bias=False))
        self.vconv = spectral_norm(nn.Conv2d(channels,self.channels_sc,kernel_size = 1,bias=False)) # weight 
        self.attnconv = spectral_norm(nn.Conv2d(self.channels_sc,channels,kernel_size = 1,bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # initializing weights
        nn.init.orthogonal_(self.qconv.weight.data)
        nn.init.orthogonal_(self.kconv.weight.data)
        nn.init.orthogonal_(self.vconv.weight.data)
        nn.init.orthogonal_(self.attnconv.weight.data)
        
    def forward(self,x):
        
        batch,_,h,w = x.size() # original channel size is not important
        
        qx = self.qconv(x).view(batch,self.channels_attn,-1) # b x oc x h x w >>>>> b x c x hw 
        kx = F.max_pool2d(self.kconv(x),2).view(batch,self.channels_attn,-1) # b x c x h/2 x w/2 >>>>> b x c x hw/4
        
        attn = torch.bmm(kx.permute(0,2,1),qx) # b x hw/4 x c and b x c x hw >>>>>> b x hw/4 x hw
        attn = F.softmax(attn,dim=1)
        
        vx = F.max_pool2d(self.vconv(x),2).view(batch,self.channels_sc,-1) #b x c*4 x hw/4
        ##b x c*4 x hw/4 mul b x hw/4 x hw >>> b x 4c x hw
        attn = torch.bmm(vx,attn).view(batch,self.channels_sc,h,w) #b x c*4 x hw
        
        attn = self.attnconv(attn) # b,oc,h,w
        
        out = self.gamma * attn + x  # attn plus residual
        
        return out 

    

class ShopeeNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        
        self.pooling =  nn.AdaptiveAvgPool2d(1)
            
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x




class ShopeeNetV2(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='tf_efficientnet_b4',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=False):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNetV2, self).__init__()
        print('Model building for {} backbone'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.final_in_features = self.backbone.classifier.in_features
        
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.attn = Attn(self.final_in_features)
        self.conv1 = Dblock(self.final_in_features, self.final_in_features, downs= True)
        self.conv2 = Dblock(self.final_in_features, self.final_in_features, downs=True)
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.swish = Swish_module()
        # self.add_fc = spectral_norm(nn.Linear(self.final_in_features,self.final_in_features * 2,bias =False))
        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(self.final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(self.final_in_features , n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(self.final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(self.final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(self.final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
#         print(x.size())
        feature = self.extract_feat(x)
#         print(feature.size())
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        # print("image start : ", x.shape)
        
        x = self.backbone(x)

        # print("image feature after backnone : ",x.shape) 
    
        # print(self.final_in_features)
        x = self.attn(x)
        # print("After attn :  ", x.shape)
        x = self.conv1(x)
        # print("After DBlock :  ", x.shape)
        x = self.conv2(x)
        x = self.pooling(x).view(batch_size, -1)
        x = self.swish(x)
        # print("After pooling :  ", x.shape)
        # x = self.add_fc(x)
        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x


class Dblock(nn.Module):
    def __init__(self,in_channels,out_channels,downs = False,optimized = False):
        super(Dblock,self).__init__()
        self.downs = downs
        self.optimized = optimized
        self.learnable_sc = in_channels != out_channels or downs
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size = 3,padding = 1,bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels,out_channels,kernel_size = 3,padding = 1,bias=False))
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size = 1,bias=False))
            nn.init.orthogonal_(self.conv_sc.weight.data)
        self.relu = nn.ReLU()
        
        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        
    def _res(self,x):
        if not self.optimized:
            x = self.relu(x)
        x = self.conv2(self.relu(self.conv1(x)))
        if self.downs:
            x = F.avg_pool2d(x,2)
            
        return x
    
    def _shorcut(self,x):
        if self.learnable_sc:
            if self.optimized:
                x = self.conv_sc(F.avg_pool2d(x,2)) if self.downs else self.conv_sc(x) # reducing computation size first
            else:
                x = F.avg_pool2d(self.conv_sc(x),2) if self.downs else self.conv_sc(x)
        return x
    
    def forward(self,x):
        #print(self._shorcut(x).size(),self._res(x).size(),'DDDDDD')
        return self._shorcut(x) + self._res(x)



def load_model(model, model_file):
    state_dict = torch.load(model_file)
    if "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
#     del state_dict['metric_classify.weight']
    model.load_state_dict(state_dict, strict=True)
    print(f"loaded {model_file}")
    model.eval()    
    return model








if __name__ =='__main__':
    model_name = 'efficientnet_b3'
    images = torch.rand(1, 3, 256, 256)
    targets = torch.randint(10, (1,))
    model = ShopeeNet(**model_params)
    output = model(images,targets)
    criterion = fetch_loss()

    loss = criterion(output,targets)
    print(loss.item())
