import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import transformers
import geffnet
import math 

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

    
 
    
class Enet_Arcface_FINAL(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(Enet_Arcface_FINAL, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained('../input/bert-base-uncased')
        self.enet = geffnet.create_model(enet_type.replace('-', '_'), pretrained=None)
        self.feat = nn.Linear(self.enet.classifier.in_features+self.bert.config.hidden_size, 512)
        self.swish = Swish_module()
        self.dropout = nn.Dropout(0.5)
        self.metric_classify = ArcMarginProduct_subcenter(512, out_dim)
        self.enet.classifier = nn.Identity()
 
    def forward(self, x,input_ids, attention_mask):
        x = self.enet(x)
        text = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        x = torch.cat([x, text], 1)
        x = self.swish(self.feat(x))
        return F.normalize(x), self.metric_classify(x)
    


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