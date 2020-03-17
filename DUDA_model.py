import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import pickle as pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class DualAttention(nn.Module):
    def __init__(self, attention_dim, feature_dim):
        super(DualAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim*2, attention_dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(attention_dim, 1, kernel_size=1, padding=0)
  
    def forward(self, X_bef, X_aft):#(b,c,h,w)
        batch_size = X_bef.shape[0]
    
        xdiff = torch.sub(X_aft, X_bef)#(b,c,h,w)

        X_bef_c = torch.cat([X_bef, xdiff], dim=1)#(b, 2c, h, w)
        X_aft_c = torch.cat([X_aft, xdiff], dim=1)#(b, 2c, h, w)

        X_bef_c = F.relu(self.conv1(X_bef_c))#(b, att_dim, h, w)
        X_aft_c = F.relu(self.conv1(X_aft_c))#(b, att_dim, h, w)

        a_bef = F.sigmoid(self.conv2(X_bef_c))#(b, 1, h, w)
        a_aft = F.sigmoid(self.conv2(X_aft_c))#(b, 1, h, w)

        l_bef = X_bef*(a_bef.repeat(1,1024,1,1))#(b, c, h, w)
        l_aft = X_aft*(a_aft.repeat(1,1024,1,1))#(b, c, h, w)

        l_bef = l_bef.sum(-2).sum(-1).view(batch_size, -1)#(b, c)
        l_aft = l_aft.sum(-2).sum(-1).view(batch_size, -1)#(b, c)

        return l_bef, l_aft, a_bef, a_aft
    
class DynamicSpeaker(nn.Module):

    def __init__(self, feature_dim, embed_dim, vocab_size, hidden_dim):
        super(DynamicSpeaker, self).__init__()

        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dynamic_att = nn.LSTMCell(hidden_dim*2, hidden_dim)

        self.decode_step = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)

        self.wd1 = nn.Linear(feature_dim*3, hidden_dim)
        self.wd2 = nn.Linear(hidden_dim, 3)
        
        self.wdc = nn.Linear(hidden_dim, vocab_size)
        self.h_da = None
        self.c_da = None

        self.h_ds = None
        self.c_ds = None

    def initialize(self, batch_size):
        self.h_da = torch.zeros(batch_size, self.hidden_dim).to(device)
        self.c_da = torch.zeros(batch_size, self.hidden_dim).to(device)

        self.h_ds = torch.zeros(batch_size, self.hidden_dim).to(device)
        self.c_ds = torch.zeros(batch_size, self.hidden_dim).to(device)

    def forward(self, l_bef, l_aft, prev_word):# (b, c)

        batch_size = l_bef.size(0)

        l_diff = torch.sub(l_aft,l_bef) #(b, c)

        v = torch.cat([l_bef,l_aft,l_diff], dim=1)#(b, 3c)
        v = F.relu(self.wd1(v)) # (b, hidden_dim)
        u_t = torch.cat([v, self.h_ds], dim=1)

        self.h_da, self.c_da = self.dynamic_att(u_t, (self.h_da, self.c_da))

        a_t = F.softmax(self.wd2(self.h_da)) 
        l_dyn = a_t[:,0].unsqueeze(1)*l_bef+ a_t[:,1].unsqueeze(1)*l_aft + a_t[:,2].unsqueeze(1)*l_diff

        # Embedding
        prev_x = self.embedding(prev_word)

        c_t = torch.cat([prev_x, l_dyn], dim =1)

        self.h_ds, self.c_ds = self.decode_step(c_t, (self.h_ds, self.c_ds))

        prediction = self.wdc(self.h_ds)

        return prediction, a_t