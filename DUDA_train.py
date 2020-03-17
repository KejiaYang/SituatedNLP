import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import pickle as pkl
from DUDA_model import DualAttention, DynamicSpeaker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SpotDataset(Dataset):
    def __init__(self, mode, data_folder="data/processed_data/"):
        self.mode = mode
        assert self.mode in {'train', 'val', 'test'}

        file_name = data_folder+mode+'.pkl'

        with open(file_name,"rb") as f:
            self.data=pkl.load(f)

        self.dataset_size = len(self.data)

    def __getitem__(self, i):
        img1 = torch.FloatTensor(self.data[i][0])
        img2 = torch.FloatTensor(self.data[i][1])
        caption = torch.LongTensor(self.data[i][2][0])
        if len(caption)>25:#TODO: maybe other padding
            caption = caption[:25]
        else:
            caption = torch.cat([caption, torch.zeros(25-len(caption)).long()], 0)
        capt_len = torch.LongTensor([min(len(self.data[i][2][0]), 25)])

        if self.mode == 'train':
            return img1, img2, caption, capt_len
        else:
            all_caption = self.data[i][2]
            return img1, img2, caption, capt_len, all_caption

    def __len__(self):
        return self.dataset_size

#word mapping
with open('data/word_list.pkl', 'rb') as f:
    tot_word = pkl.load(f)
    word2idx = pkl.load(f)
    idx2word = pkl.load(f)

# Model parameters
embed_dim = 300
attention_dim = 512
feature_dim = 1024

# Training parameters
epoch = 40
batch_size = 16
workers = 1
decoder_lr = 1e-4
encoder_lr = 1e-4
best_bleu4 = 0.
print_freq = 100

#pretrained res101
res101=models.resnet101(pretrained=True)
modules=list(res101.children())[:-3]
res101=nn.Sequential(*modules)
for p in res101.parameters():
    p.requires_grad = False
res101=res101.to(device)

dualattention = DualAttention(attention_dim, feature_dim).to(device)

dynamicspeaker = DynamicSpeaker(feature_dim = feature_dim,
                           embed_dim = embed_dim,
                           vocab_size = tot_word,
                           hidden_dim = attention_dim).to(device)

#training
encoder_optimizer = torch.optim.Adam(dualattention.parameters(), 
                                     lr=encoder_lr)

decoder_optimizer = torch.optim.Adam(dynamicspeaker.parameters(), 
                                     lr=decoder_lr)

criterion = nn.CrossEntropyLoss().to(device)

train_loader = torch.utils.data.DataLoader(
    SpotDataset('train'),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

for e in range(epoch):
    print("Epoch:", e)
    for i, (img1, img2, caption, capt_len) in enumerate(train_loader):
        cur_batch_size = img1.shape[0]
        img1 = img1.to(device)
        img2 = img2.to(device)
        caption = caption.to(device)
        capt_len = capt_len.to(device)
        img1 = res101(img1)
        img2 = res101(img2)
        l_bef, l_aft, _, _ = dualattention(img1, img2)
        
        loss=0.0
        dynamicspeaker.initialize(cur_batch_size)
        prev_word = torch.ones(cur_batch_size).long().to(device)
        for step in range(25):
            pred, a_t = dynamicspeaker(l_bef, l_aft, prev_word)
            prev_word = caption[:, step]
            loss += criterion(pred, prev_word)/25
        print(loss)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    if e%5==0:
        torch.save(dualattention.state_dict(), "saved_models/DualAttention_epoch"+str(e)+".model")
        torch.save(dynamicspeaker.state_dict(), "saved_models/DynamicSpeaker_epoch"+str(e)+".model")