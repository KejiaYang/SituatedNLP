import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import pickle as pkl
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('wordnet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class EmbeddingLayer(nn.Module):
    
    def __init__(self, embed_dim, vocab_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, w):
        w = self.embedding(w)
        return w
    
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

class SentenceDecoder(nn.Module):
    
    def __init__(self, feature_dim, embed_dim, vocab_size):
        super(SentenceDecoder, self).__init__()
        self.hidden_dim = feature_dim*2
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.encoder = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
    
    def init_hc(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c
    
    def forward(self, sentence, h, c):#(b, len, embd)
        h, c = self.encoder(sentence, (h, c))
        pred = self.fc(h)
        return pred, h, c #(batch_size, feat_dim*2)
    
class SentenceEncoder(nn.Module):
    
    def __init__(self, feature_dim, embed_dim, vocab_size):
        super(SentenceEncoder, self).__init__()
        self.hidden_dim = feature_dim*2
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        self.encoder = nn.LSTMCell(self.embed_dim, self.hidden_dim)
    
    def init_hc(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c
    
    def forward(self, sentence, h, c):#(b, len, embd)
        h, c = self.encoder(sentence, (h, c))
        return h, c #(batch_size, feat_dim*2)

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
        caption = torch.LongTensor(self.data[i][2][i%len(self.data[i][2])])
        caption = torch.cat([caption, torch.zeros(60-len(caption)).long()], 0)
        capt_len = torch.LongTensor([len(self.data[i][2][i%len(self.data[i][2])])])

        if self.mode == 'train':
            return img1, img2, caption, capt_len
        else:
            all_caption = self.data[i][2].to_list()
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
pretrain_DUDA_epoch = 100
pretrain_SE_epoch = 0
epoch = 51
batch_size = 16
workers = 1
decoder_lr = 1e-4
encoder_lr = 1e-4
best_bleu4 = 0.
print_freq = 100
l1_lambda = 3e-3
img_lambda = 1.0
stc_lambda = 1.0

#pretrained res101
res101=models.resnet101(pretrained=True)
modules=list(res101.children())[:-3]
res101=nn.Sequential(*modules)
for p in res101.parameters():
    p.requires_grad = False
res101=res101.to(device)

wordembedding = EmbeddingLayer(embed_dim, tot_word).to(device)

dualattention = DualAttention(attention_dim, feature_dim).to(device)

sentencedecoder = SentenceDecoder(feature_dim = feature_dim,
                                embed_dim = embed_dim,
                                vocab_size = tot_word).to(device)

sentenceencoder = SentenceEncoder(feature_dim = feature_dim,
                                embed_dim = embed_dim,
                                vocab_size = tot_word).to(device)

#training
att_optimizer = torch.optim.Adam(dualattention.parameters(), 
                                     lr=1e-4)

decoder_optimizer = torch.optim.Adam(sentencedecoder.parameters(), 
                                     lr=1e-4)

encoder_optimizer = torch.optim.Adam(sentenceencoder.parameters(), 
                                     lr=1e-4)

embedding_optimizer = torch.optim.Adam(wordembedding.parameters(), 
                                     lr=1e-4)

criterion = nn.CrossEntropyLoss().to(device)
MSEloss = nn.MSELoss().to(device)

train_loader = torch.utils.data.DataLoader(
    SpotDataset('train'),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

for e in range(pretrain_DUDA_epoch):
    print("Pretrain DUDA Epoch:", e)
    for i, (img1, img2, caption, capt_len) in enumerate(train_loader):
        cur_batch_size = img1.shape[0]
        img1 = img1.to(device)
        img2 = img2.to(device)
        caption = caption.to(device)
        capt_len = capt_len.to(device)
        capt_len = capt_len.squeeze(1)
        img1 = res101(img1)
        img2 = res101(img2)
        l_bef, l_aft, a_bef, a_aft = dualattention(img1, img2)
        
        loss= l1_lambda*(torch.sum(torch.abs(a_bef))+torch.sum(torch.abs(a_aft)))
        l = torch.cat([l_bef,l_aft], dim=1)
        
        decode_h, decode_c = sentencedecoder.init_hc(cur_batch_size, device)
        decode_c = l
        prev_word = torch.ones(cur_batch_size).long().to(device)
        for step in range(60):
            cur_active = (capt_len>step).nonzero().squeeze(1)
            if cur_active.shape[0]==0:
                break
            pred, decode_h, decode_c = sentencedecoder(wordembedding(prev_word), decode_h, decode_c)
            prev_word = caption[:, step]
            loss += criterion(pred[cur_active], prev_word[cur_active])
        print(loss)
        embedding_optimizer.zero_grad()
        att_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        att_optimizer.step()
        decoder_optimizer.step()
        embedding_optimizer.step()

torch.save(dualattention.state_dict(), "saved_models/DualAttention_gd_epoch-1.model")
torch.save(sentencedecoder.state_dict(), "saved_models/SentenceDecoder_gd_epoch-1.model")
torch.save(wordembedding.state_dict(), "saved_models/WordEmbedding_gd_epoch-1.model")

for e in range(pretrain_SE_epoch):
    print("Pretrain SE Epoch:", e)
    for i, (img1, img2, caption, capt_len) in enumerate(train_loader):
        cur_batch_size = img1.shape[0]
        img1 = img1.to(device)
        img2 = img2.to(device)
        caption = caption.to(device)
        capt_len = capt_len.to(device)
        capt_len = capt_len.squeeze(1)
        img1 = res101(img1)
        img2 = res101(img2)
        l_bef, l_aft, a_bef, a_aft = dualattention(img1, img2)
        l = torch.cat([l_bef,l_aft], dim=1)
        h, c = sentenceencoder.init_hc(cur_batch_size, device)
        for step in range(60):
            cur_active = (capt_len>step).nonzero().squeeze(1)
            if cur_active.shape[0]==0:
                break
            h[cur_active], c[cur_active] = sentenceencoder(wordembedding(caption[cur_active,step]), h[cur_active], c[cur_active])
#             print(pred[cur_active], prev_word[cur_active])
        loss = MSEloss(c, l)
        print(loss)
        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        
for e in range(epoch):
    print("Co-train Epoch:", e)
    for i, (img1, img2, caption, capt_len) in enumerate(train_loader):
        cur_batch_size = img1.shape[0]
        img1 = img1.to(device)
        img2 = img2.to(device)
        caption = caption.to(device)
        capt_len = capt_len.to(device)
        capt_len = capt_len.squeeze(1)
        img1 = res101(img1)
        img2 = res101(img2)
        l_bef, l_aft, a_bef, a_aft = dualattention(img1, img2)
        
        l = torch.cat([l_bef,l_aft], dim=1)
        
        loss= l1_lambda*(torch.sum(torch.abs(a_bef))+torch.sum(torch.abs(a_aft)))
        print(loss)
        
        #image reconstruction
        decode_h, decode_c = sentencedecoder.init_hc(cur_batch_size, device)
        encode_h, encode_c = sentenceencoder.init_hc(cur_batch_size, device)
        decode_c = l
        decoded_seq = torch.zeros(cur_batch_size, 60).long().to(device)
        dec_capt_len = torch.zeros(cur_batch_size).long().to(device)
        cur_active = torch.arange(cur_batch_size).long().to(device)
        prev_word = torch.ones(cur_batch_size).long().to(device)
        for step in range(60):
            dec_capt_len[cur_active] += 1
            pred, decode_h, decode_c = sentencedecoder(wordembedding(prev_word), decode_h, decode_c)
            prev_word = torch.argmax(pred, dim=1)
            decoded_seq[cur_active, step] = prev_word[cur_active]
            cur_active = cur_active[prev_word[cur_active].nonzero().squeeze(1)]
            if cur_active.shape[0]==0:
                break
            loss += criterion(pred[cur_active], caption[cur_active, step])
        for step in range(60):
            cur_active = (dec_capt_len>step).nonzero().squeeze(1)
            if cur_active.shape[0]==0:
                break
            encode_h[cur_active], encode_c[cur_active] = sentenceencoder(wordembedding(decoded_seq[cur_active,step]), 
                                                                         encode_h[cur_active], encode_c[cur_active])
        loss += img_lambda * MSEloss(encode_c, l)

        print(loss)
        decode_h, decode_c = sentencedecoder.init_hc(cur_batch_size, device)
        encode_h, encode_c = sentenceencoder.init_hc(cur_batch_size, device)
        #sentence reconstruction
        for step in range(60):
            cur_active = (capt_len>step).nonzero().squeeze(1)
            if cur_active.shape[0]==0:
                break
            encode_h[cur_active], encode_c[cur_active] = sentenceencoder(wordembedding(caption[cur_active,step]), 
                                                                         encode_h[cur_active], encode_c[cur_active])

        prev_word = torch.ones(cur_batch_size).long().to(device)
        decode_c = encode_c
        for step in range(60):
            cur_active = (capt_len>step).nonzero().squeeze(1)
            if cur_active.shape[0]==0:
                break
            pred, decode_h, decode_c = sentencedecoder(wordembedding(prev_word), decode_h, decode_c)
            prev_word = caption[:, step]
            loss += criterion(pred[cur_active], prev_word[cur_active])
        
        print(loss)
        embedding_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        att_optimizer.zero_grad()
        loss.backward()
        att_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()
        embedding_optimizer.step()
    if e%5==0:
        torch.save(dualattention.state_dict(), "saved_models/DualAttention_gd_epoch"+str(e)+".model")
        torch.save(sentencedecoder.state_dict(), "saved_models/SentenceDecoder_gd_epoch"+str(e)+".model")
        torch.save(wordembedding.state_dict(), "saved_models/WordEmbedding_gd_epoch"+str(e)+".model")
        torch.save(sentenceencoder.state_dict(), "saved_models/SentenceEncoder_gd_epoch"+str(e)+".model")