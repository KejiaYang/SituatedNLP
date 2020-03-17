import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
import pickle as pkl
from DUDA_model import DualAttention, DynamicSpeaker
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('wordnet')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

def validation(val_data, dualattention, dynamicspeaker, idx2word, res101):
    print("Validating")      
    weights1 = (1.0, 0.0, 0.0, 0.0)
    weights2 = (0.5, 0.5, 0.0, 0.0)
    weights3 = (0.33, 0.33, 0.33, 0.0)
    weights4 = (0.25, 0.25, 0.25, 0.25)
    avg_meteor = 0.0
    avg_bleu1 = 0.0
    avg_bleu2 = 0.0
    avg_bleu3 = 0.0
    avg_bleu4 = 0.0
    chencherry = SmoothingFunction()
    with torch.no_grad():
        for i, (img1, img2, all_capt) in enumerate(val_data):
            for j in range(len(all_capt)):
                all_capt[j] = list(map(lambda x:idx2word[x] if x>1 else '', all_capt[j]))
                all_capt[j] = [w for w in all_capt[j] if w!='']
            #print(all_capt)
            img1 = torch.FloatTensor(img1).unsqueeze(0)
            img2 = torch.FloatTensor(img2).unsqueeze(0)
            img1 = img1.to(device)
            img2 = img2.to(device)
            img1 = res101(img1)
            img2 = res101(img2)
            l_bef, l_aft, _, _ = dualattention(img1, img2)
        
            dynamicspeaker.initialize(1)
            prev_word = torch.ones(1).long().to(device)
            decoded_seq=torch.ones(1, 1).long().to(device)
            for step in range(25):
                pred, a_t = dynamicspeaker(l_bef, l_aft, prev_word)
                prev_word = torch.argmax(pred, dim=1)
                decoded_seq = torch.cat([decoded_seq, prev_word.view(-1, 1)], dim=1)
            decoded_seq = list(map(lambda x:idx2word[x] if x>1 else '', decoded_seq.tolist()[0]))
            decoded_seq = [w for w in decoded_seq if w!='']
            avg_bleu1 += sentence_bleu(all_capt, decoded_seq, weights1, smoothing_function=chencherry.method1)
            avg_bleu2 += sentence_bleu(all_capt, decoded_seq, weights2, smoothing_function=chencherry.method1)
            avg_bleu3 += sentence_bleu(all_capt, decoded_seq, weights3, smoothing_function=chencherry.method1)
            avg_bleu4 += sentence_bleu(all_capt, decoded_seq, weights4, smoothing_function=chencherry.method1)
            avg_meteor += meteor_score([' '.join(s) for s in all_capt], ' '.join(decoded_seq))
            
    avg_meteor /= len(val_data)
    avg_bleu1 /= len(val_data)
    avg_bleu2 /= len(val_data)
    avg_bleu3 /= len(val_data)
    avg_bleu4 /= len(val_data)
    return avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_meteor

dualattention = DualAttention(attention_dim, feature_dim).to(device)

dynamicspeaker = DynamicSpeaker(feature_dim = feature_dim,
                           embed_dim = embed_dim,
                           vocab_size = tot_word,
                           hidden_dim = attention_dim).to(device)

dualattention.load_state_dict(torch.load("saved_models/DualAttention_epoch35.model"))
dynamicspeaker.load_state_dict(torch.load("saved_models/DynamicSpeaker_epoch35.model"))

val_data_file = "data/processed_data/test.pkl"
with open(val_data_file,"rb") as f:
    val_data=pkl.load(f)

avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4, avg_meteor = validation(val_data, dualattention, dynamicspeaker, idx2word, res101)
print('avg_bleu1: ', avg_bleu1)
print('avg_bleu2: ', avg_bleu2)
print('avg_bleu3: ', avg_bleu3)
print('avg_bleu4: ', avg_bleu4)
print('avg_meteor: ', avg_meteor)

    