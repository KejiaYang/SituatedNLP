import json
import cv2
import pickle as pkl
import torchvision.models as models
import torch
import torch.nn as nn

mode_list=['train', 'val', 'test']

with open('word_list.pkl', 'rb') as f:
    n = pkl.load(f)
    word2idx = pkl.load(f)
    idx2word = pkl.load(f)

for mode in mode_list:
    print(mode)
    file_name = 'annotations/'+mode+'.json'

    with open(file_name) as f:
        label = json.load(f)

    tot = len(label)
    i=0
    batch_size=10
    pkl_data=[]
    for item in label:
        idx = item['img_id']
        gt = item['sentences']
        img0_name = 'images/'+str(idx)+'.png'
        img1_name = 'images/'+str(idx)+'_2.png'
        img0 = cv2.imread(img0_name).transpose(2,0,1)
        img1 = cv2.imread(img1_name).transpose(2,0,1)
        one_hot_gt = []
        for sentence in gt:
            sentence = sentence.split(' ')
            sentence = list(map(lambda x:word2idx[x], sentence))
            sentence.append(0)
            one_hot_gt.append(sentence)
        pkl_data.append([img0, img1, one_hot_gt])
        i+=1
        if i%10==0:
            print('Finished: ', i/tot)

    output_name = 'processed_data/'+mode+'.pkl'
    with open(output_name, 'wb') as f:
        pkl.dump(pkl_data, f)
