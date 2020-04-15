# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import pickle as pkl

mode_list=['train', 'val', 'test']

word2idx={}
idx2word={}
word2idx['<end>']=0
word2idx['<start>']=1
idx2word[0]='<end>'
idx2word[1]='<start>'
n = 2
lens=[]

for mode in mode_list:
    file_name = 'annotations/'+mode+'.json'
    
    with open(file_name) as f:
        label = json.load(f)
    
    for item in label:
        idx = item['img_id']
        gt = item['sentences']
        for sentence in gt:
            words=sentence.split(' ')
            lens.append(len(words)+1)
            for word in words:
                if not word in word2idx.keys():
                    word2idx[word]=n
                    idx2word[n]=word
                    n+=1
print(lens)
print(max(lens))
                    
# output_name = 'word_list.pkl'
# with open(output_name, 'wb') as f:
#     pkl.dump(n, f)
#     pkl.dump(word2idx, f)
#     pkl.dump(idx2word, f)