
# coding: utf-8

# In[1]:


import pickle
import re

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import verbnet


# In[3]:


pickle_in = open("visual_feature_reg.pkl", "rb")
data = pickle.load(pickle_in)
# print(data[0][3])


# In[8]:


verbNet = []
for unit in data:
    sentence = []
    for cap in unit[3]:
#         print(unit[3])
        tagged = nltk.pos_tag(cap)
        verb_lists = []
        for idx, tp in enumerate(tagged):
            if (tp[1][:2] == 'VB'):
                base_form = WordNetLemmatizer().lemmatize(cap[idx], 'v')
                if (verbnet.classids(base_form) != []):
                    attr = verbnet.classids(base_form)
                    at_list = []
                    for at in attr:
                        splitted_at = []
                        splitted_string = at.split('-')
                        splitted_at.append(splitted_string[0])
                        splitted_at.append(splitted_string[1].split('.')[0])
                        at_list.append([])
                        at_list[-1] = splitted_at
                    verb_lists.append([base_form, at_list, len(attr)])
        sentence.append(verb_lists)
#         print(sentence[-1])
    verbNet.append(sentence)

# print(verbNet)


# In[9]:


output = open('verbNet_visual.pkl', "wb")
pickle.dump(verbNet, output)
output.close()

# pickle_in = open('verbNet_visual.pkl', "rb")
# data = pickle.load(pickle_in)
# print(data)

