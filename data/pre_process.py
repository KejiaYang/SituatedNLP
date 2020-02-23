import json
import cv2
import pickle as pkl

mode_list=['train', 'val', 'test']

mode = 'test'
file_name = 'annotations/'+mode+'.json'

with open(file_name) as f:
    label = json.load(f)

pkl_data=[]
for item in label[:1]:
    idx = item['img_id']
    gt = item['sentences']
    img0_name = 'image/'+int(idx)+'.png'
    img1_name = 'image/'+int(idx)+'_2.png'
    img0 = cv2.imread(img0_name)
    img1 = cv2.imread(img1_name)
    pkl_data.append([img0, img1, gt])

output_name = 'data/'+mode+'.pkl'
with open(output_name, 'wb') as f:
    f.dump(pkl_data)
