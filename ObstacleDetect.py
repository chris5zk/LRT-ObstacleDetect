# -*- coding: utf-8 -*-
"""
Created on Tue May 24 00:15:36 2022

@author: chrischris
"""

from importpackage import *
from hyperparameter import *
from functions import *

# yolact-edge



# yolov5

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5','custom', pt, device=device)

print("------------------------- ↓ yolov5 ↓ -------------------------")
print(model)
print("------------------------- ↑ yolov5 ↑ -------------------------", flush=True)

# Output path
obs_mask_path,seg_mask_path,output_path = make_output_dir(output_base_path)

# Image
dataset = os.listdir(dataset_path)
imgs = []
imgs_BGR = []
for data in dataset:
    img = cv2.imread(f"{dataset_path}/{data}")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgs.append(img_RGB)
    imgs_BGR.append(img)
    
# Inference
results = model(imgs,size=640)

# Results
results.print()
results.save()

# Obstacle BBOX mask
pre = results.pandas().xyxy
for index,p in enumerate(pre):
    mask = np.zeros(imgs[index].shape,np.uint8)
    for i in range(len(p)):
        xmin = int(p.iloc[i]['xmin'])
        xmax = int(p.iloc[i]['xmax'])
        ymin = int(p.iloc[i]['ymin'])
        ymax = int(p.iloc[i]['ymax'])
        mask[ymin:ymax,xmin:xmax] = 255;
    cv2.imwrite(f"{obs_mask_path}/{dataset[index]}_obs_mask.png",mask)
        
# Rail mask
segs = os.listdir(seg_path)
for index,seg in enumerate(segs):
    rail = cv2.imread(f"{seg_path}/{seg}")
    mask_B = rail[:,:,0] == 33
    mask_G = rail[:,:,1] == 150
    mask_R = rail[:,:,2] == 243
    mask = mask_B.astype(np.uint8) * mask_G.astype(np.uint8) * mask_R.astype(np.uint8)
    #mask = ndimage.binary_opening(mask, structure=np.ones((5,5))).astype(np.uint8)
    cv2.imwrite(f"{seg_mask_path}/{dataset[index]}_seg_mask.png",mask*255)

# Obstacle detect
obs_dataset = os.listdir(obs_mask_path)
seg_dataset = os.listdir(seg_mask_path)
obs_masks = []
seg_masks = []
color_masks = []
for data in obs_dataset:
    obs_mask = cv2.imread(f"{obs_mask_path}/{data}")
    obs_masks.append(obs_mask)
    
for data in seg_dataset:
    seg_mask = cv2.imread(f"{seg_mask_path}/{data}")
    seg_masks.append(seg_mask)

for i in range(len(obs_dataset)):
    obs_mask = obs_masks[i]
    seg_mask = seg_masks[i]
    intersection = obs_mask * seg_mask
    if(intersection.any() == 1):
        color_mask = seg_mask
        color_mask[:,:,0] = 0 
        color_mask[:,:,1] = 0
    else:
        color_mask = seg_mask
        color_mask[:,:,0] = 0
        color_mask[:,:,2] = 0
    color_masks.append(color_mask)

for index,img in enumerate(imgs_BGR):
    output = cv2.addWeighted(imgs_BGR[index], 1, color_masks[index], 0.3, 0)
    for i in range(len(pre[index])):
        xmin = int(pre[index].iloc[i]['xmin'])
        xmax = int(pre[index].iloc[i]['xmax'])
        ymin = int(pre[index].iloc[i]['ymin'])
        ymax = int(pre[index].iloc[i]['ymax'])
        output = cv2.rectangle(output,(xmin,ymax),(xmax,ymin),(255,0,0),2)
    cv2.imwrite(f"{output_path}/{dataset[index]}_output.jpg",output)
  
    
