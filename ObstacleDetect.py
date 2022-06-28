# -*- coding: utf-8 -*-
"""
Created on Tue May 24 00:15:36 2022

@author: chrischris
"""

from dataset import *
from importpackage import *
from hyperparameter import *
from functions import *

# yolact-edge



# yolov5

# Model
device = "cpu"
model = torch.hub.load('ultralytics/yolov5','custom', pt, device=device, force_reload=True)

print("------------------------- ↓ yolov5 ↓ -------------------------")
print(model)
print("------------------------- ↑ yolov5 ↑ -------------------------", flush=True)

# Output path
obs_mask_path,seg_mask_path,output_path = make_output_dir(output_base_path)

# DataLoader
dataset = os.listdir(dataset_path)
images = []
videos = []
for data in dataset:
    if data.endswith(".mp4" or ".webm" or ".avi" or ".mpg"):
        videos.append(data)
    if data.endswith(".jpg" or ".png" or ".tif" or ".bmp"):
        images.append(data)
        

"""
imgs = []
imgs_BGR = []
for data in dataset:
    img = cv2.imread(f"{dataset_path}/{data}")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgs.append(img_RGB)
    imgs_BGR.append(img)
"""
i = 0
dataset = os.listdir(dataset_path)
imgs = []
imgs_BGR = []
for data in dataset:
    cap = cv2.VideoCapture(f"{dataset_path}/{data}")
    while True:
        ret, frame = cap.read()
        if ret:
            frame_RGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            imgs.append(frame_RGB)
            imgs_BGR.append(frame)
        if i == 300:
            break;
        i += 1
    #img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #imgs.append(img_RGB)
    #imgs_BGR.append(img)
cap.release()

# Inference
results = model(imgs,size=640)

# Results
results.print()
results.save()

# Obstacle BBOX mask
obs_masks = []
pre = results.pandas().xyxy
for index,p in enumerate(pre):
    mask = np.zeros(imgs[index].shape,np.uint8)
    for i in range(len(p)):
        xmin = int(p.iloc[i]['xmin'])
        xmax = int(p.iloc[i]['xmax'])
        ymin = int(p.iloc[i]['ymin'])
        ymax = int(p.iloc[i]['ymax'])
        mask[ymin:ymax,xmin:xmax] = 255;
    obs_masks.append(mask)
    #cv2.imwrite(f"{obs_mask_path}/{dataset[index]}_obs_mask.png",mask)
        
# Rail mask
i = 0
seg_masks = []
segs = os.listdir(seg_path)
for index,seg in enumerate(segs):
    cap2 = cv2.VideoCapture(f"{seg_path}/{seg}")
    while True:
        ret,rail = cap2.read()
        if ret:
    #rail = cv2.imread(f"{seg_path}/{seg}")
            mask_B = rail[:,:,0] == 33
            mask_G = rail[:,:,1] == 150
            mask_R = rail[:,:,2] == 243
            mask = mask_B.astype(np.uint8) * mask_G.astype(np.uint8) * mask_R.astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask[:,:,0] = 0
            mask[:,:,2] = 0
            seg_masks.append(mask)
        if i == 300:
            break
        i+=1
    #mask = ndimage.binary_opening(mask, structure=np.ones((5,5))).astype(np.uint8)
    #cv2.imwrite(f"{seg_mask_path}/{dataset[index]}_seg_mask.png",mask*255)
cap2.release()

# Obstacle detect
obs_dataset = os.listdir(obs_mask_path)
seg_dataset = os.listdir(seg_mask_path)
#obs_masks = []
#eg_masks = []
color_masks = []
outputs = []
"""
for data in obs_dataset:
    obs_mask = cv2.imread(f"{obs_mask_path}/{data}")
    obs_masks.append(obs_mask)
    
for data in seg_dataset:
    seg_mask = cv2.imread(f"{seg_mask_path}/{data}")
    seg_masks.append(seg_mask)
"""
for i in range(len(imgs)):
    obs_mask = obs_masks[i]
    seg_mask = seg_masks[i]
    intersection = obs_mask * seg_mask
    if(intersection.any() == 1):
        color_mask = seg_mask
        color_mask[:,:,0] = 0 
        color_mask[:,:,1] = 0
        color_mask[:,:,2] = 255
    else:
        color_mask = seg_mask
        color_mask[:,:,0] = 0
        color_mask[:,:,1] = 255
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
    outputs.append(output)   
    #cv2.imwrite(f"{output_path}/{dataset[index]}_output.jpg",output)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"{output_path}/output.avi", fourcc, 30, (1280,720))

for frame_pre in outputs:    
    out.write(frame_pre)
    
out.release()
