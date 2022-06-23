# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:18:33 2022

@author: chrischris
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd

# Model
model_rail = torch.hub.load('ultralytics/yolov5','custom','rail_1.pt')
model_object = torch.hub.load('ultralytics/yolov5','custom','object_4.pt')

# Image
dataset_path = 'dataset/batch1'
dataset = os.listdir(dataset_path)
imgs_for_rail = []
imgs_for_object = []
imgs_ori = []
for data in dataset:
    img = cv2.imread('{}/{}'.format(dataset_path,data))
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgs_ori.append(img_RGB)
    imgs_for_rail.append(img)
imgs_for_object = imgs_for_rail

# Inference
results_rail = model_rail(imgs_for_rail,size=640)
results_object = model_object(imgs_for_object,size=640)

# Results
results_rail.print()
results_object.print()

results_rail.save()
results_object.save()

# ROI
pre_rail = results_rail.pandas().xyxy
pre_object = results_object.pandas().xyxy
pre_all = []

for i in range(len(dataset)):
    pre_all.append(pd.concat([pre_rail[i],pre_object[i]],axis=0))

for index,pre in enumerate(pre_all):
    rails = pre[pre['name'] == 'rail']
    obstacles = pre[pre['name'] != 'rail']
    mask = np.zeros(imgs_for_object[index].shape,np.uint8)
    # check each rail bbox
    for i in range(len(rails)):
        ymin = int(rails.iloc[i]['ymin'])
        ymax = int(rails.iloc[i]['ymax'])
        h =  ymax - ymin
        xmin = int(rails.iloc[i]['xmin'])
        xmax = int(rails.iloc[i]['xmax'])
        w = xmax - xmin
        # check if each object is a obstacle or not
        for j in range(len(obstacles)):
            left = max(xmin-int(w/2),obstacles.iloc[j]['xmin'])
            right = min(xmax+int(w/2),obstacles.iloc[j]['xmax'])
            top = min(ymax+int(h/2),obstacles.iloc[j]['ymax'])
            bottom = max(ymin-int(h/2),obstacles.iloc[j]['ymin'])
            # interact or not
            if left >= right or top >= bottom:
                flag = 0
            else:
                flag = 1
        if len(obstacles) == 0:
            flag = 0
        mask[max(ymin-int(h/2),0):min(ymax+int(h/2),mask.shape[0]),max(xmin-int(w/2),0):min(xmax+int(w/2),mask.shape[1])] = 255
    # HoughLines detect
    ROI = imgs_ori[index] & mask
    ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    ROI_blur = cv2.GaussianBlur(ROI_gray, (5,5), 3)
    ROI_edge = cv2.Canny(ROI_blur,50,150)
    theta = np.pi/180
    lines = cv2.HoughLinesP(ROI_edge, 1, theta, 100, np.array([]), 50, 50)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                color = (255,0,0) if flag == 0 else (0,0,255)
                cv2.line(imgs_for_object[index],(x1,y1),(x2,y2),color,4)
    cv2.imwrite('runs/ROI/roi{}.jpg'.format(index), ROI)
    cv2.imwrite('runs/HoughLine/Lines{}.jpg'.format(index), imgs_for_object[index])

