# -*- coding: utf-8 -*-
"""
Created on Sat May 28 03:16:40 2022

@author: chrischris
"""
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s',device='cpu')

# Image
im = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(im)

results.pandas().xyxy[0]