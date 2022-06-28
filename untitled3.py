# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:19:30 2022

@author: chrischris
"""
import torch
import torchvision

device = 'cuda'
boxes = torch.tensor([[0., 1., 2., 3.]]).to(device)
scores = torch.randn(1).to(device)
iou_thresholds = 0.5

print(torchvision.ops.nms(boxes, scores, iou_thresholds))