# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:46:12 2022

@author: chrischris
"""

from importpackage import *

# Inference

# yolact-edge






# yolov5
# pt file
pt = 'object_4.pt'
# Input path:should be modified when merge with yolact
dataset_base_path = 'dataset' 
dataset_path = f"{dataset_base_path}/batch2/original"
seg_path = f"{dataset_base_path}/batch2/seg"
# Output path
output_base_path = 'runs/output'