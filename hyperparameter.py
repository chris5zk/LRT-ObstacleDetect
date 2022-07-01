# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:46:12 2022

@author: chrischris
"""

from importpackage import *

########## Datasets ##########
dataset_base_path = './dataset'
target = 'images'     # images/videos

### train ###
train_dataset_path = f"{dataset_base_path}/train"

### test ###
test_dataset_path = f"{dataset_base_path}/test"
test_path = f"{test_dataset_path}/{target}"

# original dataset
test_org_path = f"{test_path}/original"

# segmentation dataset
test_seg_path = f"{test_path}/seg"          # should be modified when merge with yolact


########## Models ##########
### yolact-edge ###
# weights
yolact_edge_pt = './yolact_edge/weights/rail_2.pt'

# parse_args


### yolov5 ###
# weights
yolov5_pt = 'object_4.pt'

# data
batch_size = 4

########## Output ##########
output_base_path = 'runs/output'