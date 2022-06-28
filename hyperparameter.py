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
dataset_base_path = './dataset'

# train
train_dataset_path = f"{dataset_base_path}/train"

# test
target = 'images'
test_dataset_path = f"{dataset_base_path}/test"

test_images_path = f"{test_dataset_path}/images"
images_org_path = f"{test_images_path}/original"
images_org_data = f"{images_org_path}/rail"
images_seg_path = f"{test_images_path}/seg"
images_org_data = f"{images_seg_path}/rail"

test_videos_path = f"{test_dataset_path}/videos"
videos_org_path = f"{test_videos_path}/original"
videos_seg_path = f"{test_videos_path}/seg"

# Output path
output_base_path = 'runs/output'

# Data
batch_size = 4
