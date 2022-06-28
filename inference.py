# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:16:07 2022

@author: chrischris
"""

from datasets import *
from importpackage import *
from hyperparameter import *
from functions import *

# yolact-edged


# yolov5
# Load Model

device = torch.device('cpu')
model = torch.hub.load('ultralytics/yolov5','custom', pt, force_reload=True)
model.to(device)

print("------------------------- ↓ yolov5 ↓ -------------------------")
print(model)
print("------------------------- ↑ yolov5 ↑ -------------------------", flush=True)

# Load Data
# Images
if target == 'images':
    images_datasets = datasets.ImageFolder(images_org_path, transform=test_transforms)
    images_dataloaders = torch.utils.data.DataLoader(images_datasets, batch_size=batch_size, shuffle=False)

# Videos
if target == 'vidoes':
    pass

# Output
obs_mask_path,seg_mask_path,output_path = make_output_dir(output_base_path)

# Inference
model.eval()
if target == 'images':
    imgs = []
    for batch in tqdm(images_dataloaders):
        img, label = batch
        for i in range(batch_size):
            imgs.append(img[i].numpy()*255)
    results = model(imgs)
    
if target == 'videos':
    pass

# Results
results.print()
results.save()

# BBOX masks
bbox_masks = get_bounding_box(imgs, results)

# Rail masks
seg_masks = get_segmentation(images_seg_path)

# Obstacle detect


#if __name__ == '__main__':
#    main()