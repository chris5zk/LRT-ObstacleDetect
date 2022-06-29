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
print("-------------------------  Loading Model...  -------------------------")
model = torch.hub.load('ultralytics/yolov5','custom', pt, force_reload=True)
model.to(device)

print("------------------------- ↓ yolov5 ↓ -------------------------")
print(model)
print("------------------------- ↑ yolov5 ↑ -------------------------", flush=True)

# Load Data
# Images
if target == 'images':
    images_org_datasets = datasets.ImageFolder(images_org_path, transform=test_transforms)
    images_org_dataloaders = torch.utils.data.DataLoader(images_org_datasets, batch_size=batch_size, shuffle=False)
    images_seg_datasets = datasets.ImageFolder(images_seg_path, transform=test_transforms)
    images_seg_dataloaders = torch.utils.data.DataLoader(images_seg_datasets, batch_size=batch_size, shuffle=False)
# Videos
if target == 'vidoes':
    pass

# Inference
model.eval()
imgs = []
segs = []
if target == 'images':
    # original
    print("-------------------------  Loading original images  -------------------------")
    for batch in tqdm(images_org_dataloaders):
        img, label = batch
        for i in range(len(img)):
            imgs.append(img[i].numpy()*255)
    # segmentation
    print("-------------------------  Loading segmentation images  -------------------------")
    for batch in tqdm(images_seg_dataloaders):
        seg, label = batch
        for i in range(len(seg)):
            segs.append(seg[i].numpy()*255)
    print("-------------------------  Start Inference  -------------------------")
    results = model(imgs)
    
if target == 'videos':
    pass

# Results
results.print()
results.save()
print("-------------------------  Finish Inference  -------------------------")

# Obstacle detect
print("-------------------------  Obstacle Detecting...  -------------------------")
# BBOX masks
bbox_masks = get_bounding_box(imgs, results)
# Rail masks
seg_masks = get_segmentation(images_seg_data, segs)
# Interaction
rail_masks, alerm_masks = obstacle_detect(bbox_masks, seg_masks)
outputs = color_and_output(imgs, rail_masks, alerm_masks, results)

# Output
print("-------------------------  Saving Obstacle Detection Results  -------------------------")
output_path = make_output_dir(output_base_path)
store(outputs, output_path)

#if __name__ == '__main__':
#    main()