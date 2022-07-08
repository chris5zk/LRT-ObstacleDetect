# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:16:07 2022

@author: chrischris
"""

from datasets import *
from importpackage import *
from hyperparameter import *
from functions import *
#from my_yolact.eval import *

if __name__ == '__main__':
    
    ########## yolact-edged ##########
    #print("-------------------------  YOLACT_EDGE Start Inference  -------------------------")
    #my_yolact_edge(parse_arguments)
    #print("-------------------------  YOLACT_EDGE Finish Inference  -------------------------")    
    
    ########## yolov5 ##########
    ## Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("-------------------------  YOLOv5  -------------------------")
    print("-------------------------  Loading Model...  -------------------------")
    model = torch.hub.load('ultralytics/yolov5','custom', yolov5_pt, force_reload=True)
    model.to(device)
    model.eval()
    #print("------------------------- ↓ yolov5 ↓ -------------------------")
    #print(model)
    #print("------------------------- ↑ yolov5 ↑ -------------------------", flush=True)
    
    ########## Load Original Data ##########    
    # Images
    if target == 'images':
        print("-------------------------  Loading & Inference images -------------------------")
        org_dataset = datasets.ImageFolder(test_org_path, transform=test_transforms)
        org_dataloaders = torch.utils.data.DataLoader(org_dataset, batch_size=batch_size, shuffle=False)
        seg_dataset = datasets.ImageFolder(test_seg_path, transform=test_transforms)
        seg_dataloaders = torch.utils.data.DataLoader(seg_dataset, batch_size=batch_size, shuffle=False)
        seg_dataloader_iterator = iter(seg_dataloaders)
        outputs = []
        for org_batch in tqdm(org_dataloaders):  
            imgs = []
            segs = []
            data, label = org_batch
            for d in data:
                d = d.numpy()*255
                imgs.append(d.astype('uint8').transpose((1,2,0)))
            
            seg_batch = next(seg_dataloader_iterator)
            data, label = seg_batch
            for d in data:
                d = d.numpy()*255
                d = d.astype('uint8').transpose((1,2,0))
                d = rail_ROI(d)
                segs.append(d)
        ## Inference
            results = model(imgs)
        ## Results
            results.print()
        ## Obstacle detect
            print("Obstacle Detecting...")
            # BBOX masks
            bbox_masks = get_bounding_box(imgs, results)
            # Rail masks
            seg_masks = get_segmentation(segs)
            # Interaction
            rail_masks, alerm_masks = obstacle_detect(bbox_masks, seg_masks)
            outputs.append(color_and_output(imgs, rail_masks, alerm_masks, results))
        ## Output
        print(" Saving Results...")
        output_path = make_output_dir(output_base_path)
        store(outputs, output_path, target, test_org_images)

    if target == 'videos':
        outputs = []
        print("-------------------------  Loading & Inference videos  -------------------------")
        cap_org = cv2.VideoCapture(test_org_video)
        cap_seg = cv2.VideoCapture(test_seg_video)
        while True:
            frame_org = []
            frame_seg = []
            ret, org = cap_org.read()
            _, seg = cap_seg.read()
            frame_org.append(org)
            frame_seg.append(seg)
            if ret: 
                results = model(frame_org)
                # BBOX masks
                bbox_masks = get_bounding_box(frame_org, results)
                # Rail masks
                seg_masks = get_segmentation(frame_seg)
                # Interaction
                rail_masks, alerm_masks = obstacle_detect(bbox_masks, seg_masks)
                outputs.append(color_and_output(frame_org, rail_masks, alerm_masks, results))
            else:
                break        
        ## Output
        print(" Saving Results...")
        output_path = make_output_dir(output_base_path)
        store(outputs, output_path, target, test_org_path)