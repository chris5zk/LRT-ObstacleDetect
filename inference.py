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
    print("-------------------------  YOLACT_EDGE Start Inference  -------------------------")
    #my_yolact_edge(parse_arguments)
    print("-------------------------  YOLACT_EDGE Finish Inference  -------------------------")    

    ########## Load Original Data ##########
    imgs = []
    segs = []
    
    # Images
    if target == 'images':
        print("-------------------------  Loading original images  -------------------------")
        dataset = datasets.ImageFolder(test_org_path, transform=test_transforms)
        dataloaders = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in tqdm(dataloaders):
            data, label = batch
            for i in range(len(data)):
                np = data[i].numpy()*255
                np = np.astype('uint8').transpose((1,2,0))
                imgs.append(np)
                
        print("-------------------------  Loading segmentation images  -------------------------")
        dataset = datasets.ImageFolder(test_seg_path, transform=test_transforms)
        dataloaders = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in tqdm(dataloaders):
            data, label = batch
            for i in range(len(data)):
                np = data[i].numpy()*255
                np = np.astype('uint8').transpose((1,2,0))
                np[0:int(np.shape[0]/3),:] = 0
                np[:,0:int(np.shape[1]/3)] = 0 
                np[:,int(2*np.shape[1]/3):] = 0
                segs.append(np)

    # Videos
    if target == 'videos':
        print("-------------------------  Loading original videos  -------------------------")
        frame = torchvision.io.read_video(test_org_video, start, end, mode)
        for data in tqdm(frame[0]):
            imgs.append(data.numpy())

        print("-------------------------  Loading segmentation videos  -------------------------")
        frame = torchvision.io.read_video(test_seg_video, start, end, mode)
        for data in tqdm(frame[0]):
            segs.append(data.numpy())

    ########## yolov5 ##########
    ## Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("-------------------------  YOLOv5  -------------------------")
    print("-------------------------  Loading Model...  -------------------------")
    model = torch.hub.load('ultralytics/yolov5','custom', yolov5_pt, force_reload=True)
    model.to(device)
    
    #print("------------------------- ↓ yolov5 ↓ -------------------------")
    #print(model)
    #print("------------------------- ↑ yolov5 ↑ -------------------------", flush=True)
    
    ## Inference
    model.eval()
    print("-------------------------  YOLOv5 Start Inference  -------------------------")  
    results = model(imgs)
    print("-------------------------  YOLOv5 Finish Inference  -------------------------")
    
    ## Results
    results.print()
    results.save() if target == 'images' else None
    
    ## Obstacle detect
    print("-------------------------  Obstacle Detecting...  -------------------------")
    # BBOX masks
    bbox_masks = get_bounding_box(imgs, results)
    # Rail masks
    seg_masks = get_segmentation(segs)
    # Interaction
    rail_masks, alerm_masks = obstacle_detect(bbox_masks, seg_masks)
    outputs = color_and_output(imgs, rail_masks, alerm_masks, results)
    
    ## Output
    print("-------------------------  Saving Obstacle Detection Results  -------------------------")
    output_path = make_output_dir(output_base_path)
    store(outputs, output_path, target)
    print("\n-------------------------  Finish -------------------------")
    