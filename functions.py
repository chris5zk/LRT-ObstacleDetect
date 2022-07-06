# -*- coding: utf-8 -*-
"""
Created on Sat May 28 03:24:45 2022

@author: chrischris
"""
from importpackage import *

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def make_output_dir(output_base_path):
    try:
        os.makedirs(output_base_path)
    except FileExistsError:
        print('{} exist'.format(output_base_path))

    # increment output dir
    save_dir = os.path.join(output_base_path, 'exp')
    save_dir = increment_path(save_dir, exist_ok=False)

    # directory in every output
    output_path = save_dir
    os.makedirs(output_path)
    
    return output_path

def get_bounding_box(imgs, results):
    masks = []
    pre = results.pandas().xyxy
    for index,p in enumerate(pre):
        mask = np.zeros(imgs[index].shape,np.uint8)
        for i in range(len(p)):
            xmin = int(p.iloc[i]['xmin'])
            xmax = int(p.iloc[i]['xmax'])
            ymin = int(p.iloc[i]['ymin'])
            ymax = int(p.iloc[i]['ymax'])            
            mask[:,max(0,int(ymin-0.25*(ymax-ymin))):ymax,max(0,int(xmin-0.25*(xmax-xmin))):max(int(xmax+0.25*(xmax-xmin)),imgs[0].shape[0])] = 255;
        masks.append(mask)
    return masks

def get_segmentation(segs):
    masks = []
    for seg in segs:
        rail = np.zeros(seg.shape,np.uint8)
        mask_R = seg[:,:,0] == 244
        mask_G = seg[:,:,1] == 67
        mask_B = seg[:,:,2] == 244
        mask = mask_R * mask_G * mask_B
        rail[:,:,0] = mask
        rail[:,:,1] = mask
        rail[:,:,2] = mask
        masks.append(rail*255)
    return masks

def obstacle_detect(bbox_masks, seg_masks):
    rail_masks = []
    alerm_masks = []
    for i in range(len(bbox_masks)):
        rail_mask = np.zeros(seg_masks[i].shape,np.uint8)
        rail_mask[:,:,0] = 255 
        rail_mask[:,:,1] = 97
        rail_mask[:,:,2] = 0
        rail_mask = rail_mask * (seg_masks[i]/255).astype(np.uint8)
        intersection = bbox_masks[i] * seg_masks[i]
        if(intersection.any() == 1):
            alerm_mask = np.ones(seg_masks[i].shape,np.uint8)*255
            alerm_mask[:,:,1] = 0 
            alerm_mask[:,:,2] = 0
        else:
            alerm_mask = np.ones(seg_masks[i].shape,np.uint8)*255
            alerm_mask[:,:,0] = 0 
            alerm_mask[:,:,2] = 0
        rail_masks.append(rail_mask)
        alerm_masks.append(alerm_mask)
    return rail_masks,alerm_masks

def color_and_output(imgs, rail_masks, alerm_masks, results):
    outputs = []
    pre = results.pandas().xyxy
    for index,img in enumerate(imgs):
        output = cv2.addWeighted(img, 1, rail_masks[index], 0.5, 0)
        output = cv2.addWeighted(output, 1, alerm_masks[index], 0.2, 0)
        for i in range(len(pre[index])):
            xmin = int(pre[index].iloc[i]['xmin'])
            xmax = int(pre[index].iloc[i]['xmax'])
            ymin = int(pre[index].iloc[i]['ymin'])
            ymax = int(pre[index].iloc[i]['ymax'])
            output = cv2.rectangle(output,(xmin,ymax),(xmax,ymin),(255,0,0),2)
        outputs.append(output)
    return outputs

def store(outputs, path, target):
    
    if target == 'images':
        for index,output in enumerate(tqdm(outputs)):
            output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{path}/inference_{index}.jpg",output)
    
    if target == 'videos':
        frames = []
        for index,output in enumerate(tqdm(outputs)):
            #output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            frames.append(output)
        frames = torch.tensor(np.array(frames))
        torchvision.io.write_video(f'{path}/results.mp4', frames, 30)
                
                
                
                
                
                
                