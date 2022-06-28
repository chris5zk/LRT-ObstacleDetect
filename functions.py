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

        # Method 1
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
    obs_mask_path = os.path.join(save_dir,'obs_mask')
    seg_mask_path = os.path.join(save_dir,'seg_mask')
    output_path = os.path.join(save_dir,'result')
    
    os.makedirs(obs_mask_path)  
    os.makedirs(seg_mask_path)
    os.makedirs(output_path)
    
    return obs_mask_path,seg_mask_path,output_path

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
            mask[ymin:ymax,xmin:xmax] = 255;
        masks.append(mask)
    return masks

def get_segmentation(path):
    masks = []
    segs = os.listdir(path)
    for index,seg in enumerate(segs):
        rail = cv2.imread(f"{path}/{seg}")
        mask_B = rail[:,:,0] == 33
        mask_G = rail[:,:,1] == 150
        mask_R = rail[:,:,2] == 243
        mask = mask_B.astype(np.uint8) * mask_G.astype(np.uint8) * mask_R.astype(np.uint8)
        masks.append(mask*255)
    return masks