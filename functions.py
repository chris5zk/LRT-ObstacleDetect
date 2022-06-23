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

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

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