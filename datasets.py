# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:03:37 2022

@author: chrischris
"""

from importpackage import *

test_transforms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])