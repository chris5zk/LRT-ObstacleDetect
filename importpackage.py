# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:43:43 2022

@author: chrischris
"""

import os
import cv2
import time
import torch
import urllib
import numpy as np
import torchvision
from torchvision import datasets, transforms
from pathlib import Path

# This is for the progress bar.
from tqdm import tqdm