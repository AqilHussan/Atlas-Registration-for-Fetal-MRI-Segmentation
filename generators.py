import torch
import os 
import time

import numpy as np
import ants

import glob
import json
import pandas as pd

from scipy import ndimage
from tqdm import tqdm

ants_dir = 'affine_reg/'

def read_ants(filename):
    try:
        ants_img = ants.image_read(filename)
    except:
        filename = filename.replace('irtk','mial')
        ants_img = ants.image_read(filename) 
    
    ants_seg = ants.image_read(filename.replace('T2w','dseg'))
    return tuple([ants_img,ants_seg])

def normalize(img):
    return (img- img.min())/(img.max() - img.min())

def ants_to_numpy(ants_img,norm=False):
    img_np = ants_img.numpy()
    if norm:
        img_np = normalize(img_np)
    img_np = img_np[50:114,:,:]
    return img_np[np.newaxis,...]
    
def apply_tfm(fixed,moving,tfm_path,interp):
    ants_warped = ants.apply_transforms(fixed= fixed,
                                 moving= moving,
                                 transformlist= tfm_path,
                                 interpolator= interp
                                )
    return ants_warped
    
def load_volfile(filename):

    ants_img, ants_seg = read_ants(filename)
    return tuple([ants_to_numpy(ants_img,True),ants_to_numpy(ants_seg)])

def numpy_to_torch(img_list,device):
    imgs_np = np.concatenate(img_list, axis=0)
    imgs_torch = torch.from_numpy(imgs_np).float()
    return imgs_torch.to(device)

def vxm_data_generator(start,vol_names, batch_size, reference, device):
    
    indices = np.arange(start,start+ batch_size,dtype=np.uint8) 
    img_segs = [load_volfile(vol_names[i]) for i in indices]
    
    scan1, seg1 = load_volfile(reference)
    volumes = {'scan1' :[[scan1]*batch_size],
              'seg1': [[seg1]*batch_size],
              'scan2': [],
              'seg2': []
              }
    
    for i,key in enumerate(['scan2','seg2']):
        volumes[key].append([img_segs[x][i] for x in range(len(img_segs))])

    scan1, seg1, scan2, seg2 = tuple([numpy_to_torch(volumes[key],device) for key in volumes.keys()])

    shape = scan1.shape[2:]
    zeros = torch.zeros((batch_size, len(shape), *shape)).to(device)

    invols = [scan1, scan2]
    outvols = [scan2, zeros] 
    seg_vols = [seg1, seg2]
        
    return (invols, outvols,seg_vols)