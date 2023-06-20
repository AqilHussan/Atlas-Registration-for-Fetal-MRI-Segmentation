import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import ants
import torch
from torchvision import transforms
from torch.utils.data import Dataset 
from ipywidgets import interact, fixed
from IPython.display import clear_output
from matplotlib import pyplot as plt
import torch.nn.functional as F
from skimage.filters import gaussian
import pandas as pd

ants_dir = '../voxel_morph/affine_reg/'

def get_filenames(institute,voltype,rectype):
    
    df = pd.read_table(f'../{institute}/participants.tsv')
    normal_df = df[df['Pathology']== voltype]
    valid_indices = [int(idx.split('-')[-1]) for idx in normal_df['participant_id']]
    filenames = [ f'../{institute}/sub-{idx:03}/anat/sub-{idx:03}_rec-{rectype}_T2w.nii.gz' for idx in valid_indices]
    return filenames

def visualise_volumes(batch_id,x, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[batch_id,x,:,:] + alpha*moving[batch_id,x,:,:] 
    plt.imshow((img).astype(np.uint8));
    plt.axis('off')
    plt.show()

def tensor_to_onehot(x):
    y = F.one_hot(x.type(torch.LongTensor))
    y = y.type(torch.FloatTensor)
#     y = y.view(new_shape)
    return y

def read_ants(filename):
    try:
        ants_img = ants.image_read(filename)
    except:
        filename = filename.replace('irtk','mial')
        ants_img = ants.image_read(filename) 
    
    ants_seg = ants.image_read(filename.replace('T2w','dseg'))
    ants_img = ants_img*(ants_seg!=0)
    return tuple([ants_img,ants_seg])

def normalize(img):
    return (img- img.min())/(img.max() - img.min())

def ants_to_numpy(ants_img,norm=False,axis=0):
    img_np = ants_img.numpy()
    if norm:
        img_np = normalize(img_np)
    if axis==0:
        img_np = img_np[50:114,:,:]
    else:
        img_np = img_np[:,96:160,:]
#         img_np = img_np[100:164,:,:]
        img_np = img_np.transpose((1,0,2))
    return img_np[np.newaxis,...]
    
def apply_tfm(fixed,moving,tfm_path,interp):
    ants_warped = ants.apply_transforms(fixed= fixed,
                                 moving= moving,
                                 transformlist= tfm_path,
                                 interpolator= interp
                                )
    return ants_warped
    
def load_volfile(filename, reference,axis):

    ants_img, ants_seg = read_ants(filename)
    ants_ref, seg_ref = read_ants(reference)
    
    tfm_path = ants_dir+ filename.split('/')[2]+ '0GenericAffine.mat'
    
    ants_ref = apply_tfm(ants_img, ants_ref,tfm_path,'linear')
    seg_ref = apply_tfm(ants_seg, seg_ref,tfm_path,'nearestNeighbor')
 
    return (ants_to_numpy(ants_ref,True,axis),ants_to_numpy(seg_ref,axis=axis), ants_to_numpy(ants_img,True,axis),ants_to_numpy(ants_seg,axis=axis))

def numpy_to_torch(img_list,device):
    imgs_np = np.concatenate(img_list, axis=0)
    imgs_torch = torch.from_numpy(imgs_np).float()
    return imgs_torch.to(device)

def vxm_start_generator(start,vol_names, batch_size, reference, device):
    
    indices = np.arange(start,start+ batch_size,dtype=np.uint8) 
    img_segs = [load_volfile(vol_names[i], reference) for i in indices]
    volumes = {'scan1' : [],
          'seg1': [],
          'scan2': [],
          'seg2': []
          }
    for i,key in enumerate(volumes.keys()):
        volumes[key].append([img_segs[x][i] for x in range(len(img_segs))])

    scan1, seg1, scan2, seg2 = tuple([numpy_to_torch(volumes[key],device) for key in volumes.keys()])

    shape = scan1.shape[2:]
    zeros = torch.zeros((batch_size, len(shape), *shape)).to(device)

    invols = [scan1, scan2]
    outvols = [scan2, zeros] 
    seg_vols = [seg1, seg2]
        
    return (invols, outvols,seg_vols)

class CustomImageDataset(Dataset):
   
    def __init__(self, vol_names, reference, do_transform=0,axis=0):
        
        self.filenames = vol_names
        self.reference = reference
        self.erase = transforms.functional.erase
        self.rotate = transforms.functional.rotate
        self.do_transform = do_transform
        self.axis = axis
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        img_path = self.filenames[idx]
        input_vols = load_volfile(img_path, self.reference,self.axis)
        input_vols = [torch.from_numpy(vol).float() for vol in input_vols]
        
        if self.do_transform == 1:
            i = np.random.randint(70,256-70)
            j = np.random.randint(70,256-70)
            h = np.random.randint(10,30)
            w = np.random.randint(10,30)
            fill = np.random.uniform(0,1)

            erased_img = self.erase(input_vols[2],i,j,h,w,fill)
            input_vols[2] = erased_img
            
        if self.do_transform == 2: 
            angle = np.random.randint(0,90)
            r_vols = [self.rotate(x,angle) for x in input_vols]
            return r_vols,img_path
        if self.do_transform == 3: 
            sigma = random.uniform(0.0,4.0)
            blurred=gaussian(input_vols[2], sigma=sigma, mode='nearest')
            input_vols[2]=blurred
        
#         input_vols[2] = input_vols[2]*(input_vols[3]!=0)
        return input_vols,img_path