#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:47:50 2020

@author: cordolo
"""
# Imports
import os
import numpy as np
import random
from math import ceil
import torch
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm

from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from scipy.ndimage import zoom

# Functions
def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    
    # crop if necesary
    I = I[:s[0],:s[1]]
    si = I.shape
    
    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])
    
    return np.pad(I,((0,p0),(0,p1)),'edge')
    
def read_sentinel_img_leq60(data_dir, normalize=True):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(data_dir)[0][:-7]
    
    r = io.imread(data_dir + im_name + "B04.tif")
    s = r.shape
    g = io.imread(data_dir + im_name + "B03.tif")
    b = io.imread(data_dir + im_name + "B02.tif")
    nir = io.imread(data_dir + im_name + "B08.tif")
    
    ir1 = adjust_shape(zoom(io.imread(data_dir + im_name + "B05.tif"),2),s)
    ir2 = adjust_shape(zoom(io.imread(data_dir + im_name + "B06.tif"),2),s)
    ir3 = adjust_shape(zoom(io.imread(data_dir + im_name + "B07.tif"),2),s)
    nir2 = adjust_shape(zoom(io.imread(data_dir + im_name + "B8A.tif"),2),s)
    swir2 = adjust_shape(zoom(io.imread(data_dir + im_name + "B11.tif"),2),s)
    swir3 = adjust_shape(zoom(io.imread(data_dir + im_name + "B12.tif"),2),s)
    
    uv = adjust_shape(zoom(io.imread(data_dir + im_name + "B01.tif"),6),s)
    wv = adjust_shape(zoom(io.imread(data_dir + im_name + "B09.tif"),6),s)
    swirc = adjust_shape(zoom(io.imread(data_dir + im_name + "B10.tif"),6),s)
    
    #I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3,uv,wv,swirc),axis=2).astype('float')
    I = np.stack((uv,b,g,r,ir1,ir2,ir3,nir,nir2,wv,swirc,swir2,swir3), axis=2).astype('float')
    
    if normalize:
        #I = (I - I.mean()) / I.std()
        normalize2plot(I)

    return I

def read_sentinel_img_trio(data_dir):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    I1 = read_sentinel_img_leq60(data_dir + '/imgs_1/')
    I2 = read_sentinel_img_leq60(data_dir + '/imgs_2/')
        
    cm = io.imread(data_dir + '/cm/cm.png') != 0
    if cm.shape[2] == 4:
      cm = rgba2rgb(cm)
    cm = rgb2gray(cm) 
    
    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2,((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0,0)),'edge')
        
    return I1, I2, cm

def read_sentinel_img_pair(data_dir):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    I1 = read_sentinel_img_leq60(data_dir + '/imgs_1/')
    I2 = read_sentinel_img_leq60(data_dir + '/imgs_2/')
            
    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2,((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0,0)),'edge')
        
    return I1, I2


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)

def normalize2plot(image, percentile = 99):
    """
    normalizes images for plotting purpose

    Parameters
    ----------
    image : numpy.ndarray. shape: (N,M) or (N,M,D)
        numpy array of image. Either greyscale 2D-array or 3D array image.
    percentile : int, optional
        percentile to use for normalization (exclude outliers). The default is 98.

    Returns
    -------
    image : numpy.ndarray. shape: (N,M) or (N,M,D)
        normalized numpy array of image. 

    """
    
    assert len(image.shape) == 2 or len(image.shape) == 3, 'image should be 2D or 3D'
    
    if len(image.shape) == 3:
        for i in range(image.shape[2]):
            image[:,:,i] = np.divide(image[:,:,i]-image[:,:,i].min(),
                np.percentile(image[:,:,i],percentile)-image[:,:,i].min()+1E-8)
    else:
        image = np.divide(image-image.min(),
                          np.percentile(image,percentile)-image.min()+1E-8)
    
    image[image>1]=1
    image[image<0]=0
        
    return image


class OSCDDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, data_dir, indices, patch_size = 96, stride = None, transform=None, one_hot=True, FP_MODIFIER=3, alt_label=True, mode='train',internal_val=True):
        """
        Args:
            csv_file (string): data_dir to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.mode = mode
        self.alt_label = alt_label
        self.one_hot = one_hot
        self.transform = transform
        self.internal_val = internal_val
        self.data_dir = data_dir
        self.patch_size = patch_size
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        self.names = indices
        self.n_imgs = self.names.shape[0]
        
        n_pix = 0
        true_pix = 0
        offset = 0
        
        # load images
        self.imgs_0 = {}
        self.imgs_1 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_starts = []
        self.patches_train = list()
        self.patches_val = list()
        for im_name in tqdm(self.names):
            # load and store each image
            if self.mode == 'train':
                I0, I1, gt = read_sentinel_img_trio(os.path.join(self.data_dir,im_name))
            else:
                I0, I1 = read_sentinel_img_pair(os.path.join(self.data_dir, im_name))
            self.imgs_0[im_name] = reshape_for_torch(I0)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            if self.mode == 'train':
                self.change_maps[im_name] = gt
            
                s = gt.shape
                n_pix += np.prod(s)
                true_pix += gt.sum()
            
            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n0 = ceil((s[1] - self.patch_size + 1) / self.stride)
            n1 = ceil((s[2] - self.patch_size + 1) / self.stride)
            n_patches_i = n0 * n1
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate data_dir coordinates
            for i in range(n0):
                for j in range(n1):
                    # coordinates in (x1, y1)
                    current_patch_starts = (im_name, 
                                    [self.stride*i, self.stride*j])
                    self.patch_starts.append(current_patch_starts)
            
            np.random.seed(92)
            random_val_idx = np.random.choice(n_patches_i, size=max(int(n_patches_i*0.05),2), replace=False)
            current_patches_val = [self.patch_starts[x+offset] for x in random_val_idx]
            self.patches_val.extend(current_patches_val)
            
            train = np.ones((I0.shape[0], I0.shape[1]),dtype=np.bool)
            for p in current_patches_val:
                not_start_row = max(p[1][0]-int(self.patch_size/2),0)
                not_start_col = max(p[1][1]-int(self.patch_size/2),0)   
                train[not_start_row:p[1][0]+int(self.patch_size/2),
                         not_start_col:p[1][1]+int(self.patch_size/2)] = False
        
            train_bool = [train[s[1][0], s[1][1]] for s in self.patch_starts[offset:offset+n_patches_i]]
            for i, p in enumerate(train_bool):
                if p:
                    self.patches_train.append(self.patch_starts[i+offset])
            offset += n_patches_i
            
            
        if self.mode == 'train':   
            self.weights = [FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        else: 
            self.weights = [1,1]
            
        print('# patches: {}'.format(self.n_patches))
        
    def get_img(self, im_name):
        if self.mode == 'train':
            return {'im0': self.imgs_0[im_name], 'im1': self.imgs_1[im_name], 'gt': self.change_maps[im_name]}
        else:
            return {'im0': self.imgs_0[im_name], 'im1': self.imgs_1[im_name]}

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_start = self.patch_starts[idx]
        im_name = current_patch_start[0]
        patch_start = current_patch_start[1]
        
        P0 = self.imgs_0[im_name][:, patch_start[0]:patch_start[0]+self.patch_size, 
                                  patch_start[1]:patch_start[1]+self.patch_size]
        P1 = self.imgs_1[im_name][:, patch_start[0]:patch_start[0]+self.patch_size, 
                                  patch_start[1]:patch_start[1]+self.patch_size]
        P0P1 = [P0, P1]
        random.shuffle(P0P1)
        
        if self.mode == 'train':
            gt = self.change_maps[im_name][patch_start[0]:patch_start[0]+self.patch_size, 
                                                   patch_start[1]:patch_start[1]+self.patch_size]
            gt = torch.from_numpy(1*np.array(gt)).float()
                
            if self.alt_label:
                P0P1 = random.choices(P0P1, k=2)
                if torch.equal(P0P1[0], P0P1[1]):
                    gt[:] = 0
                label_alt = random.getrandbits(1)
                #label_alt = 0
                if label_alt == 1:
                    third_idx = idx
                    while third_idx == idx:
                        third_idx = int(min(self.n_patches-1,max(0,np.random.randint(idx-5, idx+5,size=1))))
                    third_patch_start = self.patch_starts[third_idx]
                    im2_name = third_patch_start[0]
                    patch2_start = third_patch_start[1]
                      
                    if random.getrandbits(1):
                        P2 = self.imgs_0[im2_name][:, patch2_start[0]:patch2_start[0]+self.patch_size, 
                                                   patch2_start[1]:patch2_start[1]+self.patch_size]
                    else:
                        P2 = self.imgs_1[im2_name][:, patch2_start[0]:patch2_start[0]+self.patch_size, 
                                                   patch2_start[1]:patch2_start[1]+self.patch_size]
                    P0P1[1] = P2
                    gt[:] = 99
            else:
                label_alt = 0
            if self.one_hot:
                label_alt = to_categorical(label_alt, num_classes=2)
    
            sample = {'patch0': P0P1[0], 'patch1': P0P1[1], 'gt': gt, 'label': torch.as_tensor(label_alt)}
                
            if self.transform:
                sample = self.transform(sample)
        else: 
            sample = {'patch0': P0P1[0], 'patch1': P0P1[1]}

        return sample

class RandomFlip(object):
    """Flip randomly the images in a sample."""

#     def __init__(self):
#         return

    def __call__(self, sample):
        P0, P1, gt, label_alt = sample['patch0'], sample['patch1'], sample['gt'], sample['label']
        
        if random.random() > 0.5:
            P0 =  P0.numpy()[:,:,::-1].copy()
            P0 = torch.from_numpy(P0)
            P1 =  P1.numpy()[:,:,::-1].copy()
            P1 = torch.from_numpy(P1)
            gt =  gt.numpy()[:,::-1].copy()
            gt = torch.from_numpy(gt)

        return {'patch0': P0, 'patch1': P1, 'gt': gt, 'label': label_alt}

class RandomRot(object):
    """Rotate randomly the images in a sample."""

    def __call__(self, sample):
        P0, P1, gt, label_alt = sample['patch0'], sample['patch1'], sample['gt'], sample['label']
        
        n = random.randint(0, 3)
        if n:
            P0 =  P0.numpy()
            P0 = np.rot90(P0, n, axes=(1, 2)).copy()
            P0 = torch.from_numpy(P0)
            P1 =  P1.numpy()
            P1 = np.rot90(P1, n, axes=(1, 2)).copy()
            P1 = torch.from_numpy(P1)
            gt =  gt.numpy()
            gt = np.rot90(gt, n, axes=(0, 1)).copy()
            gt = torch.from_numpy(gt)

        return {'patch0': P0, 'patch1': P1, 'gt': gt, 'label': label_alt}

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy. source: keras.utils
    
    parameters
    ----------
    y : numpy.ndarray
        class vector to be converted into a matrix
        (integers from 0 to num_classes).
    num_classes : int
        total number of classes.
    dtype: string
        The data type expected by the input, as a string
        (`float32`, `float64`, `int32`...)
    
    Returns
    -------
        A binary matrix representation of the input. The classes axis
        is placed last.
    
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class OSCDTrainDataset(OSCDDataset):
    """ Generate dataset with patch triplets, based on partial overlap """
    
    def __init__(self, data_dir, indices, patch_size = 96, stride = None, transform=None, 
                 one_hot=True, FP_MODIFIER=3, alt_label=True, mode='train',internal_val=True):
        super(OSCDTrainDataset, self).__init__(data_dir, indices, patch_size, stride, 
                                               transform, one_hot, FP_MODIFIER, alt_label, mode,internal_val)
    
    def __len__(self):
        return len(self.patches_train)
    
    def __getitem__(self, idx):
     current_patch_start = self.patches_train[idx]
     im_name = current_patch_start[0]
     patch_start = current_patch_start[1]
     
     P0 = self.imgs_0[im_name][:, patch_start[0]:patch_start[0]+self.patch_size, 
                               patch_start[1]:patch_start[1]+self.patch_size]
     P1 = self.imgs_1[im_name][:, patch_start[0]:patch_start[0]+self.patch_size, 
                               patch_start[1]:patch_start[1]+self.patch_size]
     P0P1 = [P0, P1]
     random.shuffle(P0P1)
     
     if self.mode == 'train':
         gt = self.change_maps[im_name][patch_start[0]:patch_start[0]+self.patch_size, 
                                                patch_start[1]:patch_start[1]+self.patch_size]
         gt = torch.from_numpy(1*np.array(gt)).float()
             
         if self.alt_label:
             P0P1 = random.choices(P0P1, k=2)
             if torch.equal(P0P1[0], P0P1[1]):
                 gt[:] = 0
             label_alt = random.getrandbits(1)
             #label_alt = 0
             if label_alt == 1:
                 third_idx = idx
                 while third_idx == idx:
                     third_idx = int(min(self.n_patches-1,max(0,np.random.randint(idx-5, idx+5,size=1))))
                 third_patch_start = self.patches_train[third_idx]
                 im2_name = third_patch_start[0]
                 patch2_start = third_patch_start[1]
                   
                 if random.getrandbits(1):
                     P2 = self.imgs_0[im2_name][:, patch2_start[0]:patch2_start[0]+self.patch_size, 
                                                patch2_start[1]:patch2_start[1]+self.patch_size]
                 else:
                     P2 = self.imgs_1[im2_name][:, patch2_start[0]:patch2_start[0]+self.patch_size, 
                                                patch2_start[1]:patch2_start[1]+self.patch_size]
                 P0P1[1] = P2
                 gt[:] = 99
         else:
             label_alt = 0
         if self.one_hot:
             label_alt = to_categorical(label_alt, num_classes=2)
 
         sample = {'patch0': P0P1[0], 'patch1': P0P1[1], 'gt': gt, 'label': torch.as_tensor(label_alt)}
             
         if self.transform:
             sample = self.transform(sample)
     else: 
         sample = {'patch0': P0P1[0], 'patch1': P0P1[1]}

     return sample
 
class OSCDValDataset(OSCDDataset):
    """ Generate dataset with patch triplets, based on partial overlap """
    
    def __init__(self, data_dir, indices, patch_size = 96, stride = None, transform=None, 
                 one_hot=True, FP_MODIFIER=3, alt_label=True, mode='train',internal_val=True):
        super(OSCDValDataset, self).__init__(data_dir, indices,patch_size, stride, 
                                             transform, one_hot, FP_MODIFIER, alt_label, mode,internal_val)
       
    def __len__(self):
        return len(self.patches_val)
    
    def __getitem__(self, idx):
     current_patch_start = self.patches_val[idx]
     im_name = current_patch_start[0]
     patch_start = current_patch_start[1]
     
     P0 = self.imgs_0[im_name][:, patch_start[0]:patch_start[0]+self.patch_size, 
                               patch_start[1]:patch_start[1]+self.patch_size]
     P1 = self.imgs_1[im_name][:, patch_start[0]:patch_start[0]+self.patch_size, 
                               patch_start[1]:patch_start[1]+self.patch_size]
     P0P1 = [P0, P1]
     random.shuffle(P0P1)
     
     if self.mode == 'train':
         gt = self.change_maps[im_name][patch_start[0]:patch_start[0]+self.patch_size, 
                                                patch_start[1]:patch_start[1]+self.patch_size]
         gt = torch.from_numpy(1*np.array(gt)).float()
             
         if self.alt_label:
             P0P1 = random.choices(P0P1, k=2)
             if torch.equal(P0P1[0], P0P1[1]):
                 gt[:] = 0
             label_alt = random.getrandbits(1)
             #label_alt = 0
             if label_alt == 1:
                 third_idx = idx
                 while third_idx == idx:
                     third_idx = int(min(self.n_patches-1,max(0,np.random.randint(idx-5, idx+5,size=1))))
                 third_patch_start = self.patches_train[third_idx]
                 im2_name = third_patch_start[0]
                 patch2_start = third_patch_start[1]
                   
                 if random.getrandbits(1):
                     P2 = self.imgs_0[im2_name][:, patch2_start[0]:patch2_start[0]+self.patch_size, 
                                                patch2_start[1]:patch2_start[1]+self.patch_size]
                 else:
                     P2 = self.imgs_1[im2_name][:, patch2_start[0]:patch2_start[0]+self.patch_size, 
                                                patch2_start[1]:patch2_start[1]+self.patch_size]
                 P0P1[1] = P2
                 gt[:] = 99
         else:
             label_alt = 0
         if self.one_hot:
             label_alt = to_categorical(label_alt, num_classes=2)
 
         sample = {'patch0': P0P1[0], 'patch1': P0P1[1], 'gt': gt, 'label': torch.as_tensor(label_alt)}
             
         if self.transform:
             sample = self.transform(sample)
     else: 
         sample = {'patch0': P0P1[0], 'patch1': P0P1[1]}

     return sample