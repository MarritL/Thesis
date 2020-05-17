#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:23:30 2020

@author: M. Leenstra
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from math import ceil
from tqdm import tqdm

class BaseDataset(Dataset):
    """Base dataset with common parameters and functions """

    def __init__(self, data_dir, indices, channels, patch_size, percentile):
        'Initialization'
        self.data_dir = data_dir
        self.indices = indices
        self.channels = channels
        self.n_channels = len(channels)
        self.patch_size = patch_size
        self.percentile = percentile
        # TODO: put back to ['a', 'b']
        self.pair_indices = ['a', 'b']
        #self.pair_indices = ['a', 'a']

        assert len(self.indices) > 0
        print('# images: {}'.format(len(self.indices)))
           
    def __len__(self):
        'Denotes the total number of images'
        return len(self.indices)
    
    def get_images(self, filenames):
        """
        loads imags into memory and normalizes them

        Parameters
        ----------
        filenames : list
            list of filenames (strings)

        Returns
        -------
        images : list
            list of np.ndarrays of shape (N,M,D)
        """        
        images = []
        for i, filename in enumerate(filenames):
            images.append(np.load(os.path.join(self.data_dir, filename))\
                          [:, :, self.channels])

            # replace potential nan values 
            images[i][np.isnan(images[i])] = 0
            
            # normalize
            images[i] = self.normalize(images[i], self.percentile)
            assert np.any(~np.isnan(images[i])), \
                'Nans in image {} after normalization'.format(filename)     

        return images
    
    def get_images_dict(self, filenames):
        """
        loads imags into memory and normalizes them

        Parameters
        ----------
        filenames : list
            list of filenames (strings)

        Returns
        -------
        images : dict
            dict of np.ndarrays of shape (N,M,D), filenames are keys
        """        
        images = {}
        for filename in filenames:
            images[filename] = np.load(
                os.path.join(self.data_dir, filename))[:, :, self.channels]

            # replace potential nan values 
            images[filename][np.isnan(images[filename])] = 0
            
            # normalize
            images[filename] = self.normalize(images[filename], self.percentile)
            assert np.any(~np.isnan(images[filename])), \
                'Nans in image {} after normalization'.format(filename)     

        return images
    
    
    def normalize(self, image, percentile):
        """
        normalizes images
        
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
        assert len(image.shape) == 2 or len(image.shape) == 3, \
        'image should be 2D or 3D'
        
        if len(image.shape) == 3:
            for i in range(self.n_channels):
                image[:, :, i] = np.divide(
                    image[:, :, i]-image[:, :, i].min(),
                    np.percentile(image[:, :, i], percentile)
                    -image[:, :, i].min()+1E-8)
        else:
            image = np.divide(
                image-image.min(),
                np.percentile(image, percentile)
                -image.min()+1E-8)
            
        image[image > 1] = 1
        image[image < 0] = 0
            
        return image
    
    def get_patches(self, patch_starts, images, pair_idxs=None):
        """
        get the patches from the images

        Parameters
        ----------
        patch_starts : list
            list of upper left corners of the patches
        images : list
            list of np.ndarrays of shape (N,M,D)
        pair_idxs : numpy.ndarray. shape: (N,), optional
            array of indices describing the pair identifier (e.g. ['a','b']).
            default is None.      

        Returns
        -------
        patches : dict
            dict of patches, the keys are 'patch'+i in which i is the patchnumber

        """
        assert len(patch_starts) is len(images) or \
            (len(pair_idxs) is len(patch_starts) and \
            len(np.unique(pair_idxs)) <= len(images)),\
            "Number of images too small. Specify from which image the patch \
                should be extracted"
        
        n_unique = len(np.unique(pair_idxs))
        
        patches = dict()
        for i, patch_start in enumerate(patch_starts):
            key = 'patch' +str(i)
            if pair_idxs is None:
                patches[key] = images[i]\
                    [patch_start[0]:patch_start[0]+self.patch_size,
                     patch_start[1]:patch_start[1]+self.patch_size, :]
            else:
                if n_unique == 1:
                    im_idx = 0 
                else: 
                    im_idx = ord(pair_idxs[i])-97
                patches[key] = images[im_idx]\
                    [patch_start[0]:patch_start[0]+self.patch_size,
                     patch_start[1]:patch_start[1]+self.patch_size, :]
        
        return patches   
    
    def get_patches_gt(self, patch_starts, gt, images):
        """
        subset the gt patch from the gt image

        Parameters
        ----------
        patch_starts : list
            list of upper left corners of the patches
        gt: np.ndarrays of shape (N,M)
            ground truth image of change detection 
        
        Returns
        -------
        gt_sub : np.ndarray of shape (N,M)
            ground truth of patches

        """        

        patches = dict()
        patches['patch0'] = images[0][patch_starts[0][0]:patch_starts[0][0]+self.patch_size,
                     patch_starts[0][1]:patch_starts[0][1]+self.patch_size, :]
        patches['patch1'] = images[1][patch_starts[0][0]:patch_starts[0][0]+self.patch_size,
                     patch_starts[0][1]:patch_starts[0][1]+self.patch_size, :]
        patches['label'] = gt[patch_starts[0][0]:patch_starts[0][0]+self.patch_size,
                     patch_starts[0][1]:patch_starts[0][1]+self.patch_size]
        
        return patches
                                
    
class PairDataset(BaseDataset):
    """ Generate dataset of patch_pairs using method inspired on Doersch"""
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99):
        super(PairDataset, self).__init__(data_dir, indices, channels, 
                                          patch_size, percentile)
    
    def __getitem__(self, index):
        'Generates one patch pair'
        # Select sample
        im_idx = self.indices[index]
            
        # get random pairindex (i.e. a/b)
        pair_idxs = np.random.choice(self.pair_indices, size=2, replace=False)

        # 'reconstruct' the filenames
        filenames = get_filenames(im_idx, pair_idxs)

        # get imagepair
        impair = self.get_images(filenames)
        
        assert impair[0].shape == impair[1].shape, \
            'Shape not matching in image pair {}'.format(im_idx)
            
        # sample start location patches 
        patch_starts, lbl = sample_patchpair(impair[0].shape, self.patch_size)
        
        # get patches
        patchpair = self.get_patches(patch_starts, impair)
        
        # augmentation
        rot = np.random.randint(0,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchpair['patch0'] = np.rot90(patchpair['patch0'], rot, axes=(0, 1)).copy()
        patchpair['patch1'] = np.rot90(patchpair['patch1'], rot, axes=(0, 1)).copy()
        if flipud:
            patchpair['patch0'] = np.flipud(patchpair['patch0']).copy()
            patchpair['patch1'] = np.flipud(patchpair['patch1']).copy()
        if fliplr:
            patchpair['patch0'] = np.fliplr(patchpair['patch0']).copy()
            patchpair['patch1'] = np.fliplr(patchpair['patch1']).copy()

        # rearange axis (channels first)
        patchpair = channelsfirst(patchpair)
                             
        assert patchpair['patch0'].shape == patchpair['patch1'].shape, \
            "Shape not matching in patch pair {}".format(im_idx)
        assert np.any(~np.isnan(patchpair['patch0'])) \
            & np.any(~np.isnan(patchpair['patch1'])), \
            "Nans in patch pair {}".format(im_idx)  
        
        # add image info
        patchpair['im_idx'] = torch.as_tensor(im_idx)
        patchpair['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchpair['patch0'] = torch.as_tensor(patchpair['patch0'])
        patchpair['patch1'] = torch.as_tensor(patchpair['patch1'])
        patchpair['label'] = torch.as_tensor(lbl, dtype=torch.int64)
        
        return patchpair

class ShuffleBandDataset(BaseDataset):
    """ Generate dataset of patch_pairs using method inspired on Doersch"""
    
    def __init__(self, data_dir, indices, channels=[1,2,3], patch_size=96, 
                 percentile=99, one_hot=True, in_memory=False):
        super(ShuffleBandDataset, self).__init__(data_dir, indices, channels, 
                                          patch_size, percentile)
    
        self.one_hot = one_hot
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)
    
    def __getitem__(self, index):
        'Generates one patch pair'
        # Select sample
        im_idx = self.indices[index]
            
        # get random pairindex (i.e. a/b)
        pair_idxs = np.random.choice(self.pair_indices, size=1, replace=False)
        
        # 'reconstruct' the filenames
        filenames = get_filenames(im_idx, pair_idxs)
    
        # load image
        if self.in_memory:
            images = list()
            for filename in filenames:
                    images.append(self.images[filename])
        else:
            # load image
            images = self.get_images(filenames)
                    
        # sample start location patches 
        patch_starts = sample_patch(
            images[0].shape, 
            self.patch_size)
        
        # get patches
        singlepatch = self.get_patches(patch_starts, images)
        
        # rearange axis (channels first)
        singlepatch = channelsfirst(singlepatch)

        lbl = np.random.randint(6)
        r = singlepatch['patch0'][2]
        g = singlepatch['patch0'][1]
        b = singlepatch['patch0'][0]
        labels = {0: [b,g,r],
               1: [b,r,g],
               2: [r,b,g],
               3: [r,g,b],
               4: [g,r,b],
               5: [g,b,r]}
        singlepatch['patch0'] = np.array(labels[lbl])
        
        if self.one_hot:
            lbl = to_categorical(lbl, num_classes=6)
            dtype = torch.float32
        else: 
            dtype = torch.int64
                             
        assert np.any(~np.isnan(singlepatch['patch0'])), \
            "Nans in patch {}".format(im_idx)  
        
        # add image info
        singlepatch['im_idx'] = torch.as_tensor(im_idx)
        singlepatch['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        singlepatch['patch0'] = torch.as_tensor(singlepatch['patch0'])
        singlepatch['label'] = torch.as_tensor(lbl, dtype=dtype)
        
        return singlepatch
  
class PairDatasetOverlap(BaseDataset):
    """ Generate dataset of patch_pairs using method inspired on Doersch"""
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99, min_overlap=0.2, max_overlap=0.5, 
                 one_hot=True, in_memory=False):
        super(PairDatasetOverlap, self).__init__(data_dir, indices, channels, 
                                          patch_size, percentile)
    
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.one_hot = one_hot
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)
    
    def __getitem__(self, index):
        'Generates one patch pair'
        # Select sample
        im_idx = self.indices[index]
            
        # get random pairindex (i.e. a/b)
        pair_idxs = np.random.choice(self.pair_indices, size=2, replace=False)
        
        # 'reconstruct' the filenames
        filenames = get_filenames(im_idx, pair_idxs)
    
        # load image
        if self.in_memory:
            images = list()
            for filename in filenames:
                    images.append(self.images[filename])
        else:
            # load image
            images = self.get_images(filenames)
        
        assert images[0].shape == images[1].shape, \
            'Shape not matching in image pair {}'.format(im_idx)
            
        # sample start location patches 
        patch_starts, lbl = sample_patchpair_overlap(
            images[0].shape, 
            self.patch_size, 
            self.min_overlap, 
            self.max_overlap)
        
        # get patches
        patchpair = self.get_patches(patch_starts, images)
        
        # augmentation
        rot = np.random.randint(0,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchpair['patch0'] = np.rot90(patchpair['patch0'], rot, axes=(0, 1)).copy()
        patchpair['patch1'] = np.rot90(patchpair['patch1'], rot, axes=(0, 1)).copy()
        if flipud:
            patchpair['patch0'] = np.flipud(patchpair['patch0']).copy()
            patchpair['patch1'] = np.flipud(patchpair['patch1']).copy()
        if fliplr:
            patchpair['patch0'] = np.fliplr(patchpair['patch0']).copy()
            patchpair['patch1'] = np.fliplr(patchpair['patch1']).copy()

        # rearange axis (channels first)
        patchpair = channelsfirst(patchpair)
                             
        assert patchpair['patch0'].shape == patchpair['patch1'].shape, \
            "Shape not matching in patch pair {}".format(im_idx)
        assert np.any(~np.isnan(patchpair['patch0'])) \
            & np.any(~np.isnan(patchpair['patch1'])), \
            "Nans in patch pair {}".format(im_idx)  
        
        # add image info
        patchpair['im_idx'] = torch.as_tensor(im_idx)
        patchpair['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchpair['patch0'] = torch.as_tensor(patchpair['patch0'])
        patchpair['patch1'] = torch.as_tensor(patchpair['patch1'])
        patchpair['label'] = torch.as_tensor(lbl, dtype=torch.float32)
        
        return patchpair
    
class TripletDataset(BaseDataset):
    """ Generate dataset with patch triplets, based on partial overlap """
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99, min_overlap=0.2, max_overlap=0.5, 
                 one_hot=True, in_memory=False):
        super(TripletDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.one_hot = one_hot
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_idx = self.indices[index]
            
        # get random image-pair index (i.e. a/b)
        triplet_pairidxs = np.random.choice(self.pair_indices, size=3, replace=True)

        # 'reconstruct' the filenames
        filenames = get_filenames(im_idx, triplet_pairidxs)
        unique_images = np.unique(filenames)
        n_images = len(unique_images)
    
        # load image
        if self.in_memory:
            images = list()
            for filename in unique_images:
                    images.append(self.images[filename])
        else:
            # load image
            images = self.get_images(unique_images)

        # check if all images have the same shape
        if n_images > 1:
            for i in range(n_images-1):
                assert images[i].shape == images[i+1].shape,\
                    'Shape not matching in image pair {}'.format(im_idx)

        # sample start locations patches
        if self.min_overlap == 1 & self.max_overlap == 1:
            patch_starts = sample_patchtriplet_apn(
                images[0].shape, self.patch_size)
        else:
            patch_starts = sample_patchtriplet(
                images[0].shape, 
                self.patch_size, 
                min_overlap=self.min_overlap, 
                max_overlap=self.max_overlap)
        
        # get a random order
        lbl = np.random.randint(2)
        if lbl == 1:
            patch_starts[1], patch_starts[2] = patch_starts[2], patch_starts[1]
        
        if self.one_hot:
            lbl = to_categorical(lbl, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # get patches
        patchtriplet = self.get_patches(patch_starts, images, triplet_pairidxs)
        
        # augmentation
        rot = np.random.randint(0,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchtriplet['patch0'] = np.rot90(patchtriplet['patch0'], rot, axes=(0, 1)).copy()
        patchtriplet['patch1'] = np.rot90(patchtriplet['patch1'], rot, axes=(0, 1)).copy()
        patchtriplet['patch2'] = np.rot90(patchtriplet['patch2'], rot, axes=(0, 1)).copy()
        if flipud:
            patchtriplet['patch0'] = np.flipud(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.flipud(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.flipud(patchtriplet['patch2']).copy()
        if fliplr:
            patchtriplet['patch0'] = np.fliplr(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.fliplr(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.fliplr(patchtriplet['patch2']).copy()
    
        # rearange axis (channels first)
        patchtriplet = channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == \
            patchtriplet['patch1'].shape == \
            patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(
                im_idx, 
                patchtriplet['patch0'].shape, 
                patchtriplet['patch1'].shape,
                patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_idx)  
        
        # add image info
        patchtriplet['im_idx'] = torch.as_tensor(im_idx)
        patchtriplet['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl, dtype=dtype)
        
        
        return patchtriplet
    
class TripletFromFileDataset(BaseDataset):
    """ Generate dataset with patch triplets, based on partial overlap """
    
    def __init__(self, data_dir, indices, patch_starts_df,
                 channels=np.arange(14), patch_size=96, 
                 percentile=99, one_hot=True, in_memory=False):
        super(TripletFromFileDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.patch_starts_df = patch_starts_df
        self.one_hot = one_hot
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        patches_data = self.patch_starts_df.iloc[index]
        # Select sample
        im_idx = patches_data['im_idx']
            
        # get random image-pair index (i.e. a/b)
        triplet_pairidxs = np.array([patches_data['im_patch0'], patches_data['im_patch1'], patches_data['im_patch2']])

        # 'reconstruct' the filenames
        filenames = get_filenames(im_idx, triplet_pairidxs)
        unique_images = np.unique(filenames)
        n_images = len(unique_images)
    
        # load image
        if self.in_memory:
            images = list()
            for filename in unique_images:
                    images.append(self.images[filename])
        else:
            # load image
            images = self.get_images(unique_images)

        # check if all images have the same shape
        if n_images > 1:
            for i in range(n_images-1):
                assert images[i].shape == images[i+1].shape,\
                    'Shape not matching in image pair {}'.format(im_idx)

        # read start locations patches        
        patch_starts = [np.array([patches_data['row_patch0'], patches_data['col_patch0']]),
                        np.array([patches_data['row_patch1'], patches_data['col_patch1']]),
                        np.array([patches_data['row_patch2'], patches_data['col_patch2']])]
        
        # read random order
        lbl = patches_data['lbl']
        if lbl == 1:
            patch_starts[1], patch_starts[2] = patch_starts[2], patch_starts[1]
        
        if self.one_hot:
            lbl = to_categorical(lbl, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # get patches
        patchtriplet = self.get_patches(patch_starts, images, triplet_pairidxs)
        
        # augmentation
        rot = patches_data['rot']
        flipud = patches_data['flipud']
        fliplr = patches_data['fliplr']
        patchtriplet['patch0'] = np.rot90(patchtriplet['patch0'], rot, axes=(0, 1)).copy()
        patchtriplet['patch1'] = np.rot90(patchtriplet['patch1'], rot, axes=(0, 1)).copy()
        patchtriplet['patch2'] = np.rot90(patchtriplet['patch2'], rot, axes=(0, 1)).copy()
        if flipud:
            patchtriplet['patch0'] = np.flipud(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.flipud(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.flipud(patchtriplet['patch2']).copy()
        if not flipud:
            if fliplr:
                patchtriplet['patch0'] = np.fliplr(patchtriplet['patch0']).copy()
                patchtriplet['patch1'] = np.fliplr(patchtriplet['patch1']).copy()
                patchtriplet['patch2'] = np.fliplr(patchtriplet['patch2']).copy()
        

        # rearange axis (channels first)
        patchtriplet = channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == \
            patchtriplet['patch1'].shape == \
            patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(
                im_idx, 
                patchtriplet['patch0'].shape, 
                patchtriplet['patch1'].shape,
                patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_idx)  
        
        # add image info
        patchtriplet['im_idx'] = torch.as_tensor(im_idx)
        patchtriplet['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl, dtype=dtype)
        
        
        return patchtriplet
    
class PairHardNegDataset(BaseDataset):
    """ Generate dataset with patch pairs, 100% overlap, second possible rotated """
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99, one_hot=True, in_memory=False):
        super(PairHardNegDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        self.one_hot = one_hot
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)

    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_idx = self.indices[index] 
        
        # get random image-pair index (i.e. a/b)
        pairidxs = np.random.choice(self.pair_indices, size=2, replace=True)
    
        # load image
        filenames = get_filenames(im_idx, pairidxs)
        if self.in_memory:
            images = list()
            for filename in filenames:
                    images.append(self.images[filename])
        else:
            # 'reconstruct' the filenames           
            unique_images = np.unique(filenames)
            # load image
            images = self.get_images(unique_images)

        # sample start locations patches
        patch_starts = sample_patchpair_ap(
                images[0].shape, self.patch_size)
        
        # get patches
        patchpair = self.get_patches(patch_starts, images, pairidxs)
        
        # get a random label
        lbl = np.random.randint(2)
        if lbl == 1:
            rot = np.random.randint(1,4)
            flipud = np.random.randint(2)
            fliplr = np.random.randint(2)
            patchpair['patch1'] = np.rot90(patchpair['patch1'], rot, axes=(0, 1)).copy()
            if flipud:
                patchpair['patch1'] = np.flipud(patchpair['patch1']).copy()
            if not flipud:
                if fliplr:
                    patchpair['patch1'] = np.fliplr(patchpair['patch1']).copy()

        if self.one_hot:
            lbl = to_categorical(lbl, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # rearange axis (channels first)
        patchpair = channelsfirst( patchpair)
                             
        assert patchpair['patch0'].shape == \
             patchpair['patch1'].shape, \
            "Shape not matching in  patchpair {}, shape: 0:{} 1:{}"\
            .format(
                im_idx, 
                patchpair['patch0'].shape, 
                patchpair['patch1'].shape)
        assert np.any(~np.isnan(patchpair['patch0'])) \
            & np.any(~np.isnan(patchpair['patch1'])), \
            "Nans in patchpair {}".format(im_idx)  
        
        # add image info
        patchpair['im_idx'] = torch.as_tensor(im_idx)
        patchpair['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchpair['patch0'] = torch.as_tensor(patchpair['patch0'])
        patchpair['patch1'] = torch.from_numpy(patchpair['patch1'])
        patchpair['label'] = torch.as_tensor(lbl, dtype=dtype)

        
        return patchpair
    

class TripletAPNDataset(BaseDataset):
    """ Generate dataset with patch triplets, anchor and positive 100% overlap """
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99, one_hot=True, second_label=False, in_memory=False):
        super(TripletAPNDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.one_hot = one_hot
        self.second_label = second_label
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_idx = self.indices[index]
            
        # get random image-pair index (i.e. a/b)
        triplet_pairidxs = np.random.choice(self.pair_indices, size=3, replace=True)
        #triplet_pairidxs = np.array(['a', 'b', 'a'])

        # load image
        filenames = get_filenames(im_idx, triplet_pairidxs)
        unique_images = np.unique(filenames)
        n_images = len(unique_images)
        if self.in_memory:
            images = list()
            for filename in unique_images:
                    images.append(self.images[filename])
        else:
            # 'reconstruct' the filenames           
            # load image
            images = self.get_images(unique_images)

        # check if all images have the same shape
        if n_images > 1:
            for i in range(n_images-1):
                assert images[i].shape == images[i+1].shape,\
                    'Shape not matching in image pair {}'.format(im_idx)

        # sample start locations patches
        patch_starts = sample_patchtriplet_apn(
            images[0].shape, 
            self.patch_size)
        
        
        # label for l1 loss
        lbl = np.zeros((self.patch_size, self.patch_size)) # if calculating pixelwise
        
        if self.second_label:
            lbl_bottle = np.random.randint(2)
            if lbl_bottle == 1:
                patch_starts[1], patch_starts[2] = patch_starts[2], patch_starts[1]
            
            if self.one_hot:
                lbl_bottle = to_categorical(lbl_bottle, num_classes=2)
                dtype = torch.float32
            else: 
                dtype = torch.int64
        
        # get patches
        patchtriplet = self.get_patches(patch_starts, images, triplet_pairidxs)
        
        # augmentation
        rot = np.random.randint(0,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchtriplet['patch0'] = np.rot90(patchtriplet['patch0'], rot, axes=(0, 1)).copy()
        patchtriplet['patch1'] = np.rot90(patchtriplet['patch1'], rot, axes=(0, 1)).copy()
        patchtriplet['patch2'] = np.rot90(patchtriplet['patch2'], rot, axes=(0, 1)).copy()
        if flipud:
            patchtriplet['patch0'] = np.flipud(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.flipud(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.flipud(patchtriplet['patch2']).copy()
        if fliplr:
            patchtriplet['patch0'] = np.fliplr(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.fliplr(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.fliplr(patchtriplet['patch2']).copy()

        # rearange axis (channels first)
        patchtriplet = channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == \
            patchtriplet['patch1'].shape == \
            patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(
                im_idx, 
                patchtriplet['patch0'].shape, 
                patchtriplet['patch1'].shape,
                patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_idx)  
        
        # add image info
        patchtriplet['im_idx'] = torch.as_tensor(im_idx)
        patchtriplet['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl)
        if self.second_label:
            patchtriplet['label_bottle'] = torch.as_tensor(lbl_bottle, dtype=dtype)
        
        return patchtriplet

class TripletAPNHardNegDataset(BaseDataset):
    """ Generate dataset with patch triplets, anchor and positive 100% overlap """
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99, one_hot=True, second_label=False, in_memory=False):
        super(TripletAPNHardNegDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.one_hot = one_hot
        self.second_label = second_label
        self.in_memory = in_memory
        
        images = [str(i)+'_a.npy' for i in indices]
        images.extend([str(i)+'_b.npy' for i in indices])
        unique_images = np.unique(images)
        if self.in_memory:
            # 'reconstruct' the filenames
            self.images = self.get_images_dict(unique_images)
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_idx = self.indices[index]
            
        # get random image-pair index (i.e. a/b)
        triplet_pairidxs = np.random.choice(self.pair_indices, size=3, replace=True)
        #triplet_pairidxs = np.array(['a', 'b', 'a'])

        # load image
        filenames = get_filenames(im_idx, triplet_pairidxs)
        unique_images = np.unique(filenames)
        n_images = len(unique_images)
        if self.in_memory:
            images = list()
            for filename in unique_images:
                    images.append(self.images[filename])
        else:
            # 'reconstruct' the filenames           
            # load image
            images = self.get_images(unique_images)

        # check if all images have the same shape
        if n_images > 1:
            for i in range(n_images-1):
                assert images[i].shape == images[i+1].shape,\
                    'Shape not matching in image pair {}'.format(im_idx)

        # sample start locations patches
        patch_starts = sample_patchpair_ap(
            images[0].shape, 
            self.patch_size)
        
        patch_starts.append(patch_starts[0])
                        
        # get patches
        patchtriplet = self.get_patches(patch_starts, images, triplet_pairidxs)
        
        # create negative
        rot = np.random.randint(1,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchtriplet['patch2'] = np.rot90(patchtriplet['patch2'], rot, axes=(0, 1)).copy()
        if flipud:
            patchtriplet['patch2'] = np.flipud(patchtriplet['patch2']).copy()
        if not flipud:
            if fliplr:
                patchtriplet['patch2'] = np.fliplr(patchtriplet['patch2']).copy()
                
        # label for l1 loss
        lbl = np.zeros((self.patch_size, self.patch_size)) # if calculating pixelwise
        
        if self.second_label:
            lbl_bottle = np.random.randint(2)
            if lbl_bottle == 1:
                patchtriplet['patch1'], patchtriplet['patch2'] = patchtriplet['patch2'], patchtriplet['patch1']
            
            if self.one_hot:
                lbl_bottle = to_categorical(lbl_bottle, num_classes=2)
                dtype = torch.float32
            else: 
                dtype = torch.int64

        # rearange axis (channels first)
        patchtriplet = channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == \
            patchtriplet['patch1'].shape == \
            patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(
                im_idx, 
                patchtriplet['patch0'].shape, 
                patchtriplet['patch1'].shape,
                patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_idx)  
        
        # add image info
        patchtriplet['im_idx'] = torch.as_tensor(im_idx)
        patchtriplet['patch_starts'] = torch.as_tensor(patch_starts)
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl)
        if self.second_label:
            patchtriplet['label_bottle'] = torch.as_tensor(lbl_bottle, dtype=dtype)
        
        return patchtriplet

    
class TripletDatasetPreSaved(BaseDataset):
    """ Generate dataset of patch triplets from patches saved on disk """
    
    def __init__(self, data_dir, indices, channels=np.arange(14), 
                 patch_size=96, percentile=99, one_hot=True):
        super(TripletDatasetPreSaved, self).__init__(
            data_dir, indices, channels, patch_size, percentile)
        
        self.one_hot = one_hot
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_patch_idx = self.indices[index]
            
        # get patch_pair_idxs
        pairidxs = np.arange(2)

        # 'reconstruct' the filenames
        filenames = get_filenames(im_patch_idx, pairidxs)
    
        # load image
        images = self.get_images(filenames)

        # check if all images have the same shape
        assert images[0].shape == images[1].shape,\
                    'Shape not matching in image pair {}'.format(im_patch_idx)

        # sample third patch
        options_for2 = list()
        for idx in self.indices:
            if idx.split('_')[0] == im_patch_idx.split('_')[0] \
                and idx.split('_')[1] != im_patch_idx.split('_')[1]:
                options_for2.append(idx)
        
        im_patch_idx2 = np.random.choice(options_for2, 1)       
        #pairidxs2 = np.random.choice([0, 1], size=1) 
        pairidxs2 = [pairidxs[1]]
        
        # 'reconstruct' filename
        filename = get_filenames(im_patch_idx2[0], pairidxs2)

        # load third patch
        images.extend(self.get_images(filename))
        
        # get a random order
        lbl = np.random.randint(2)
        if lbl == 1:
            images[1], images[2] = images[2], images[1]
        
        if self.one_hot:
            lbl = to_categorical(lbl, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # get patches
        patchtriplet = dict()
        patchtriplet['patch0'] = images[0]
        patchtriplet['patch1'] = images[1]
        patchtriplet['patch2'] = images[2]
        
        rot = np.random.randint(0,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchtriplet['patch0'] = np.rot90(patchtriplet['patch0'], rot, axes=(0, 1)).copy()
        patchtriplet['patch1'] = np.rot90(patchtriplet['patch1'], rot, axes=(0, 1)).copy()
        patchtriplet['patch2'] = np.rot90(patchtriplet['patch2'], rot, axes=(0, 1)).copy()
        if flipud:
            patchtriplet['patch0'] = np.flipud(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.flipud(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.flipud(patchtriplet['patch2']).copy()
        if fliplr:
            patchtriplet['patch0'] = np.fliplr(patchtriplet['patch0']).copy()
            patchtriplet['patch1'] = np.fliplr(patchtriplet['patch1']).copy()
            patchtriplet['patch2'] = np.fliplr(patchtriplet['patch2']).copy()

        # rearange axis (channels first)
        patchtriplet = channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == \
            patchtriplet['patch1'].shape == \
            patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(
                im_patch_idx, 
                patchtriplet['patch0'].shape, 
                patchtriplet['patch1'].shape,
                patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_patch_idx)  
        
        # add image info
        patchtriplet['im_idx'] = torch.as_tensor(im_patch_idx)
        patchtriplet['patch_starts'] = torch.as_tensor(filename)
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl, dtype=dtype)
        
        return patchtriplet
    
class PartlyOverlapDataset(BaseDataset):
    """ Generate dataset of patch pairs with binary gt of overlapping part """   
    def __init__(self, data_dir, indices, channels=np.arange(14), 
                 patch_size=96, percentile=99, one_hot=True):
        super(PartlyOverlapDataset, self).__init__(
            data_dir, indices, channels, patch_size, percentile)
        
        self.one_hot = one_hot
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_patch_idx = self.indices[index]
            
        # get patch_pair_idxs
        pairidxs = np.arange(2)

        # 'reconstruct' the filenames
        filenames = get_filenames(im_patch_idx, pairidxs)
    
        # load image
        images = self.get_images(filenames)

        # check if all images have the same shape
        assert images[0].shape == images[1].shape,\
                    'Shape not matching in image pair {}'.format(im_patch_idx)

        # get gt
        gt = np.load(os.path.join(
            self.data_dir,
            'gt_S21C_' + self.data_dir.split('/')[-1].split('_')[-1], 
            im_patch_idx + '.npy'))
      
        if self.one_hot:
            gt = to_categorical(gt, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # get patches
        patchpair = dict()
        patchpair['patch0'] = images[0]
        patchpair['patch1'] = images[1]
        patchpair['label'] = gt
        
        # augmentation
        rot = np.random.randint(0,4)
        flipud = np.random.randint(2)
        fliplr = np.random.randint(2)
        patchpair['patch0'] = np.rot90(patchpair['patch0'], rot, axes=(0, 1)).copy()
        patchpair['patch1'] = np.rot90(patchpair['patch1'], rot, axes=(0, 1)).copy()
        patchpair['label'] = np.rot90(patchpair['label'], rot, axes=(0, 1)).copy()
        if flipud:
            patchpair['patch0'] = np.flipud(patchpair['patch0']).copy()
            patchpair['patch1'] = np.flipud(patchpair['patch1']).copy()
            patchpair['label'] = np.flipud(patchpair['label']).copy()
        if fliplr:
            patchpair['patch0'] = np.fliplr(patchpair['patch0']).copy()
            patchpair['patch1'] = np.fliplr(patchpair['patch1']).copy()
            patchpair['label'] = np.fliplr(patchpair['label']).copy()

        # rearange axis (channels first)
        patchpair = channelsfirst(patchpair)
                             
        assert patchpair['patch0'].shape == \
            patchpair['patch1'].shape, \
            "Shape not matching in patch pair {}, shape: 0:{} 1:{}"\
            .format(
                im_patch_idx, 
                patchpair['patch0'].shape, 
                patchpair['patch1'].shape)
        assert np.any(~np.isnan(patchpair['patch0'])) \
            & np.any(~np.isnan(patchpair['patch1'])) \
            & np.any(~np.isnan(patchpair['label'])), \
            "Nans in patch pair {}".format(im_patch_idx)  
        
        # add image info
        patchpair['im_idx'] = torch.as_tensor(im_patch_idx)
        patchpair['patch_starts'] = torch.as_tensor(filenames)
        
        # cast to tensors
        patchpair['patch0'] = torch.as_tensor(patchpair['patch0'])
        patchpair['patch1'] = torch.as_tensor(patchpair['patch1'])
        patchpair['label'] = torch.as_tensor(patchpair['label'],dtype=dtype)
        
        return patchpair

class TripletAPNFinetuneDataset(BaseDataset):
    """ Generate dataset with patch pair, anchor and positive 100% overlap """
    
    def __init__(self, data_dir, data_dir_gt, indices, channels=np.arange(14), 
                 patch_size=96, percentile=99, one_hot=False, batch_size=25):
        super(TripletAPNFinetuneDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.one_hot = one_hot # todo: not hardcoded..
        self.data_dir_gt = data_dir_gt
        self.weight = torch.ones((2,), dtype=torch.float())
        self.batch_size = batch_size
    
    def __getitem__(self, index):
        'Generates one patch pair'
        # Select sample
        im_idx = self.indices[index]
        # get random image-pair index (i.e. a/b)
        #triplet_pairidxs = np.random.choice(self.pair_indices, size=3, replace=True)
        pair_pairidxs = np.array(['a', 'b'])

        # 'reconstruct' the filenames
        filenames = get_filenames(im_idx, pair_pairidxs)
        unique_images = np.unique(filenames)
        n_images = len(unique_images)
    

        # load image
        images = self.get_images(unique_images)
        gt = get_gt(self.data_dir_gt, str(im_idx)+'.npy')

        # check if all images have the same shape
        if n_images > 1:
            for i in range(n_images-1):
                assert images[i].shape == images[i+1].shape,\
                    'Shape not matching in image pair {}'.format(im_idx)
        assert images[0].shape[:2] == gt.shape,\
            'Shape gt not matchin in image pair {}'.format(im_idx)

        batch_of_patch0 = torch.zeros(
            (self.batch_size, self.n_channels, self.patch_size, self.patch_size), 
            dtype=torch.float)
        batch_of_patch1 = torch.zeros(
            (self.batch_size, self.n_channels, self.patch_size, self.patch_size), 
            dtype=torch.float)
        batch_of_labels = torch.zeros(
            (self.batch_size, self.patch_size, self.patch_size), 
            dtype=torch.long)
        
        for i in range(self.batch_size):
            # sample start locations patches
            patch_starts = sample_patchpair_ap(
                images[0].shape, 
                self.patch_size)
    
            # get gt patch and calculate number of changed pixels
            patches = self.get_patches_gt(patch_starts, gt, images)
            self.weight[0] += np.sum(patches['label'] == 2)
            self.weight[1] += np.sum(patches['label'] == 1)
            
            if self.one_hot:
                to_categorical(patches['label'], num_classes=2)
    
            # rearange axis (channels first)
            patches['patch0'] = np.moveaxis(patches['patch0'], -1, 0)
            patches['patch1'] = np.moveaxis(patches['patch1'], -1, 0)
            
            assert patches['patch0'].shape == patches['patch1'].shape,\
                "Shape not matching in patch pair {}, shape: 0:{} 1:{}"\
                .format(
                    im_idx, 
                    patches['patch0'].shape, 
                    patches['patch1'].shape)
            assert np.any(~np.isnan(patches['patch0'])) \
            & np.any(~np.isnan(patches['patch1'])), \
                "Nans in patch pair {}".format(im_idx)  
            
            # add image info
            patches['im_idx'] = torch.as_tensor(im_idx)
            patches['patch_starts'] = torch.as_tensor(patch_starts)
            
            # cast to tensors
            patches['patch0'] = torch.as_tensor(patches['patch0'])
            patches['patch1'] = torch.as_tensor(patches['patch1'])
            patches['label'] = torch.as_tensor(patches['label']-1, dtype=torch.long)
            
            batch_of_patch0[i] = patches['patch0']
            batch_of_patch1[i] = patches['patch1']
            batch_of_labels[i] = patches['label']
        
        batch = dict()
        batch['patch0'] = batch_of_patch0
        batch['patch1'] = batch_of_patch1
        batch['label'] = batch_of_labels
            
        return batch

class OSCDDataset(BaseDataset):
    """Change Detection dataset class, used for both training and test data."""
    def __init__(self, data_dir, data_dir_gt, indices, channels=np.arange(13), 
                 patch_size=96, percentile=99, one_hot=False):
        super(OSCDDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.one_hot = one_hot 
        self.data_dir_gt = data_dir_gt
        self.weight = torch.ones((2,), dtype=torch.float)
        self.stride = self.patch_size
        self.n_imgs = len(self.indices)
        
        n_pix = 0
        true_pix = 0
        pair_pairidxs = np.array(['a', 'b'])
        
        # load images
        self.im_0 = {}
        self.im_1 = {}
        self.gts = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_idx in tqdm(self.indices):
            # load and store each image
            filenames = get_filenames(im_idx, pair_pairidxs)
            images = self.get_images(filenames)
            gt = get_gt(self.data_dir_gt, str(im_idx)+'.npy')
            
            self.im_0[im_idx] = torch.as_tensor(np.moveaxis(images[0], -1, 0))
            self.im_1[im_idx] = torch.as_tensor(np.moveaxis(images[1], -1, 0))
            self.gts[im_idx] = gt
            
            s = gt.shape
            n_pix += np.prod(s)
            true_pix += np.sum(gt == 2)
            
            # calculate the number of patches
            s = self.im_0[im_idx].shape
            n1 = ceil((s[1] - self.patch_size + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_size + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_idx] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate data_dir coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_idx, 
                                    [self.stride*i, self.stride*i + self.patch_size, self.stride*j, self.stride*j + self.patch_size],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)
                    
        self.weights = [ 3 * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        
        

    def get_img(self, im_idx):
        return {'im_0': self.im_0[im_idx],'im_1': self.im_1[im_idx], 'label': self.gts[im_idx]}

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_idx = current_patch_coords[0]
        limits = current_patch_coords[1]
        
        patch0 = self.im_0[im_idx][:, limits[0]:limits[1], limits[2]:limits[3]]
        patch1 = self.im_1[im_idx][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.gts[im_idx][limits[0]:limits[1], limits[2]:limits[3]]
        #label = torch.from_numpy(1*np.array(label)).float()
        label = torch.as_tensor(label-1, dtype=torch.long)
        
        patches = {'patch0': patch0, 'patch1': patch1, 'label': label}
        
        return patches

def get_filenames(im_idx, pair_idxs):
    """
    reconstruct the filenames form the image index and pair index

    Parameters
    ----------
    im_idx : int or string
        image identifier.
    pair_idxs : list
        list of indices describing the pair identifier (e.g. 'a'/'b')

    Returns
    -------
    filenames : list
        list of filenames (strings)
    """       
    # 'reconstruct' the image filenames 
    filenames = []
    for pair_idx in pair_idxs:
        filenames.append(str(im_idx) + '_' + str(pair_idx) + '.npy')
    
    return filenames   

def get_gt(data_dir, filename):
    return np.load(os.path.join(data_dir, filename))
   
def channelsfirst(patches, src=-1, dest=0):
    """
    move axis in numpy array to get channels first

    Parameters
    ----------
    patches : dict 
        every key in the dict should describe anther patch
    src : int, optional
        source channels axis. Default is -1 (last)
    dest : int, optional
        dest channels axis. Default is 0 (first)

    Returns
    -------
    patches : dict
        dict of patches with channels first
    """
    for patch in patches:
        patches[patch] = np.moveaxis(patches[patch], src, dest)
    
    return patches

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
    
def sample_patchpair(im1_shape, patch_size=96, gap=0, maxjitter=0):
    """
    samples a patchpair from the image following the method of Doersch et al, 2015 

    Parameters
    ----------
    im1_shape : tuple
        tuple of shape of the image to smaple the patches from
    patch_size : int, optional
        width and height of the patches in pixels, patches are squared. 
        The default is 96.
    gap : int, optional
        gap between the two patches, in pixels. The default is 0.
    maxjitter : int, optional
        maximum amount to shift the patch, in pixles. The default is 0.

    Returns
    -------
    patch_starts : list
        list of upper left corners of patch 1 and patch 2. 
        patch_starts[0] : starting point of patch 1 (central patch), described
            by numpy.ndarray of shape (2,) representing resp. row, column
        patch_starts[1] : starting point of patch 2, described by numpy.ndarray
        of shape (2,) representing respectively row and column.
    patch2_lbl : int
        label describing the location of the second patch with respect to the 
        first patch. Options are: # 0 1 2 #
                                  # 3   4 #
                                  # 5 6 7 #
    """
    patch_starts = list()

    # sample jitter (if jitter is larger than 0)
    if maxjitter > 0:
        assert maxjitter < gap, "maxjitter should be smaller than gap"
        jitterrow = np.random.randint(-maxjitter, maxjitter)
        jittercol = np.random.randint(-maxjitter, maxjitter)
    else:
        jitterrow = 0
        jittercol = 0
    
    # options of patch2 with respect to patch1
    patch2_options = {0: np.array([-patch_size-gap+jitterrow, 
                                   -patch_size-gap+jittercol]), 
                      1: np.array([-patch_size-gap+jitterrow, 
                                   0+jittercol]),
                      2: np.array([-patch_size-gap+jitterrow, 
                                   patch_size+gap+jittercol]),
                      3: np.array([0+jitterrow,
                                   -patch_size-gap+jittercol]),
                      4: np.array([0+jitterrow, 
                                   patch_size+gap+jittercol]),
                      5: np.array([patch_size+gap+jitterrow, 
                                   -patch_size-gap+jittercol]),
                      6: np.array([patch_size+gap+jitterrow, 
                                   0+jittercol]),
                      7: np.array([patch_size+gap+jitterrow, 
                                   patch_size+gap+jittercol])}
    
    
    # sample the starting point of the central patch = patch1
    patch_starts.append(np.zeros((2,), dtype=np.int64))
    for i in range(2):
        patch_starts[0][i] = np.random.randint(
            patch_size+gap+maxjitter, 
            im1_shape[i]-2*patch_size-gap-maxjitter, 
            size=1)
    
    # sample the location of patch 2
    patch2_lbl = np.random.randint(8)
    patch_starts.append(patch_starts[0] + patch2_options[patch2_lbl])

# =============================================================================
#     ## TODO: turn on if testing one patch
#     patch_starts = [np.array([230, 370]), np.array([326, 466])]
#     patch2_lbl = 7
#     ##
# =============================================================================
    return (patch_starts, patch2_lbl)

def sample_patchtriplet(im1_shape, patch_size=96, min_overlap=0.2, 
                        max_overlap=0.5):
    """
    Samples a patch triplet from the image. The first two patches (0,1) are 
    partly overlapping the third patch (2) is not overlapping patch 0 and 1.

    Parameters
    ----------
    im1_shape : tuple
        tuple of shape of the image to smaple the patches from
    patch_size : int, optional
        width and height of the patches in pixels, patches are squared. 
        The default is 96.
    min_overlap : float, optional
        minimal proportion overlap between patch 1 and patch 2. 
        The default is 0.2.
    max_overlap : float, optional
        maximum proportion overlap between patch 1 and patch 2. 
        The default is 0.5.

    Returns
    -------
    patch_starts : list
        list of upper left corners of patch 0 to 2. 
        patch_starts[0] : starting point of patch 0, described by numpy.ndarray 
            of shape (2,) representing resp. row, column
        patch_starts[1] : starting point of patch 1, described by numpy.ndarray
            of shape (2,) representing respectively row and column.
        patch_starts[2] : starting point of patch2, described by numpy.ndarray
            of shape (2,) representing respectively row and column.
    """
    patch_starts = list()

    # determine min and max shift based on overlap
    min_shift_pix = patch_size - patch_size*max_overlap
    max_shift_pix = patch_size - patch_size*min_overlap
 
    # sample starting point of the central patch = patch1
    patch0_row = np.random.randint(
        np.ceil(max_shift_pix), 
        im1_shape[0]-patch_size-np.ceil(max_shift_pix), 
        size=1)
    patch0_col = np.random.randint(
        np.ceil(max_shift_pix), 
        im1_shape[1]-patch_size-np.ceil(max_shift_pix), 
        size=1)
    patch_starts.append(np.concatenate([patch0_row, patch0_col]))
    
    # sample shift of patch2 w.r.t patch1, constraind by defined overlap percentage
    shift_lr = np.random.randint(-max_shift_pix, max_shift_pix)
    max_shift_ud = np.floor(np.sqrt(max_shift_pix**2 - shift_lr**2)) 
    min_shift_ud = np.round(np.sqrt(max(0, min_shift_pix**2 - shift_lr**2)))
    random_ud = np.random.choice([-1, 1])
    if max_shift_ud == min_shift_ud:
        shift_ud = min_shift_ud*random_ud
    else:
        shift_ud = np.random.randint(min_shift_ud, max_shift_ud)*random_ud
    
    # calculate overlap
    #overlap = (patch_size - abs(shift_lr)) * (patch_size - abs(shift_ud))
    #print("overlap = ", overlap/(patch_size*patch_size))

    # save in variable
    patch_starts.append(np.array([int(patch_starts[0][0] + shift_ud),
                                  int(patch_starts[0][1] + shift_lr)]))
        
    
    # sample starting point of the 3th patch
    patch3_options = np.ones((im1_shape[0]-patch_size,
                              im1_shape[1]-patch_size),
                             dtype=np.bool)
    # make sure overlapping regions with patch 1 and 2 are excluded
    for i in range(2):
        not_start_row = int(max(patch_starts[i][0]-patch_size, 0))
        not_start_col = int(max(patch_starts[i][1]-patch_size, 0))   
        patch3_options[not_start_row:patch_starts[i][0]+patch_size,
                       not_start_col:patch_starts[i][1]+patch_size] = False

   
    idx = np.random.randint(np.where(patch3_options)[0].shape[0])
    patch3_row = np.where(patch3_options)[0][idx]    
    patch3_col = np.where(patch3_options)[1][idx]      

    patch_starts.append(np.array([patch3_row, patch3_col]))
        
# =============================================================================
#     ## TODO: turn on if testing one patch
#     patch_starts = [np.array([341,  82]), np.array([279,  51]), np.array([463, 360])]
#     ##
# =============================================================================
    return patch_starts

def sample_patchpair_overlap(im1_shape, patch_size=96, min_overlap=0.2, 
                        max_overlap=0.5):
    """
    Samples a patch pair from the image. Patch1 is partly overlapping patch0.
    Note: overlapping patch only shifted to right and down.

    Parameters
    ----------
    im1_shape : tuple
        tuple of shape of the image to smaple the patches from
    patch_size : int, optional
        width and height of the patches in pixels, patches are squared. 
        The default is 96.
    min_overlap : float, optional
        minimal proportion overlap between patch 1 and patch 2. 
        The default is 0.2.
    max_overlap : float, optional
        maximum proportion overlap between patch 1 and patch 2. 
        The default is 0.5.

    Returns
    -------
    patch_starts : list
        list of upper left corners of patch 0 to 2. 
        patch_starts[0] : starting point of patch 0, described by numpy.ndarray 
            of shape (2,) representing resp. row, column
        patch_starts[1] : starting point of patch 1, described by numpy.ndarray
            of shape (2,) representing respectively row and column.
    """
    patch_starts = list()

    # determine min and max shift based on overlap
    min_shift_pix = patch_size - patch_size*max_overlap
    max_shift_pix = patch_size - patch_size*min_overlap
   
    # sample starting point of the central patch = patch1
    patch0_row = np.random.randint(
        np.ceil(max_shift_pix), 
        im1_shape[0]-patch_size-np.ceil(max_shift_pix), 
        size=1)
    patch0_col = np.random.randint(
        np.ceil(max_shift_pix), 
        im1_shape[1]-patch_size-np.ceil(max_shift_pix), 
        size=1)
    patch_starts.append(np.concatenate([patch0_row, patch0_col]))
    
    # sample shift of patch2 w.r.t patch1, constraind by defined overlap percentage
    #shift_lr = np.random.randint(-max_shift_pix, max_shift_pix)
    shift_lr = np.random.randint(0, max_shift_pix) # only positive shift
    max_shift_ud = np.floor(np.sqrt(max_shift_pix**2 - shift_lr**2)) 
    min_shift_ud = np.round(np.sqrt(max(0, min_shift_pix**2 - shift_lr**2)))
    #random_ud = np.random.choice([-1, 1])
    if max_shift_ud == min_shift_ud:
        #shift_ud = min_shift_ud*random_ud
        shift_ud = min_shift_ud # only positive shift
    else:
        #shift_ud = np.random.randint(min_shift_ud, max_shift_ud)*random_ud
        shift_ud = np.random.randint(min_shift_ud, max_shift_ud) # only positive shift
    
    # calculate overlap
    #overlap = (patch_size - abs(shift_lr)) * (patch_size - abs(shift_ud))
    #print("overlap = ", overlap/(patch_size*patch_size))

    # save in variable
    patch_starts.append(np.array([int(patch_starts[0][0] + shift_ud),
                                  int(patch_starts[0][1] + shift_lr)]))
                

# =============================================================================
#     ## TODO: turn on if testing one patch
#     patch_starts = [np.array([341,  82]), np.array([279,  51])]
#     ##
# =============================================================================
    return patch_starts, (shift_lr, shift_ud)

def sample_patchtriplet_apn(im1_shape, patch_size=96):
    """
    Samples a patch triplet from the image. The first two patches (0,1) are 
    100% overlapping the third patch (2) is not overlapping patch 0 and 1.

    Parameters
    ----------
    im1_shape : tuple
        tuple of shape of the image to smaple the patches from
    patch_size : int, optional
        width and height of the patches in pixels, patches are squared. 
        The default is 96.

    Returns
    -------
    patch_starts : list
        list of upper left corners of patch 0 to 2. 
        patch_starts[0] : starting point of patch 0, described by numpy.ndarray 
            of shape (2,) representing resp. row, column
        patch_starts[1] : starting point of patch 1, described by numpy.ndarray
            of shape (2,) representing respectively row and column.
        patch_starts[2] : starting point of patch2, described by numpy.ndarray
            of shape (2,) representing respectively row and column.
    """
    patch_starts = list()
 
    # sample starting point of the central patch = patch1
    patch0_row = np.random.randint(0,im1_shape[0]-patch_size, size = 1)
    patch0_col = np.random.randint(0,im1_shape[1]-patch_size, size = 1)
    patch_starts.append(np.concatenate([patch0_row,patch0_col]))
    # patch start 0 and 1 are the same (100% overlap)
    patch_starts.append(np.concatenate([patch0_row,patch0_col])) 
    
    # sample starting point of the 2th patch
    patch2_options = np.ones((im1_shape[0]-patch_size, im1_shape[1]-patch_size),dtype=np.bool)
    # exclude patch 0 (no overlap with 0 and 1)
    not_start_row = max(patch_starts[0][0]-patch_size,0)
    not_start_col = max(patch_starts[0][1]-patch_size,0)   
    patch2_options[not_start_row:patch_starts[0][0]+patch_size,
                   not_start_col:patch_starts[0][1]+patch_size] = False
    idx = np.random.randint(np.where(patch2_options)[0].shape[0])
    patch2_row = np.where(patch2_options)[0][idx]    
    patch2_col = np.where(patch2_options)[1][idx]      
    
    patch_starts.append(np.array([patch2_row, patch2_col]))
    
    return patch_starts

def sample_patchpair_ap(im1_shape, patch_size=96):
    """
    Samples an overlapping patch pair from the image. 
    
    Parameters
    ----------
    im1_shape : tuple
        tuple of shape of the image to smaple the patches from
    patch_size : int, optional
        width and height of the patches in pixels, patches are squared. 
        The default is 96.

    Returns
    -------
    patch_starts : list
        list of upper left corners of patch 0 to 1. 
        patch_starts[0] : starting point of patch 0, described by numpy.ndarray 
            of shape (2,) representing resp. row, column
        patch_starts[1] : starting point of patch 1, described by numpy.ndarray
            of shape (2,) representing respectively row and column.
    """
    patch_starts = list()
 
    # sample starting point of the central patch = patch1
    patch0_row = np.random.randint(0,im1_shape[0]-patch_size, size = 1)
    patch0_col = np.random.randint(0,im1_shape[1]-patch_size, size = 1)
    patch_starts.append(np.concatenate([patch0_row,patch0_col]))
    # patch start 0 and 1 are the same (100% overlap)
    patch_starts.append(np.concatenate([patch0_row,patch0_col])) 
    
    return patch_starts

def sample_patch(im1_shape, patch_size=96):
    """
    Samples 1 patch from image. 
    
    Parameters
    ----------
    im1_shape : tuple
        tuple of shape of the image to smaple the patches from
    patch_size : int, optional
        width and height of the patches in pixels, patches are squared. 
        The default is 96.

    Returns
    -------
    patch_starts : list
        list of upper left corners of patch 0 
        patch_starts[0] : starting point of patch 0, described by numpy.ndarray 
            of shape (2,) representing resp. row, column
    """
    patch_starts = list()
 
    # sample starting point of the central patch = patch1
    patch0_row = np.random.randint(0,im1_shape[0]-patch_size, size = 1)
    patch0_col = np.random.randint(0,im1_shape[1]-patch_size, size = 1)
    patch_starts.append(np.concatenate([patch0_row,patch0_col]))
    
    return patch_starts