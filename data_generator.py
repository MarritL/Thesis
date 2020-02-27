#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:23:30 2020

@author: M. Leenstra
"""

import torch
from torch.utils.data import Dataset
import os
import numpy as np

class BaseDataset(Dataset):

    def __init__(self, data_dir, indices, channels, patch_size, percentile):
        'Initialization'
        self.data_dir = data_dir
        self.indices= indices
        self.channels = channels
        self.n_channels = len(channels)
        self.patch_size = patch_size
        self.percentile = percentile
        # TODO: put back to ['a', 'b']
        #self.pair_indices = ['a','b']
        self.pair_indices = ['a','a']

        assert len(self.indices) > 0
        print('# images: {}'.format(len(self.indices)))
           
    def __len__(self):
        'Denotes the total number of images'
        return len(self.indices)
    
    def getFilenames(self, im_idx, pair_idxs):
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
    
    def getImages(self, filenames):
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
                          [:,:,self.channels])

            # replace potential nan values 
            images[i][np.isnan(images[i])]=0
            
            # normalize
            images[i] = self.normalize(images[i], self.percentile)
            assert np.any(~np.isnan(images[i])), \
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
                image[:,:,i] = np.divide(image[:,:,i]-image[:,:,i].min(),
                    np.percentile(image[:,:,i],percentile)-image[:,:,i].min()+1E-8)
        else:
            image = np.divide(image-image.min(),
                              np.percentile(image,percentile)-image.min()+1E-8)
            
        image[image>1]=1
        image[image<0]=0
            
        return image
    
    def channelsfirst(self, patches, src=-1, dest=0):
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
    
    def getPatches(self, patch_starts, images, pair_idxs=None):
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
        for i in range(len(patch_starts)):
            key = 'patch' +str(i)
            if pair_idxs is None:
                patches[key] = images[i]\
                    [patch_starts[i][0]:patch_starts[i][0]+self.patch_size,
                     patch_starts[i][1]:patch_starts[i][1]+self.patch_size, :]
            else:
                if n_unique == 1:
                    im_idx = 0 
                else: 
                    im_idx = ord(pair_idxs[i])-97
                patches[key] = images[im_idx]\
                    [patch_starts[i][0]:patch_starts[i][0]+self.patch_size,
                     patch_starts[i][1]:patch_starts[i][1]+self.patch_size, :]
                
        return patches
    
    def to_categorical(self, y, num_classes=None, dtype='float32'):
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
        
    
class PairDataset(BaseDataset):
    
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
        filenames = self.getFilenames(im_idx, pair_idxs)

        # get imagepair
        impair = self.getImages(filenames)
        
        assert impair[0].shape == impair[1].shape, \
            'Shape not matching in image pair {}'.format(im_idx)
            
        # sample start location patches 
        patch_starts, lbl = sample_patchpair(impair[0].shape, self.patch_size)
        
        # get patches
        patchpair = self.getPatches(patch_starts, impair)

        # rearange axis (channels first)
        patchpair = self.channelsfirst(patchpair)
                             
        assert patchpair['patch0'].shape == patchpair['patch1'].shape, \
            "Shape not matching in patch pair {}".format(im_idx)
        assert np.any(~np.isnan(patchpair['patch0'])) \
            & np.any(~np.isnan(patchpair['patch1'])), \
            "Nans in patch pair {}".format(im_idx)  
        
        # add image info
        patchpair['im_idx'] = im_idx
        
        # cast to tensors
        patchpair['patch0'] = torch.as_tensor(patchpair['patch0'])
        patchpair['patch1'] = torch.as_tensor(patchpair['patch1'])
        patchpair['label'] = torch.as_tensor(lbl, dtype=torch.int64)
        
        return patchpair
    
class TripletDataset(BaseDataset):
    
    def __init__(self, data_dir, indices, channels=np.arange(14), patch_size=96, 
                 percentile=99, min_overlap = 0.2, max_overlap = 0.5, 
                 one_hot = True):
        super(TripletDataset, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.one_hot = one_hot
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_idx = self.indices[index]
            
        # get random image-pair index (i.e. a/b)
        triplet_pairidxs = np.random.choice(self.pair_indices, size=3, replace=True)

        # 'reconstruct' the filenames
        filenames = self.getFilenames(im_idx, triplet_pairidxs)
        unique_images = np.unique(filenames)
        n_images = len(unique_images)
    
        # load image
        images = self.getImages(unique_images)

        # check if all images have the same shape
        if n_images > 1:
            for i in range(n_images-1):
                assert images[i].shape == images[i+1].shape,\
                    'Shape not matching in image pair {}'.format(im_idx)

        # sample start locations patches
        patch_starts = sample_patchtriplet(images[0].shape, self.patch_size, 
            min_overlap=self.min_overlap, max_overlap = self.max_overlap)
        
        # get a random order
        lbl = np.random.randint(2)
        if lbl == 1:
            patch_starts[1],patch_starts[2] = patch_starts[2], patch_starts[1]
        
        if self.one_hot:
            lbl = self.to_categorical(lbl, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # get patches
        patchtriplet = self.getPatches(patch_starts, images, triplet_pairidxs)

        # rearange axis (channels first)
        patchtriplet = self.channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == patchtriplet['patch1'].shape \
            == patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(im_idx, patchtriplet['patch0'].shape,patchtriplet['patch1'].shape,
            patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_idx)  
        
        # add image info
        patchtriplet['im_idx'] = im_idx
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl, dtype=dtype)
        
        return patchtriplet
    
class TripletDatasetPreSaved(BaseDataset):
    
    def __init__(self, data_dir, indices, indices_patch2, channels=np.arange(14), 
                 patch_size=96, percentile=99, one_hot = True):
        super(TripletDatasetPreSaved, self).__init__(data_dir, indices, channels, 
                                             patch_size, percentile)
        
        self.indices_patch2 =  indices_patch2
        self.one_hot = one_hot
    
    def __getitem__(self, index):
        'Generates one patch triplet'
        # Select sample
        im_patch_idx = self.indices[index]
            
        # get patch_pair_idxs
        pairidxs = np.arange(2)

        # 'reconstruct' the filenames
        filenames = self.getFilenames(im_patch_idx, pairidxs)
    
        # load image
        images = self.getImages(filenames)

        # check if all images have the same shape
        assert images[0].shape == images[1].shape,\
                    'Shape not matching in image pair {}'.format(im_patch_idx)

        # sample third patch
        options_for2 = list()
        for index in self.indices_patch2:
            if index.split('_')[0] == im_patch_idx.split('_')[0] \
                and index.split('_')[1] != im_patch_idx.split('_')[1]:
                options_for2.append(index)
        
        im_patch_idx2 =  np.random.choice(options_for2, 1)       
        pairidxs2 = np.random.choice([0,1], size=1) 
        
        # 'reconstruct' filename
        filename = self.getFilenames(im_patch_idx2[0], pairidxs2)

        # load third patch
        images.extend(self.getImages(filename))
        
        # get a random order
        lbl = np.random.randint(2)
        if lbl == 1:
            images[1],images[2] = images[2], images[1]
        
        if self.one_hot:
            lbl = self.to_categorical(lbl, num_classes=2)
            dtype = torch.float32
        else: 
            dtype = torch.int64
        
        # get patches
        patchtriplet = dict()
        patchtriplet['patch0'] = images[0]
        patchtriplet['patch1'] = images[1]
        patchtriplet['patch2'] = images[2]

        # rearange axis (channels first)
        patchtriplet = self.channelsfirst(patchtriplet)
                             
        assert patchtriplet['patch0'].shape == patchtriplet['patch1'].shape \
            == patchtriplet['patch2'].shape, \
            "Shape not matching in patch triplet {}, shape: 0:{} 1:{} 2:{}"\
            .format(im_patch_idx, patchtriplet['patch0'].shape,patchtriplet['patch1'].shape,
            patchtriplet['patch2'].shape)
        assert np.any(~np.isnan(patchtriplet['patch0'])) \
            & np.any(~np.isnan(patchtriplet['patch1'])) \
            & np.any(~np.isnan(patchtriplet['patch2'])), \
            "Nans in patch triplet {}".format(im_patch_idx)  
        
        # add image info
        patchtriplet['im_idx'] = im_patch_idx
        
        # cast to tensors
        patchtriplet['patch0'] = torch.as_tensor(patchtriplet['patch0'])
        patchtriplet['patch1'] = torch.as_tensor(patchtriplet['patch1'])
        patchtriplet['patch2'] = torch.as_tensor(patchtriplet['patch2'])
        patchtriplet['label'] = torch.as_tensor(lbl, dtype=dtype)
        
        return patchtriplet
    
    
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
        patch_starts[0][i] = np.random.randint(patch_size+gap+maxjitter, 
                          im1_shape[i]-2*patch_size-gap-maxjitter, size =1)
    
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
    patch0_row = np.random.randint(np.ceil(max_shift_pix), 
                        im1_shape[0]-patch_size-np.ceil(max_shift_pix), 
                        size = 1)
    patch0_col = np.random.randint(np.ceil(max_shift_pix), 
                        im1_shape[1]-patch_size-np.ceil(max_shift_pix), 
                        size = 1)
    patch_starts.append(np.concatenate([patch0_row,patch0_col]))
    
    # sample shift of patch2 w.r.t patch1, constraind by defined overlap percentage
    shift_lr = np.random.randint(-max_shift_pix, max_shift_pix)
    max_shift_ud = np.floor(np.sqrt(max_shift_pix**2 - shift_lr**2)) 
    min_shift_ud = np.round(np.sqrt(max(0,min_shift_pix**2 - shift_lr**2)))
    random_ud = np.random.choice([-1,1])
    if max_shift_ud == min_shift_ud:
        shift_ud = min_shift_ud*random_ud
    else:
        shift_ud = np.random.randint(min_shift_ud, max_shift_ud)*random_ud
    
    # calculate overlap
    #overlap = (patch_size - abs(shift_lr)) * (patch_size - abs(shift_ud))
    #print("overlap = ", overlap/(patch_size*patch_size))

    # save in variable
    patch_starts.append(np.array([patch_starts[0][0] + shift_ud,
                                 patch_starts[0][1] + shift_lr]))
        
    
    # sample starting point of the 3th patch
    patch3_options = np.ones((im1_shape[0]-patch_size, im1_shape[1]-patch_size),dtype=np.bool)
    # make sure overlapping regions with patch 1 and 2 are excluded
    for i in range(2):
        not_start_row = max(patch_starts[i][0]-patch_size,0)
        not_start_col = max(patch_starts[i][1]-patch_size,0)   
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
