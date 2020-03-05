#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:49:59 2020

@author: M. Leenstra
"""
import os
import random
import numpy as np
import pandas as pd

def list_directories(directory):
    dirlist = []
    for f in os.listdir(directory): 
        if os.path.isdir(os.path.join(directory,f)):
            dirlist.append(f)
    return dirlist

class AverageMeter(object):
    """
    Computes and stores the average and current value
    source: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py
    """
    
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def cv_test_split(csv_file, n_test_ims, folds, k):
    """ split dataset in a train, validation and test split based on original
    images to avoid data leakage. Cross-validation based split for the train
    and validation sets.
    
    Parameters
    ----------
    csvfile: string
        path to file with info about the saved patches
    n_test_patches: int
        number of test patches to extract from the dataset
    folds: int
        number of folds for cross-validation
    k: int
        kth fold to use for validation
    
    Returns
    -------
    index_train: numpy.ndarray of shape (N,)
        indices of patches to be used for training
    index_val: numpy.ndarray of shape (N,)
        indices of patches to be used for validation
    index test: numpy.ndarray of shape (n,)
        indices of patches to be used for testing
    """
    dataset = pd.read_csv(csv_file, sep=',')
    dataset = dataset.loc[dataset['patchpair_idx'] == 0]
    
    # train / test split
    dataset_ims = np.unique(dataset['im_idx'])
    np.random.seed(234)
    test_ims = np.random.choice(dataset_ims, n_test_ims, replace=False)

    # remove test images
    dataset_ims = dataset_ims[np.isin(dataset_ims, test_ims) == False]
    
    # shuffle and save train-part
    random.seed(890)
    random.shuffle(dataset_ims)
    
    ims_per_fold = int(len(dataset_ims)/folds)
    
    # train / val split  
    val_ims = dataset_ims[k*ims_per_fold:(k+1)*ims_per_fold]
    train_ims = dataset_ims[np.isin(dataset_ims, val_ims) == False]
    
    # find indices
    dataset_val = dataset[np.isin(dataset['im_idx'], val_ims)]
    dataset_train = dataset[np.isin(dataset['im_idx'], train_ims)]
    dataset_test = dataset[np.isin(dataset['im_idx'], test_ims)]
    
    index_val = np.array(dataset_val['im_patch_idx'])
    index_train = np.array(dataset_train['im_patch_idx'])
    index_test = np.array(dataset_test['im_patch_idx'])
    
    random.seed(123)
    random.shuffle(index_val)
    random.shuffle(index_train)
    random.shuffle(index_test)
    
    return index_train, index_val, index_test
