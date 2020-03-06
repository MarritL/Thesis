#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:13:04 2020

@author: M. Leenstra
"""
import numpy as np
import os
from models.netbuilder import NetBuilder
import torch
from plots import normalize2plot
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu




def extract_features(directories, imagelist, model_settings, layers):
    """
    extract features from specified layers (in branch) using trained model 

    Parameters
    ----------
    directories : dict
        dictionary with strings specifying directories
    image : numpy nd.array of shape (N,M,D)
        image to extract features from, D should be equal to n_channels in model_settings.
    model_settings : dict 
        settings of trained model
    layers : list
        list of integers, representing the layers to extract features from.
        layers should be in branch of model.

    Raises
    ------
    Exception
        if architecture specified in model_settings is undefied an exception is raised

    Returns
    -------
    np.ndarray of shape (N,M,D)
        al featuremaps are upsampled to original image size and concatenated in D.

    """
          

    # cfgs are saved as strings, cast back to list
    branch = model_settings['cfg_branch'].split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    branch = branch.replace(" ", "")
    branch = branch.split(",")
    top = model_settings['cfg_top'].split("[" )[1]
    top = top.split("]" )[0]
    top = top.replace("'", "")
    top = top.replace(" ", "")
    top = top.split(",")
    classifier = model_settings['cfg_classifier'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    # save in model_settigns
    model_settings['cfg'] = {'branch': np.array(branch, dtype='object'), 
                             'top': np.array(top,dtype='object'),
                             'classifier': np.array(classifier, dtype='object')}
    
    # batch_norm saved as string cast back to bool
    if model_settings['batch_norm'] == 'False' : 
        model_settings['batch_norm'] = False
    elif model_settings['batch_norm'] == 'True' : 
        model_settings['batch_norm'] = True
          
    # build network
    model_settings['network'] = model_settings['networkname']
    if model_settings['network'] == 'siamese':
        n_branches = 2
    elif model_settings['network'] == 'triplet':
        n_branches = 3
        model_settings['network'] = 'siamese'
    elif model_settings['network'] == 'hypercolumn':
        n_branches = 2
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet", hypercolumn')
        
        
    net = NetBuilder.build_network(
        net=model_settings['network'],
        cfg=model_settings['cfg'],
        n_channels=int(model_settings['n_channels']), 
        n_classes=int(model_settings['n_classes']),
        patch_size=int(model_settings['patch_size']),
        im_size=imagelist[0].shape[:2],
        batch_norm=model_settings['batch_norm'],
        n_branches=n_branches)  
    
    # load weights
    net.load_state_dict(torch.load(model_settings['filename']))
    
    outlist = list()
    for j, im in enumerate(imagelist): 
        # prepare image
        im_n = normalize2plot(im)
        im_n = np.moveaxis(im_n, -1, 0)
        im_n = np.expand_dims(im_n, axis=0)
        im_t = torch.as_tensor(im_n)
        
# =============================================================================
#         im_n = normalize2plot(imagelist[0])
#         im_n = np.moveaxis(im_n, -1, 0)
#         im_n = np.expand_dims(im_n, axis=0)
#         im_t1 = torch.as_tensor(im_n)
#         
#         im_n = normalize2plot(imagelist[1])
#         im_n = np.moveaxis(im_n, -1, 0)
#         im_n = np.expand_dims(im_n, axis=0)
#         im_t2 = torch.as_tensor(im_n)
# =============================================================================
    
        # get features
        im_out = net([im_t.float(),im_t], 1, extract_features=layers)
        
        # upsample if needed and cast back to numpy array
        for i, fl in enumerate(im_out):
            if i == 0:
                ref_size = fl.shape[2:]
            if fl.shape[2:] != ref_size:
                fl = nn.functional.interpolate(fl, size=ref_size, mode='bilinear', align_corners=True)
            im_out[i] = np.moveaxis(np.squeeze(fl.detach().numpy()), 0,-1)
        
        # concat
        outlist.append(np.concatenate(im_out,axis=2))
        
    return outlist

def calculate_distancemap(f1, f2):
    """
    calcualtes pixelwise euclidean distance between images with multiple imput channels

    Parameters
    ----------
    f1 : np.ndarray of shape (N,M,D)
        image 1 with the channels in the third dimension 
    f2 : np.ndarray of shape (N,M,D)
        image 2 with the channels in the third dimension 
        
    Returns
    -------
    np.ndarray of shape(N,M)
        pixelwise euclidean distance between image 1 and image 2

    """
    dist_per_fmap= [(f1[:,:,i]-f2[:,:,i])**2 for i in range(f1.shape[2])]
    
    return np.sqrt(sum(dist_per_fmap))

def calculate_differencemaps(f1, f2):
    """
    calcualtes pixelwise difference between images with multiple imput channels

    Parameters
    ----------
    f1 : np.ndarray of shape (N,M,D)
        image 1 with the channels in the third dimension 
    f2 : np.ndarray of shape (N,M,D)
        image 2 with the channels in the third dimension 
        
    Returns
    -------
    np.ndarray of shape(N,M)
        pixelwise difference between image 1 and image 2

    """
    diff_per_fmap= [abs(f1[:,:,i]-f2[:,:,i]) for i in range(f1.shape[2])]
    
    return diff_per_fmap

def calculate_changemap(distmap, plot=False):
    """
    create binary change map using otsu thresholding

    Parameters
    ----------
    distmap : np.ndarry of shape (N,M)
        pixelwise distance map bewteen 2 images

    Returns
    -------
    binary : TYPE
        DESCRIPTION.

    """
    
    thresh = threshold_otsu(distmap)
    binary =  distmap > thresh
    
    if plot:
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
        
        ax[0].imshow(distmap, cmap=plt.cm.gray)
        ax[0].set_title('Distance map')
        ax[0].axis('off')
        
        ax[1].hist(distmap.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(thresh, color='r')
        
        ax[2].imshow(binary, cmap=plt.cm.gray)
        ax[2].set_title('Change map')
        ax[2].axis('off')
        
        plt.show() 
    
    return binary