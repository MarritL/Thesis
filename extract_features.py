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
from skimage.filters import threshold_otsu, threshold_local, try_all_threshold, threshold_triangle, threshold_niblack
from skimage.morphology import remove_small_objects

def get_network(model_settings):
    
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
        im_size=(96,96),
        batch_norm=model_settings['batch_norm'],
        n_branches=n_branches)  
    
    # load weights
    net.load_state_dict(torch.load(model_settings['filename']))
    
    return net

def extract_features(net, layers, imagepair):
    
    featlist = list()
    for j, im in enumerate(imagepair): 
        outlist = list()
        # prepare image
        im_n = normalize2plot(im)
        im_n = np.moveaxis(im_n, -1, 0)
        im_n = np.expand_dims(im_n, axis=0)
        im_t = torch.as_tensor(im_n)
            
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
        outlist.extend(im_out)
        featlist.append(outlist)
        
    return featlist



def extract_features_perImage(directories, imagelist, model_settings, layers):
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
        im_size=(96,96),
        batch_norm=model_settings['batch_norm'],
        n_branches=n_branches)  
    
    # load weights
    net.load_state_dict(torch.load(model_settings['filename']))
              
    featlist = list()
    for j, im in enumerate(imagelist): 
        outlist = list()
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
        outlist.extend(im_out)
        featlist.append(outlist)
        
    return outlist

def get_dcv(features, layers_diffmap, height, width, prop):
    dcv = list()
    for i in layers_diffmap:
        diffmaps = calculate_differencemaps(features[0][i], features[1][i])
        # calculate variance in feature of each split 
        var_split0 = [(fl,np.var(diffmap[:height,:width])) for fl, diffmap in enumerate(diffmaps)]
        var_split1 = [(fl,np.var(diffmap[:height, width:])) for fl, diffmap in enumerate(diffmaps)]
        var_split2 = [(fl,np.var(diffmap[height:,:width])) for fl, diffmap in enumerate(diffmaps)]
        var_split3 = [(fl,np.var(diffmap[height:, width:])) for fl, diffmap in enumerate(diffmaps)]
        var_split0.sort(key=lambda tup: tup[1], reverse=True)
        var_split1.sort(key=lambda tup: tup[1], reverse=True)
        var_split2.sort(key=lambda tup: tup[1], reverse=True)
        var_split3.sort(key=lambda tup: tup[1], reverse=True)
        # select n features with heighest variance
        n = int(prop* len(diffmaps))
        var_layers = list()
        var_layers.extend([l[0] for l in var_split0[:n]])
        var_layers.extend([l[0] for l in var_split1[:n]])
        var_layers.extend([l[0] for l in var_split2[:n]])
        var_layers.extend([l[0] for l in var_split3[:n]])
        var_layers = np.unique(np.array(var_layers))
        
        dcv.extend([diffmaps[i] for i in var_layers])
          
    return dcv

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
    dist_per_fmap= [(f2[:,:,i]-f1[:,:,i])**2 for i in range(f1.shape[2])]
    
    return np.sqrt(sum(dist_per_fmap))

def calculate_magnitudemap(dcv):
    """
    calculates pixelwise change magnitude in deep change hypervector 

    Parameters
    ----------
    dcv: list
        each item in the list is a featuremap
        
    Returns
    -------
    np.ndarray of shape(N,M)
        pixelwise deep magnitude of combined featuremaps 

    """
    sq_dcv= [g**2 for g in dcv]
    
    return np.sqrt(sum(sq_dcv))

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
    diff_per_fmap= [f2[:,:,i]-f1[:,:,i] for i in range(f1.shape[2])]
    
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
    #thresh = threshold_local(distmap, block_size=101, method='gaussian')
    #thresh = threshold_triangle(distmap)
    #thresh = threshold_niblack(distmap, window_size=101)
    #fig, ax = try_all_threshold(distmap_norm)
    #fig, ax = plt.subplots()
    #d = ax.imshow(thresh)
    #fig.colorbar(d)
    
    binary =  distmap > thresh
    binary = remove_small_objects(binary,min_size=7)
    
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


