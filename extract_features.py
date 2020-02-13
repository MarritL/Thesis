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


def extract_features(directories, image, model_settings, layers):
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
    branch = branch.split("'")
    if len(branch) > 1:
        branch = branch[1::2]
    top = model_settings['cfg_top'].split("[" )[1]
    top = top.split("]" )[0]
    top = top.split("'")
    if len(top) > 1:
        top = top[1::2]  
    
    model_settings['network'] = model_settings['networkname']

    model_settings['cfg'] = {'branch': np.array(branch, dtype='object'), 
                               'top': np.array(top,dtype='object')}

              
    # build network
    if model_settings['network'] == 'siamese':
        n_branches = 2
    elif model_settings['network'] == 'triplet':
        n_branches = 3
        model_settings['network'] = 'siamese'
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet"')
        
        
    net = NetBuilder.build_network(
        net=model_settings['network'],
        cfg=model_settings['cfg'],
        n_channels=model_settings['n_channels'], 
        n_classes=model_settings['n_classes'],
        n_branches=n_branches)  
    
    net.load_state_dict(torch.load(model_settings['filename']))

    # prepare iamge
    im_n = normalize2plot(image)
    im_n = np.moveaxis(im_n, -1, 0)
    im_n = np.expand_dims(im_n, axis=0)
    im_t = torch.as_tensor(im_n)

    im_out = net([im_t.float(),im_t], 1, extract_features=layers)
    
    for i, fl in enumerate(im_out):
        if i == 0:
            ref_size = fl.shape[2:]
        if fl.shape[2:] != ref_size:
            fl = nn.functional.interpolate(fl, size=ref_size, mode='bilinear', align_corners=True)
        im_out[i] = np.moveaxis(np.squeeze(fl.detach().numpy()), 0,-1)

    return np.concatenate(im_out,axis=2)
