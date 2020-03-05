#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:34:53 2020

@author: M. Leenstra
"""

import torch
import torch.nn as nn
import numpy as np

__all__ = ['hypercolumn_net']

class Hypercolumn(nn.Module):
    def __init__(self, branches, classifier, upsample_size):
        super(Hypercolumn, self).__init__()
                
        # siamese branches
        self.branches = branches 
        
        # interpolation
        self.interpol = nn.Upsample(size=upsample_size, mode='bilinear', align_corners=True)
        
        # classifiction layers
        self.classifier = classifier

    def forward(self, data, n_branches, extract_features=None):
        res = list()
        for i in range(n_branches): # Siamese/triplet nets; sharing weights
            x = data[i]
            
            # if in feature extracting phase, extract hypercolumn for specified features
            if extract_features is not None:
                activations = dict()
                names = list()
                for i, l in enumerate(self.branches):
                        names.append('x'+str(i))
                        if i == 0:
                            activations[names[i]] = l(x)
                        else:
                            activations[names[i]] = l(activations[names[i-1]])
                            
                # return a list of features
                features_list = [x]
                features_list.extend([activations[names[i]] for i in extract_features])
                  
                # construct hypercolumn
                features = features_list[0]
                for i in range(1,len(features_list)): 
                    features = torch.cat((features, self.interpol(features_list[i])),1)
            
                #return features
            
            # if in training or validation phase forward images through branch   
                res.append(features)
        
        # concatenate the output of both branches
        x = torch.cat(res, 1)
        
        # classification layers
        x = self.classifier(x)
        
        return x                    
    
def make_layers(cfg, n_channels, batch_norm=False, kernel_size=3):
    layers = []
    in_channels = int(n_channels)
    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def hypercolumn_net(cfg, n_channels=13, n_classes=2, im_size=(96,96), batch_norm=False, n_branches=2):
    """
    Create network

    Parameters
    ----------
    cfg : dictionary
        dictionairy specifying architecture of branches and top of network. 
        example: {'branch': np.array([64, 'M', 128, 'M']), 'top': np.array([256])}
        integers for number of filters in conv-layers, 'M' for maxpool. 
        ReLU is added after each conv-layer, batch-norm optional.
    n_channels : int, optional
        number of input channels. The default is 13.
    n_classes : int, optional
        number of output classes. The default is 2.
    patch_size : int, optional
        input size of patch (squared). The default is 96.
    batch_norm : boolean, optional
        whether to use batch norm are not. The default is False.
    n_branches : int, optional
        number of branches to use in network. The default is 2.

    Returns
    -------
    net : nn.Module
        deep learning network.

    """
    # create layers
    branches = make_layers(cfg['branch'],n_channels,batch_norm=batch_norm)
    
    n_channels_classifier = (sum(cfg['branch'][cfg['branch'] != 'M'].astype(np.int)) + n_channels)*n_branches
    classifier = make_layers(cfg['classifier'], n_channels=n_channels_classifier, batch_norm=batch_norm, kernel_size=1)
    
    upsample_size = im_size
    
    # create network
    net = Hypercolumn(branches=branches, classifier=classifier, upsample_size=upsample_size) 
    
    print(net)
    return net
