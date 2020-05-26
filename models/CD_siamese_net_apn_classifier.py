#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:03:09 2020

@author: cordolo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.siamese_net import make_classifier, make_conv_classifier
from models.siamese_net_apn import make_layers

__all__ = ['siamese_net_apn_classifier']

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CDSiameseNetAPNClassifier(nn.Module):   

    def __init__(self, branches, layer_branches, classifier):
        super(CDSiameseNetAPNClassifier, self).__init__()
        
        if layer_branches > 0:
            self.branches = branches
        else:
            self.branches = Identity()
        
        self.classifier = classifier
        

    def forward(self, data, n_branches, avg_pool=False , conv_classifier=False, 
            use_softmax=False,**kwargs):
        """
        forward pass through network
    
        Parameters
        ----------
        data : torch array
            DESCRIPTION.
        n_branches : int
            number of branches in the network (e.g. siamese has 2 branches)
        extract_features : list, optional
            list with the number of the layer in the branch to extract features from. 
            The default is None (i.e. for training phase)
    
        Returns
        -------
        x : torch array
            output of the network for each image. Number of channels is number of classes
            used to train the network
        features : torch array
            specified featuremaps from the branch, upsampled to original patchsize
            used for feature extraction
        """
        res = list()
        for i in range(n_branches): # Siamese/triplet nets; sharing weights
            x = data[i]
            if avg_pool:
                x = F.adaptive_avg_pool2d(x, (1,1))
            res.append(self.branches(x))
        
        # concatenate the output of difference of branches
        x = torch.abs(res[1] - res[0])
        if n_branches == 3:
            x = torch.cat(x, torch.abs(res[2] - res[1]), 1)
        
        x = nn.functional.adaptive_avg_pool2d(x, (data[0].shape[2], data[0].shape[3]))
        if not conv_classifier:
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.classifier(x)
            if use_softmax: # is True during inference
                x = nn.functional.softmax(x, dim=1)
            else:
                x = nn.functional.log_softmax(x, dim=1)

        return x

def siamese_net_apn_classifier(cfg, n_channels=13,n_classes=2, patch_size=96, batch_norm=True, n_branches=2):
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
    # determine number of max-pool layers
    n_mpool = np.sum(cfg['branch'] == 'M') + np.sum(cfg['top'] == 'M')   
    
    # determine input sizes for input classification layers
    patch_size_lin = patch_size
    for i in range(n_mpool):
        patch_size_lin = int(patch_size_lin/2) 
    
    # create layers
    layer_branches = len(cfg['branch'])
    if layer_branches == 0:
        branches = None
    else:
        branches = make_layers(cfg['branch'],n_channels,batch_norm=batch_norm)
        
    
    if cfg['classifier'][0] != 'C':       
        n_channels_lin = int(cfg['branch'][cfg['branch'] != 'M'][-1])*n_branches
        n_channels_classifier = n_channels_lin * patch_size_lin * patch_size_lin  
        classifier = make_classifier(cfg['classifier'], n_channels_classifier)
    else:
        if cfg['branch'] is not None:
            in_channels = int(cfg['branch'][cfg['branch'] != 'M'][-1])
        else:
            in_channels = n_channels
        classifier = make_conv_classifier(cfg['classifier'][1:],in_channels,batch_norm=batch_norm)
        
    # create network
    net = CDSiameseNetAPNClassifier(branches, layer_branches, classifier) 
    
    print(net)
    return net