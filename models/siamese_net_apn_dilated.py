#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:58:58 2020

@author: M. Leenstra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(0)

__all__ = ['siamese_net_apn_dilated']
        
class SiameseDilatedNetAPN(nn.Module):
    def __init__(self, branches):
        super(SiameseDilatedNetAPN, self).__init__()
                
        self.branches = branches    

    def forward(self, data, n_branches, avg_pool=False , **kwargs):
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
        
        return res
             
    
def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = int(n_channels)    
    # iterate over layers and add to sequential
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'CU':
            layers += [nn.ConvTranspose2d(in_channels, v, kernel_size=2, stride=2)]
        elif v == 'BU':
            layers += [nn.Upsample(scale_factor=8, mode='bilinear')]        # TODO: scale factor hard-coded  
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=i*2+3, padding=(i+1)**2, dilation=i+1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def siamese_net_apn_dilated(cfg, n_channels=13,n_classes=2, batch_norm=False):
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

    # create network
    net = SiameseDilatedNetAPN(branches) 
    
    print(net)
    return net
    