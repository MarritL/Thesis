#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:26:01 2020

@author: M. Leenstra
"""

import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)

__all__ = ['logistic_regression']

        
class LogisticRegression(nn.Module):
    def __init__(self, n_channels_in, n_classes):
        super(LogisticRegression, self).__init__()
                
        self.linear = torch.nn.Linear(n_channels_in, n_classes)

    def forward(self, data, n_branches, extract_features=None, **kwargs):
        """
        forward pass for logistic regression 

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
        """
            
        # concatenate the output of difference of branches
        x = torch.abs(data[1] - data[0])
        if n_branches == 3:
            x = torch.cat(x, torch.abs(data[2] - data[1]), 1)
        
        if extract_features == 'diff':
            return x
        # joint layers
        x = torch.flatten(x, 1)
        x = self.linear(x)
        if extract_features == 'joint': 
            return x
        return x

             

def logistic_regression(n_channels=13, n_classes=2, patch_size=96):
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
    
    n_channels_in = n_channels * patch_size * patch_size
    net = LogisticRegression(n_channels_in, n_classes)
    
    print(net)
    return net
    
