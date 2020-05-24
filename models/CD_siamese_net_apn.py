#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:55:01 2020

@author: M. Leenstra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.siamese_net import make_classifier, make_conv_classifier

__all__ = ['cd_siamese_net_apn']

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CDSiameseNetAPN(nn.Module):   

    def __init__(self, network, layers_branches, classifier):
        super(CDSiameseNetAPN, self).__init__()
        
        if layers_branches > 0:
            self.branches = nn.Sequential(
                *list(network.branches)[:layers_branches])
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
        
 
    
def cd_siamese_net_apn(network, layers_branches, cfg_classifier, 
                       batch_norm=True, patch_size=96):
    
    """
    Create network

    Parameters
    ----------


    Returns
    -------
    net : nn.Module
        deep learning network.

    """
    
    last_conv = layers_branches-3
    in_channels = network.branches[last_conv].out_channels
    
    if cfg_classifier[0] != 'C':
        n_channels_lin = int(in_channels)
        n_channels_classifier = n_channels_lin * patch_size * patch_size  
        classifier = make_classifier(cfg_classifier, n_channels_classifier)
    else:
       classifier = make_conv_classifier(cfg_classifier[1:],in_channels,batch_norm=batch_norm)
        
    # create network
    net = CDSiameseNetAPN(network, layers_branches, classifier) 
    
    print(net)
    return net


