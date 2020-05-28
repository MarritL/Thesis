#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:55:01 2020

@author: M. Leenstra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['siamese_net_apn']
        
class SiameseNetAPN(nn.Module):
    def __init__(self, branches):
        super(SiameseNetAPN, self).__init__()
                
        self.branches = branches    

    def forward(self, data, n_branches, avg_pool=False, extract_features=None,**kwargs):
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
        for j in range(n_branches): # Siamese/triplet nets; sharing weights
            x = data[j]
            
            if isinstance(extract_features,list):
                activations = dict()
                names = list()
                for i, l in enumerate(self.branches):
                    names.append('x'+str(i))
                    if i == 0:
                        activations[names[i]] = l(x)
                        if activations[names[i]].shape[2:] != data[j].shape[2:]:
                            activations[names[i]] = nn.functional.interpolate(
                                 activations[names[i]], size=data[j].shape[2:], 
                                 mode='bilinear', align_corners=True)
                    else:
                        activations[names[i]] = l(activations[names[i-1]])
                        if activations[names[i]].shape[2:] != data[j].shape[2:]:
                            activations[names[i]] = nn.functional.interpolate(
                                 activations[names[i]], size=data[j].shape[2:], 
                                 mode='bilinear', align_corners=True)
                            
                # return a list of features
                #features = [x]
                features=list()
                features.extend([activations[names[i]] for i in extract_features])
            
                return features
            else:
                if avg_pool:
                    x = F.adaptive_avg_pool2d(x, (1,1))
                res.append(self.branches(x))
        
        return res
# =============================================================================
#         anchor = res[0]
#         pos = res[1]
#         pos_dist = (anchor - pos).pow(2).sum(1)#.sqrt()
#         
#         if n_branches == 3:
#             neg = res[2]
#             neg_dist = (anchor - neg).pow(2).sum(1)#.sqrt()
#         
#         return [pos_dist, neg_dist] if n_branches == 3 else [pos_dist]
# =============================================================================
 
    
def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = int(n_channels)    
    # iterate over layers and add to sequential
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



def siamese_net_apn(cfg, n_channels=13,n_classes=2, batch_norm=False):
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
    net = SiameseNetAPN(branches) 
    
    print(net)
    return net



