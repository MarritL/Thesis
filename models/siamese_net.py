#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:55:01 2020

@author: M. Leenstra
"""
import torch
import torch.nn as nn
import numpy as np

__all__ = ['siamese_net']

class SiameseNet(nn.Module):

    def __init__(self, branches, joint, n_channels_lin, n_classes, n_mpool, 
                 patch_size):
        super(SiameseNet, self).__init__()
        
        # determine input sizes for input classification layers
        patch_size_lin = patch_size
        for i in range(n_mpool):
            patch_size_lin = int(patch_size_lin/2)
        
        self.branches = branches    
        self.joint = joint
                
        self.avgpool = nn.AdaptiveAvgPool2d((patch_size_lin, patch_size_lin))
        self.classifier = nn.Sequential(
            nn.Linear(n_channels_lin * patch_size_lin * patch_size_lin, 256),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(128, n_classes),
        )


    def forward(self, data, n_branches, extract_features=None):
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
                features = [x]
                features.extend([activations[names[i]] for i in extract_features])
            
                return features
                
            # if in training or validation phase forward images through branch   
            else:
                res.append(self.branches(x))
        
        # concatenate the output of both branches
        x = torch.cat(res, 1)
        
        # joint layers
        x = self.joint(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # classification layers
        x = self.classifier(x)
        return x

             
    
def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = int(n_channels)
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

def conv_bn_relu(in_channels, out_channels, stride=1, padding=1, batch_norm=False):
    "3x3 convolution + BN + relu"
    layers = []
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=padding)
    if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers += [conv2d, nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


def siamese_net(cfg, n_channels=13,n_classes=2, patch_size=96, batch_norm=False, n_branches=2):
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
    #
    
    # determine number of max-pool layers
    n_mpool = np.sum(cfg['branch'] == 'M') + np.sum(cfg['top'] == 'M')    

    if cfg['top'] is not None:
        n_channels_lin = int(cfg['top'][cfg['top'] != 'M'][-1])
    else:
        n_channels_lin = int(cfg['branch'][cfg['branch'] != 'M'][-1])*n_branches
    
    # create layers
    branches = make_layers(cfg['branch'],n_channels,batch_norm=batch_norm)
    if cfg['top'] is not None:
        joint = make_layers(cfg['top'],
                            int(cfg['branch'][cfg['branch'] != 'M'][-1])*n_branches,
                            batch_norm=batch_norm)
    else:
        # does nothing because next layer is the same
        joint = nn.AdaptiveAvgPool2d((patch_size, patch_size)) 
    
    # create network
    net = SiameseNet(branches, joint, n_channels_lin, n_classes, n_mpool, patch_size) 
    
    print(net)
    return net
    