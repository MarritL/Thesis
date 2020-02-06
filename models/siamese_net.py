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
                 patch_size_lin):
        super(SiameseNet, self).__init__()
        
        self.branches = branches    
        self.joint = joint
        
        self.avgpool = nn.AdaptiveAvgPool2d((patch_size_lin, patch_size_lin))
        self.classifier = nn.Sequential(
            nn.Linear(n_channels_lin * patch_size_lin * patch_size_lin, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, n_classes),
        )


    def forward(self, data):
        res = []
        for i in range(2): # Siamese nets; sharing weights
            x = data[i]
            x = self.branches(x)
            res.append(x)
         
        x = torch.cat([res[0], res[1]], dim=1) 
        
        x = self.joint(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = n_channels
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


def siamese_net(cfg, n_channels=13,n_classes=8, patch_size=96, batch_norm=False):
    #cfg = {'branch': np.array([64, 'M', 128, 'M']), 'top': np.array([256, 'M'])}
    
    # determine number of max-pool layers
    n_mpool = np.sum(cfg['branch'] == 'M') + np.sum(cfg['top'] == 'M')
    
    # determine input sizes for input classification layers
    for i in range(n_mpool):
        patch_size = int(patch_size/2)
    if cfg['top'] is not None: 
        n_channels_lin = int(cfg['top'][cfg['top'] != 'M'][-1])
    else:
        n_channels_lin = int(cfg['branch'][cfg['branch'] != 'M'][-1])*2
    
    # create layers
    branches = make_layers(cfg['branch'],n_channels,batch_norm=batch_norm)
    if cfg['top'] is not None:
        joint = make_layers(cfg['top'],
                            int(cfg['branch'][cfg['branch'] != 'M'][-1])*2,
                            batch_norm=batch_norm)
    else:
        # does nothing because next layer is the same
        joint = nn.AdaptiveAvgPool2d((patch_size, patch_size)) 
    
    # create network
    net = SiameseNet(branches, joint,
                     n_channels_lin, n_classes, n_mpool, patch_size) 
    
    print(net)
    return net
    