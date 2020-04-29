#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:21:23 2020

@author: M. Leenstra
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.siamese_net_apn import SiameseNetAPN

class siameseNetAPNFinetune(nn.Module):
    def __init__(self, pretrained_net, last_in, n_classes):
        super(siameseNetAPNFinetune, self).__init__()
        if isinstance(pretrained_net, SiameseNetAPN): 
            self.branches = nn.Sequential(
                # do not use fc layers
                *list(pretrained_net.branches.children())
            )
        else:
            self.branches = pretrained_net
        self.lastconv = nn.Conv2d(last_in, n_classes, kernel_size=1)
        self.sm = nn.LogSoftmax(dim=1)
        
    def forward(self, data, n_branches=2, avg_pool=False, **kwargs):
        res = list()
        #avg_pool = False
        #n_branches = 2
        for i in range(n_branches): # Siamese/triplet nets; sharing weights
            x = data[i]
            if avg_pool:
                x = F.adaptive_avg_pool2d(x, (1,1))
            res.append(self.branches(x))
    
        x = res[1].pow(2) - res[0].pow(2)
        x = self.lastconv(x)
        x = self.sm(x)
        
        return x

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
      
def siamese_net_apn_finetune(cfg_branch, batch_norm, n_channels, n_classes):
    
   
    branch = cfg_branch.split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    #branch = branch.replace(" ", "")
    branch = branch.split(" ")
    branch = np.array(branch, dtype='object')
    pretrained_net = make_layers(branch, n_channels, batch_norm=batch_norm)
    
    last_in = int(branch[-1])
    
    # create network
    net = siameseNetAPNFinetune(pretrained_net, last_in, n_classes) 
    
    print(net)
    return net