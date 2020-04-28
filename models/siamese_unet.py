#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:04:58 2020

@author: cordolo
"""

import torch.nn as nn
import torch
from torch.nn.modules.padding import ReplicationPad2d

__all__ = ['siamese_unet']


class SiameseUNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, patch_size):
        super(SiameseUNet, self).__init__()
        
        self.conv1 = nn.Conv2d(n_channels, 32, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(32, 64,kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(64, 128,kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True) 
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(128, 128,kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True) 
        
        self.convT5 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(256,64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.convT6 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2,padding=0)
        
        self.conv6 = nn.Conv2d(128,32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.convT7 = nn.ConvTranspose2d(32, 32,kernel_size=2, stride=2,padding=0)
        
        self.conv7 = nn.Conv2d(64,32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(32*patch_size*patch_size, 128)
        self.relu8 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, 64)
        self.relu9 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(64, n_classes)


    def forward(self, data, n_branches, extract_features=None, **kwargs):      
        layer1 = list()
        layer2 = list()
        layer3 = list()
        res = list()
        for i in range(n_branches): # Siamese/triplet nets; sharing weights
            x = data[i]
            x1 = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool1(x1)
            x2 = self.relu2(self.bn2(self.conv2(x)))
            x = self.maxpool2(x2)
            x3 = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool3(x3)
            
            layer1.append(x1)
            layer2.append(x2)
            layer3.append(x3)
            res.append(x)
        
        x = torch.abs(res[1] - res[0])
        if n_branches == 3:
            x = torch.cat(x, torch.abs(res[2] - res[1]), 1)

        x = self.relu4(self.bn4(self.conv4(x)))
        if extract_features == 'joint':
            return(x)
        
        x = self.convT5(x)
        pad = ReplicationPad2d((0, layer3[0].shape[3] - x.shape[3], 0, layer3[0].shape[2] - x.shape[2]))
        x = torch.cat([pad(x), torch.abs(layer3[1]-layer3[0])], dim=1)
        x = self.relu5(self.bn5(self.conv5(x)))
        
        x = self.convT6(x)
        pad = ReplicationPad2d((0, layer2[0].shape[3] - x.shape[3], 0, layer2[0].shape[2] - x.shape[2]))
        x = torch.cat([pad(x), torch.abs(layer2[1]-layer2[0])], dim=1)
        x = self.relu6(self.bn6(self.conv6(x)))
        
        x = self.convT7(x)
        pad = ReplicationPad2d((0, layer1[0].shape[3] - x.shape[3], 0, layer1[0].shape[2] - x.shape[2]))
        x = torch.cat([pad(x), torch.abs(layer1[1]-layer1[0])], dim=1)
        x = self.relu7(self.bn7(self.conv7(x)))

        if extract_features == 'last':
            return(x)
        
        x = torch.flatten(x, 1)
        x = self.relu8(self.linear1(x))
        x = self.relu9(self.linear2(x))
        x = self.linear3(x)
        
        return x
    
def siamese_unet(n_channels=13, n_classes=2, patch_size=96, **kwargs):
    net = SiameseUNet(n_channels=n_channels, n_classes=n_classes, patch_size=patch_size,**kwargs)
    print(net)
    return net