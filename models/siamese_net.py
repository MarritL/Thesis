#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:55:01 2020

@author: M. Leenstra
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['siamese_net']

class SiameseNet(nn.Module):
   
    def __init__(self, n_channels, n_classes):
      super(SiameseNet, self).__init__()
      
      self.conv1 = nn.Conv2d(n_channels, 64, 3)
      self.pool1 = nn.MaxPool2d(2,2)
      
      self.conv2 = nn.Conv2d(128, 128, 3)
      self.linear1 = nn.Linear(128*45*45, 512)
      self.linear2 = nn.Linear(512, n_classes)

      
    def forward(self, data):
      res = []
      for i in range(2): # Siamese nets; sharing weights
         x = data[i]
         x = F.relu(self.conv1(x))
         x = self.pool1(x)
         
         res.append(x)
         
      x = torch.cat([res[0], res[1]], dim=1)
      x = F.relu(self.conv2(x))
      
      x = x.view(x.shape[0], -1)
      x = self.linear1(x)
      x = self.linear2(x)

      return x

def siamese_net(n_channels = 13, n_classes = 8):    
    net = SiameseNet(n_channels=n_channels, n_classes=n_classes)
    return net