#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:36:52 2020

@author: M. Leenstra
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:04:58 2020

@author: cordolo
"""

import torch.nn as nn
import torch
from torch.nn.modules.padding import ReplicationPad2d

__all__ = ['triplet_unet']


class TripletUNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, patch_size):
        super(TripletUNet, self).__init__()
        
        # encoder
        self.conv1a = nn.Conv2d(n_channels, 16, stride=1, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(16)
        self.relu1a = nn.ReLU(inplace=True)
# =============================================================================
#         self.conv1b = nn.Conv2d(16, 16, stride=1, kernel_size=3, padding=1)
#         self.bn1b = nn.BatchNorm2d(16)
#         self.relu1b = nn.ReLU(inplace=True)
# =============================================================================
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2a = nn.Conv2d(16, 32,kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.relu2a = nn.ReLU(inplace=True)
# =============================================================================
#         self.conv2b = nn.Conv2d(32, 32,kernel_size=3, padding=1)
#         self.bn2b = nn.BatchNorm2d(32)
#         self.relu2b = nn.ReLU(inplace=True)
# =============================================================================
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3a = nn.Conv2d(32, 64,kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(64)
        self.relu3a = nn.ReLU(inplace=True) 
        self.conv3b = nn.Conv2d(64, 64,kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(64)
        self.relu3b = nn.ReLU(inplace=True) 
# =============================================================================
#         self.conv3c = nn.Conv2d(64, 64,kernel_size=3, padding=1)
#         self.bn3c = nn.BatchNorm2d(64)
#         self.relu3c = nn.ReLU(inplace=True) 
# =============================================================================
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4a = nn.Conv2d(64, 128,kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(128)
        self.relu4a = nn.ReLU(inplace=True) 
        self.conv4b = nn.Conv2d(128, 128,kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(128)
        self.relu4b = nn.ReLU(inplace=True) 
# =============================================================================
#         self.conv4c = nn.Conv2d(128, 128,kernel_size=3, padding=1)
#         self.bn4c = nn.BatchNorm2d(128)
#         self.relu4c = nn.ReLU(inplace=True) 
# =============================================================================
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv_bottle = nn.Conv2d(128, 128,kernel_size=3, padding=1)
        self.bn_bottle = nn.BatchNorm2d(128)
        self.relu_bottle = nn.ReLU(inplace=True) 
# =============================================================================
#         self.conv_joint = nn.Conv2d(128, 128,kernel_size=3, padding=1)
#         self.bn_joint = nn.BatchNorm2d(128)
#         self.relu_joint = nn.ReLU(inplace=True) 
# =============================================================================
        
        # bottleneck classifier
        self.linear1 = nn.Linear(int(128*patch_size/2**4*patch_size/2**4), 128)
        self.relu8 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(128, 64)
        self.relu9 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(64, n_classes)
        
        
        # decoder
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        
        self.conv4cd = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn4cd = nn.BatchNorm2d(128)
        self.relu4cd = nn.ReLU(inplace=True)
# =============================================================================
#         self.conv4bd = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
#         self.bn4bd = nn.BatchNorm2d(128)
#         self.relu4bd = nn.ReLU(inplace=True)
# =============================================================================
        self.conv4ad = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn4ad = nn.BatchNorm2d(64)
        self.relu4ad = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv3cd = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn3cd = nn.BatchNorm2d(64)
        self.relu3cd = nn.ReLU(inplace=True)
# =============================================================================
#         self.conv3bd = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
#         self.bn3bd = nn.BatchNorm2d(64)
#         self.relu3bd = nn.ReLU(inplace=True)
# =============================================================================
        self.conv3ad = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn3ad = nn.BatchNorm2d(32)
        self.relu3ad = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv2bd = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn2bd = nn.BatchNorm2d(32)
        self.relu2bd = nn.ReLU(inplace=True)
        self.conv2ad = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn2ad = nn.BatchNorm2d(16)
        self.relu2ad = nn.ReLU(inplace=True)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv1bd = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn1bd = nn.BatchNorm2d(16)
        self.relu1bd = nn.ReLU(inplace=True)
        
        # difference 
        
    def forward(self, data, n_branches, extract_features=None, **kwargs):      

        res = list()
        end = list()
        for i in range(n_branches): # Siamese/triplet nets; sharing weights
            # encoding
            x = data[i]
            x1 = self.relu1a(self.bn1a(self.conv1a(x)))
            #x1b = self.relu1b(self.bn1b(self.conv1b(x1a)))
            x = self.maxpool1(x1)
            x2 = self.relu2a(self.bn2a(self.conv2a(x)))
            #x2b = self.relu2b(self.bn2b(self.conv2b(x2a)))
            x = self.maxpool2(x2)
            x3a = self.relu3a(self.bn3a(self.conv3a(x)))
            x3 = self.relu3b(self.bn3b(self.conv3b(x3a)))
            #x3c = self.relu3c(self.bn3c(self.conv3c(x3b)))
            x = self.maxpool3(x3)
            x4a = self.relu4a(self.bn4a(self.conv4a(x)))
            x4 = self.relu4b(self.bn4b(self.conv4b(x4a)))
            #x4c = self.relu4c(self.bn4c(self.conv4c(x4b)))
            x = self.maxpool4(x4)
            xb = self.relu_bottle(self.bn_bottle(self.conv_bottle(x)))
            
            res.append(xb)
        
            # decoding
            x4d = self.upconv4(xb)
            pad4 = ReplicationPad2d((0, x4.size(3) - x4d.size(3), 0, x4.size(2) - x4d.size(2)))
            x4d = torch.cat((pad4(x4d), x4), 1)
            x4d = self.relu4cd(self.bn4cd(self.conv4cd(x4d)))
            #x4bd = self.relu4bd(self.bn4bd(self.conv4bd(x4cd)))
            x4d = self.relu4ad(self.bn4ad(self.conv4ad(x4d)))
    
            # Stage 3d
            x3d = self.upconv3(x4d)
            pad3 = ReplicationPad2d((0, x3.size(3) - x3d.size(3), 0, x3.size(2) - x3d.size(2)))
            x3d = torch.cat((pad3(x3d), x3), 1)
            x3d = self.relu3cd(self.bn3cd(self.conv3cd(x3d)))
            #x3bd = self.relu3bd(self.bn3bd(self.conv3bd(x3cd)))
            x3d = self.relu3ad(self.bn3ad(self.conv3ad(x3d)))
    
            # Stage 2d
            x2d = self.upconv2(x3d)
            pad2 = ReplicationPad2d((0, x2.size(3) - x2d.size(3), 0, x2.size(2) - x2d.size(2)))
            x2d = torch.cat((pad2(x2d), x2), 1)
            x2d = self.relu2bd(self.bn2bd(self.conv2bd(x2d)))
            x2d = self.relu2ad(self.bn2ad(self.conv2ad(x2d)))
    
            # Stage 1d
            x1d = self.upconv1(x2d)
            pad1 = ReplicationPad2d((0, x1.size(3) - x1d.size(3), 0, x1.size(2) - x1d.size(2)))
            x1d = torch.cat((pad1(x1d), x1), 1)
            x1d = self.relu1bd(self.bn1bd(self.conv1bd(x1d)))   
            
            end.append(x1d)
     
        # bottleneck classifier
        diff = torch.abs(res[1] - res[0])
        if extract_features == 'joint':
            return diff
        
        if extract_features == 'last':
            return end
                      
        bottle = torch.flatten(diff, 1)
        bottle = self.relu8(self.linear1(bottle))
        bottle = self.relu9(self.linear2(bottle))
        bottle = self.linear3(bottle)
                        
        return [bottle, end]


# =============================================================================
#     def forward(self, data, n_branches, extract_features=None, **kwargs):      
#         layer1 = list()
#         layer2 = list()
#         layer3 = list()
#         layer4 = list()
#         res = list()
#         # encoding
#         for i in range(n_branches): # Siamese/triplet nets; sharing weights
#             x = data[i]
#             x1 = self.relu1a(self.bn1a(self.conv1a(x)))
#             #x1b = self.relu1b(self.bn1b(self.conv1b(x1a)))
#             x = self.maxpool1(x1)
#             x2 = self.relu2a(self.bn2a(self.conv2a(x)))
#             #x2b = self.relu2b(self.bn2b(self.conv2b(x2a)))
#             x = self.maxpool2(x2)
#             x3a = self.relu3a(self.bn3a(self.conv3a(x)))
#             x3 = self.relu3b(self.bn3b(self.conv3b(x3a)))
#             #x3c = self.relu3c(self.bn3c(self.conv3c(x3b)))
#             x = self.maxpool3(x3)
#             x4a = self.relu4a(self.bn4a(self.conv4a(x)))
#             x4 = self.relu4b(self.bn4b(self.conv4b(x4a)))
#             #x4c = self.relu4c(self.bn4c(self.conv4c(x4b)))
#             #x = self.maxpool4(x4c)
#             
#             layer1.append(x1)
#             layer2.append(x2)
#             layer3.append(x3)
#             layer4.append(x4)
#             res.append(x4)
#             
#         if extract_features == 'joint':
#             return torch.abs(res[1] - res[0])
#             
#         # decoding
#         #x4d = self.upconv4(res[1])
#         #pad4 = ReplicationPad2d((0, layer4[0].size(3) - x4d.size(3), 0, layer4[0].size(2) - x4d.size(2)))
#         #x4d = torch.cat((pad4(res[2), torch.abs(layer4[0] - layer4[1])), 1)
#         
#         pad4 = ReplicationPad2d((0, layer4[0].size(3) - res[2].size(3), 0, layer4[0].size(2) - res[2].size(2)))
#         x4d = torch.cat((pad4(res[2]), torch.abs(layer4[0] - layer4[1])), 1)
#         x4d = self.relu4cd(self.bn4cd(self.conv4cd(x4d)))
#         #x4bd = self.relu4bd(self.bn4bd(self.conv4bd(x4cd)))
#         x4d = self.relu4ad(self.bn4ad(self.conv4ad(x4d)))
# 
#         # Stage 3d
#         x3d = self.upconv3(x4d)
#         pad3 = ReplicationPad2d((0, layer3[0].size(3) - x3d.size(3), 0, layer3[0].size(2) - x3d.size(2)))
#         x3d = torch.cat((pad3(x3d), torch.abs(layer3[0] - layer3[1])), 1)
#         x3d = self.relu3cd(self.bn3cd(self.conv3cd(x3d)))
#         #x3bd = self.relu3bd(self.bn3bd(self.conv3bd(x3cd)))
#         x3d = self.relu3ad(self.bn3ad(self.conv3ad(x3d)))
# 
#         # Stage 2d
#         x2d = self.upconv2(x3d)
#         pad2 = ReplicationPad2d((0, layer2[0].size(3) - x2d.size(3), 0, layer2[0].size(2) - x2d.size(2)))
#         x2d = torch.cat((pad2(x2d), torch.abs(layer2[0] - layer2[1])), 1)
#         x2d = self.relu2bd(self.bn2bd(self.conv2bd(x2d)))
#         x2d = self.relu2ad(self.bn2ad(self.conv2ad(x2d)))
# 
#         # Stage 1d
#         x1d = self.upconv1(x2d)
#         pad1 = ReplicationPad2d((0, layer1[0].size(3) - x1d.size(3), 0, layer1[0].size(2) - x1d.size(2)))
#         x1d = torch.cat((pad1(x1d), torch.abs(layer1[0] - layer1[1])), 1)
#         x1d = self.relu1bd(self.bn1bd(self.conv1bd(x1d)))    
#         
#         if extract_features == 'last':
#             return(x1d)
#         
#         x = torch.flatten(x1d, 1)
#         x = self.relu8(self.linear1(x))
#         x = self.relu9(self.linear2(x))
#         x = self.linear3(x)
#         
#         return x
# =============================================================================
    
def triplet_unet(n_channels=13, n_classes=2, patch_size=96, **kwargs):
    net = TripletUNet(n_channels=n_channels, n_classes=n_classes, patch_size=patch_size,**kwargs)
    print(net)
    return net

