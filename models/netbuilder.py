#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:36:09 2020

@author: M. Leenstra
@source: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
"""
import torch
from torch import nn
from torch import optim


from models import siamese_net, hypercolumn_net

class NetBuilder:
    # custom weights initialization
    @staticmethod
    def init_weight(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.0001)
            
    @staticmethod
    def printname(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            print(classname, '-', m.weight)


    @staticmethod
    def build_network(net, cfg, weights='', n_channels=13,n_classes=8, 
                      patch_size=96, batch_norm=False, n_branches=2,
                      im_size=(96,96)):
        
        if net == 'siamese':
            net = siamese_net.__dict__['siamese_net'](
                cfg=cfg, 
                n_channels=n_channels,                                                       
                n_classes=n_classes, 
                patch_size=patch_size,
                batch_norm=batch_norm,
                n_branches=n_branches) 
        elif net == 'hypercolumn':
            net = hypercolumn_net.__dict__['hypercolumn_net'](
                cfg=cfg,
                n_channels=n_channels,
                n_classes=n_classes,
                im_size=im_size,
                batch_norm=batch_norm, 
                n_branches=n_branches)
        else:
            raise Exception('Architecture undefined!\n \
                        Choose one of: "siamese", "hypercolumn"')

        # initiate weighs 
        if len(weights) > 0:
            print('Loading weights for network...')
            net.load_state_dict(
                torch.load(weights), strict=False)
        else:
            net.apply(NetBuilder.init_weight)
        
        return net


def create_loss_function(lossfunction, pos_weight=1):
    acc_functions = {'accuracy': accuracy,
                    'accuracy_onehot':accuracy_onehot}
    
    if lossfunction == 'cross_entropy':
        one_hot = False
        loss_func = nn.CrossEntropyLoss() 
        acc_func = acc_functions['accuracy']
    elif lossfunction == 'bce_sigmoid':
        pos_weight = torch.tensor(pos_weight)
        one_hot = True
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        acc_func = acc_functions['accuracy_onehot']
    else:
        raise Exception('loss function not implemented! \n \
                        Choose one of: "cross_entropy", "bce_sigmoid"')
        
    return loss_func, acc_func, one_hot

def create_optimizer(optimizer_string, params, lr, weight_decay=0):
       
    if optimizer_string == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_string == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-08, 
                                  weight_decay=weight_decay, momentum=0, 
                                  centered=False)
    else:
        raise Exception('optimizer not implemented! \n \
                        Choose one of: "adam", "rmsprop"')
        
    return optimizer


def accuracy(outputs, labels):
    val, preds = torch.max(outputs, dim=1)
    acc_sum = torch.sum(preds == labels)
    acc = acc_sum.float() / (len(labels) + 1e-10)
    return acc

def accuracy_onehot(outputs, labels, im_size=(1,1)):
    _, preds = torch.max(outputs, dim=1)
    _, labs = torch.max(labels, dim=1)
    acc_sum = torch.sum(preds == labs)
    acc = acc_sum.float() / (len(labels) + 1e-10) / (im_size[0]*im_size[1])
    return acc

def accuracy_segm(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _, labs = torch.max(labels, dim=1)
    acc_sum = torch.sum(preds == labs)
    acc = acc_sum.float() / (len(labels) + 1e-10) 
    return acc    