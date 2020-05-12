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
from torch.nn import functional as F


from models import siamese_net, hypercolumn_net, siamese_unet_diff, siamese_net_apn, \
    siamese_unet, triplet_unet, siamese_net_dilated, siamese_net_apn_dilated, siamese_net_concat

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
        elif net == 'siamese_concat':
            net = siamese_net_concat.__dict__['siamese_net_concat'](
                cfg=cfg, 
                n_channels=n_channels,                                                       
                n_classes=n_classes, 
                patch_size=patch_size,
                batch_norm=batch_norm,
                n_branches=n_branches) 
        elif net == 'siamese_dilated':
            net = siamese_net_dilated.__dict__['siamese_net_dilated'](
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
        elif net == 'siamese_unet_diff':
            net = siamese_unet_diff.__dict__['siamese_unet_diff'](
                n_channels=n_channels,
                n_classes=n_classes)
        elif net == 'siamese_unet':
            net = siamese_unet.__dict__['siamese_unet'](
                n_channels=n_channels,
                n_classes=n_classes,
                patch_size = patch_size)
        elif net == 'triplet_unet':
            net = triplet_unet.__dict__['triplet_unet'](
                n_channels=n_channels,
                n_classes=n_classes,
                patch_size = patch_size)
        elif net == 'triplet_apn':
            net = siamese_net_apn.__dict__['siamese_net_apn'](
                cfg=cfg, 
                n_channels=n_channels,
                n_classes=n_classes, 
                batch_norm=batch_norm)
        elif net == 'triplet_apn_dilated':
            net = siamese_net_apn_dilated.__dict__['siamese_net_apn_dilated'](
                cfg=cfg, 
                n_channels=n_channels,
                n_classes=n_classes, 
                batch_norm=batch_norm)
        else:
            raise Exception('Architecture undefined!\n \
                        Choose one of: "siamese", "hypercolumn", \
                            "siamese_unet_diff", "triplet_apn", "siamese_unet",\
                            "siamese_dilated", "siamese_apn_dilated", "triplet_unet",\
                            "siamese_concat"')

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
                    'accuracy_onehot':accuracy_onehot,
                    'accuracy_fake':accuracy_fake,
                    'f1':F1}
    
    if lossfunction == 'cross_entropy':
        one_hot = False
        loss_func = nn.CrossEntropyLoss() 
        acc_func = acc_functions['accuracy']
    elif lossfunction == 'bce_sigmoid':
        pos_weight = torch.tensor(pos_weight)
        one_hot = True
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        acc_func = acc_functions['accuracy_onehot']
    elif lossfunction == 'nll':
        one_hot = True
        loss_func = nn.NLLLoss()
        acc_func = acc_functions['accuracy_onehot']
    elif lossfunction == 'nll_finetune':
        one_hot = False
        #pos_weight = torch.tensor(pos_weight.clone().detach(), dtype=torch.float)
        pos_weight = torch.tensor(pos_weight)
        loss_func = nn.NLLLoss(pos_weight)
        acc_func = acc_functions['f1']
    elif lossfunction == 'l1+triplet':
        one_hot = False
        loss_func = CombinedLossL1()
        acc_func = acc_functions['accuracy_fake']
    elif lossfunction == 'l2+triplet':
        one_hot = False
        loss_func = CombinedLossL2()
        acc_func = acc_functions['accuracy_fake']
    elif lossfunction == 'triplet':
        one_hot = False
        loss_func = nn.TripletMarginLoss()
        acc_func = acc_functions['accuracy_fake']
    elif lossfunction == 'l1+triplet+bce':
        one_hot = True
        loss_func = CombinedL1TripletBCE()
        acc_func = acc_functions['accuracy_onehot']
    elif lossfunction == 'mse':
        one_hot = False
        loss_func = nn.MSELoss()
        acc_func = acc_functions['accuracy_fake']
    elif lossfunction == 'bce_sigmoid+l1reg':
        pos_weight = torch.tensor(pos_weight)
        one_hot = True
        loss_func = BCEl1RegLoss(pos_weight=pos_weight)
        acc_func = acc_functions['accuracy_onehot']      
    else:
        raise Exception('loss function not implemented! \n \
                        Choose one of: "cross_entropy", "bce_sigmoid" \
                            "l1+triplet", "nll", "nll_finetune" \
                            "l1+triplet+bce", "mse", "bce_sigmoid+l1reg"')
        
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


def accuracy(outputs, labels, im_size=(1,1)):
    val, preds = torch.max(outputs, dim=1)
    acc_sum = torch.sum(preds == labels)
    acc = acc_sum.float() / (len(labels) + 1e-10) / (im_size[0]*im_size[1])
    return acc

def accuracy_onehot(outputs, labels, im_size=(1,1)):
    _, preds = torch.max(outputs, dim=1)
    _, labs = torch.max(labels, dim=1)
    acc_sum = torch.sum(preds == labs)
    acc = acc_sum.float() / (len(labels) + 1e-10) / (im_size[0]*im_size[1])
    return acc

def accuracy_fake(outputs, labels, im_size=(1,1)):
    return torch.max(labels).float()

def F1(outputs, labels, im_size=(1,1)):
    val, preds = torch.max(outputs, dim=1)
    tp = torch.sum((preds == 1) & (preds == labels))
    fp = torch.sum((preds == 1) & (preds != labels))
    tn = torch.sum((preds == 0) & (preds == labels))
    fn = torch.sum((preds == 0) & (preds != labels))
    precision = tp.float() / (tp.float()+fp.float())
    recall = tp.float() / (tp.float()+fn.float())
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = 'mean'
    
    def forward(self, inputs, targets):
        pos_dist = inputs[0]
        neg_dist = inputs[1]
        losses = F.relu(self.margin + pos_dist - neg_dist)
        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise Exception('reduction undefined! \n \
                            Choose one of: "mean", "sum", "none"!')
    
# =============================================================================
#     def forward(self, pos_dist, neg_dist, reduction='mean'):
#         losses = F.relu(self.margin + pos_dist - neg_dist)
#         if reduction == 'none':
#             return losses
#         elif reduction == 'mean':
#             return losses.mean()
#         elif reduction == 'sum':
#             return losses.sum()
#         else:
#             raise Exception('reduction undefined! \n \
#                             Choose one of: "mean", "sum", "none"!')
# =============================================================================

# Here L1 with power and sum. 
# =============================================================================
# class CombinedLoss(nn.Module):
#     def __init__(self, margin = 1.0, weight=1):
#         super(CombinedLoss, self).__init__()
#         self.weight = weight
#         self.l1 = nn.L1Loss(reduction='mean')
#         self.ltriplet = nn.TripletMarginLoss()
#         
#     def forward(self, inputs, targets):
#         anchor = inputs[0]
#         positives = inputs[1]
#         negatives = inputs[2]
#         
#         loss1 = self.l1(positives.pow(2).sum(1), anchor.pow(2).sum(1))
#         loss2 = self.ltriplet(anchor, positives, negatives)
#         
#         loss = loss1 + self.weight * loss2
#         
#         return loss, loss1, loss2
# =============================================================================
    
# Here L1 without power and sum    
class CombinedLossL1(nn.Module):
    def __init__(self, margin = 1.0, weight=1):
        super(CombinedLossL1, self).__init__()
        self.weight = weight
        self.l1 = nn.L1Loss(reduction='mean')
        self.ltriplet = nn.TripletMarginLoss()
        
    def forward(self, inputs, targets):
        anchor = inputs[0]
        positives = inputs[1]
        negatives = inputs[2]
        
        loss1 = self.l1(positives, anchor)
        loss2 = self.ltriplet(anchor, positives, negatives)
        
        loss = loss1 + self.weight * loss2
        
        return loss, loss1, loss2

class CombinedLossL2(nn.Module):
    def __init__(self, margin = 1.0, weight=1):
        super(CombinedLossL1, self).__init__()
        self.weight = weight
        self.l2 = nn.MSELoss(reduction='mean')
        self.ltriplet = nn.TripletMarginLoss()
        
    def forward(self, inputs, targets):
        anchor = inputs[0]
        positives = inputs[1]
        negatives = inputs[2]
        
        loss1 = self.l2(positives, anchor)
        loss2 = self.ltriplet(anchor, positives, negatives)
        
        loss = loss1 + self.weight * loss2
        
        return loss, loss1, loss2
    
class CombinedL1TripletBCE(nn.Module):
    def __init__(self, margin = 1.0, weight1=1, weight2=1, weight3=1):
        super(CombinedL1TripletBCE, self).__init__()
        self.weight1 = 1
        self.weight2 = 1
        self.weight3 = 1
        self.l1 = nn.L1Loss(reduction='mean')
        self.ltriplet = nn.TripletMarginLoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets, target_bottle):
        input_bottle = inputs[0]
        anchor = inputs[1][0]
        positives = torch.zeros_like(inputs[1][1])
        negatives = torch.zeros_like(inputs[1][2])
        for i, label in enumerate(target_bottle):
            if torch.argmax(label) == 0:
                positives[i] = inputs[1][1][i]
                negatives[i] = inputs[1][2][i]
            elif torch.argmax(label) == 1:
                positives[i] = inputs[1][2][i]
                negatives[i] = inputs[1][1][i]
        
        loss1 = self.l1(positives, anchor)
        loss2 = self.ltriplet(anchor, positives, negatives)
        loss3 = self.bce(input_bottle,target_bottle)
        
        loss = self.weight1 * loss1 + self.weight2 * loss2 + self.weight3 * loss3
        
        return loss, loss1, loss2, loss3

class BCEl1RegLoss(nn.Module):
    def __init__(self, pos_weight):
        super(BCEl1RegLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, inputs, targets, params):
        bce = self.bce(inputs, targets)
        l1_reg = 0
        for param in params:
            l1_reg += self.l1(param,torch.zeros_like(param))
            
        loss = bce + 1e-4*l1_reg
        return loss
    
# =============================================================================
# class CombinedLoss(nn.Module):
#     def __init__(self, margin = 1.0, weight=1):
#         super(CombinedLoss, self).__init__()
#         self.weight = weight
#         self.L1 = nn.L1Loss(reduction='mean') 
#         self.Ltriplet = TripletLoss(margin)
#         
#     def forward(self, inputs, targets):
#         pos_dist = inputs[0]
#         neg_dist = inputs[1]
#         
#         # average over image
#         #loss1 = self.L1(pos_dist.mean((1,2)), targets) 
#         #loss2 = self.Ltriplet(pos_dist.mean((1,2)), neg_dist.mean((1,2))) 
#         
#         # pixel-wise
#         loss1 = self.L1(pos_dist, targets)      
#         loss2 = self.Ltriplet(pos_dist, neg_dist)
#         #print("L1: {}, triplet: {}".format(loss1, loss2))
#         
#         loss = loss1 + self.weight * loss2
#         
#         return loss, loss1, loss2
# =============================================================================

# =============================================================================
# fig, ax = plt.subplots()
# im2 = ax.imshow(F.relu(1 + pos_dist-neg_dist)[0].detach())
# fig.colorbar(im2, ax=ax)
# =============================================================================













