#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:08:21 2020

@author: cordolo
"""
import numpy as np
import os
import torch

from extract_features import get_network, calculate_distancemap, calculate_magnitudemap, calculate_changemap
from train import determine_branches, get_dataset
from plots import normalize2plot

def detect_changes(model_settings, directories, dataset_settings, network_settings, train_settings, threshold_methods=['triangle']):
    
    # get_network
    n_branches, pos_weight = determine_branches(network_settings, dataset_settings)
    net = get_network(model_settings) 
    
    tp = dict()
    tn = dict()
    fp = dict()
    fn = dict()

    for method in threshold_methods:       
        tp[method] = dict()
        tn[method] = dict()
        fp[method] = dict()
        fn[method] = dict()
            
    for idx in dataset_settings['indices_eval']:
        
        im_a = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_a.npy')))
        im_b = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_b.npy')))
        gt = np.load(os.path.join(directories['labels_path'], str(idx)+'.npy'))
        gt -= 1
        gt_pos = gt > 0

        # prepare for network
        im_a = torch.as_tensor(np.expand_dims(np.moveaxis(im_a,-1,0), axis=0))
        im_b = torch.as_tensor(np.expand_dims(np.moveaxis(im_b,-1,0), axis=0))
        data = [im_a.float(), im_b.float()]
       
        # get features
        features = net(
            data, 
            n_branches=n_branches, 
            extract_features=network_settings['extract_features'])       
        
        features = features.squeeze().detach().numpy()
        
        # calculate the distancemap
        if network_settings['extract_features'] == 'joint':
            distmap = calculate_magnitudemap(features)
        elif network_settings['extract_features'] == 'diff':
            distmap = calculate_magnitudemap(features)
        else:
             raise Exception('distance map calculation not implemented for these settings of extract_features')
        

        for method in threshold_methods:
            changemap, threshold = calculate_changemap(distmap, method=method, plot=True)
            
            # calculate change detection accuracy
            cm_pos = changemap > 0    
            tp[method][str(idx)] = np.logical_and(cm_pos, gt_pos).sum()
            tn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), np.logical_not(gt_pos)).sum()
            fp[method][str(idx)] = np.logical_and(cm_pos, np.logical_not(gt_pos)).sum()
            fn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), gt_pos).sum()

    return (tp, tn, fp, fn)
            
            
        
        