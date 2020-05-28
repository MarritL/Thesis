#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:08:21 2020

@author: cordolo
"""
import numpy as np
import os
import torch

from train import get_network, determine_branches
from extract_features import  calculate_distancemap, calculate_magnitudemap, calculate_changemap, extract_features_only_feat
from train import determine_branches, get_dataset
from plots import normalize2plot

def detect_changes(model_settings, directories, dataset_settings, network_settings, train_settings, threshold_methods=['triangle']):
    
    save_networkname = model_settings['filename'].split('/')[-1]
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname)):
        os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'cva')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'cva'))
    
    # get_network
    if model_settings.loc['networkname'].endswith('finetune'): 
        model_settings.loc['networkname'] = model_settings.loc['networkname'].split('_')[0]
        model_settings.loc['network'] = model_settings.loc['networkname'].split('_')[0]
    n_branches = 2
    net = get_network(model_settings, gpu=train_settings['gpu']) 
    classifier = model_settings['cfg_classifier'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    conv_classifier = True if classifier[0] == 'C' else False
    
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

        if isinstance(network_settings['extract_features'], list): 
            features = extract_features_only_feat(net, network_settings['extract_features'], [im_a, im_b])
        else:
            # prepare for network
            im_a = torch.as_tensor(np.expand_dims(np.moveaxis(im_a,-1,0), axis=0))
            im_b = torch.as_tensor(np.expand_dims(np.moveaxis(im_b,-1,0), axis=0))
            data = [im_a.float(), im_b.float()]
           
            # get features            
            features = net(
                data, 
                n_branches=n_branches, 
                extract_features=network_settings['extract_features'],
                conv_classifier=conv_classifier, 
                use_softmax=True)       
            
            features = features.squeeze().detach().numpy()
        
        # calculate the distancemap
        if conv_classifier == True:
            changemap = np.argmax(features, axis=0)
            #changemap = features[1] > 0.9
            # calculate change detection accuracy
            cm_pos = changemap > 0    
            tp[method][str(idx)] = np.logical_and(cm_pos, gt_pos).sum()
            tn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), np.logical_not(gt_pos)).sum()
            fp[method][str(idx)] = np.logical_and(cm_pos, np.logical_not(gt_pos)).sum()
            fn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), gt_pos).sum()
        else:
            if network_settings['extract_features'] == 'joint':
                distmap = calculate_magnitudemap(features)
            elif network_settings['extract_features'] == 'diff':
                distmap = calculate_magnitudemap(features)
            elif isinstance(network_settings['extract_features'], list): 
                distmap = calculate_distancemap(features[0], features[1])
            else:
                 raise Exception('distance map calculation not implemented for these settings of extract_features')              
            np.save(os.path.join(directories['results_dir_cd'], save_networkname,'cva','distmap_'+str(idx)+'.npy'), distmap)
            
            for method in threshold_methods:
                changemap, threshold = calculate_changemap(distmap, method=method, plot=True)
                np.save(os.path.join(directories['results_dir_cd'], save_networkname,'cva',
                                     'changemap_threshold_'+method+'_'+str(threshold)+'_'+str(idx)+'.npy'), changemap)
                # calculate change detection accuracy
                cm_pos = changemap > 0    
                tp[method][str(idx)] = np.logical_and(cm_pos, gt_pos).sum()
                tn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), np.logical_not(gt_pos)).sum()
                fp[method][str(idx)] = np.logical_and(cm_pos, np.logical_not(gt_pos)).sum()
                fn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), gt_pos).sum()

    return (tp, tn, fp, fn)
            
def detect_changes_no_gt(model_settings, directories, dataset_settings, network_settings, train_settings, threshold_methods=['triangle']):
    
    save_networkname = model_settings['filename'].split('/')[-1]
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname)):
        os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'cva')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'cva'))
    # get_network
    if model_settings.loc['networkname'].endswith('finetune'): 
        model_settings.loc['networkname'] = model_settings.loc['networkname'].split('_')[0]
        model_settings.loc['network'] = model_settings.loc['networkname'].split('_')[0]
    n_branches = 2
    net = get_network(model_settings, gpu=train_settings['gpu']) 
    classifier = model_settings['cfg_classifier'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    conv_classifier = True if classifier[0] == 'C' else False
                
    for idx in dataset_settings['indices_eval']:
        
        im_a = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_a.npy')))
        im_b = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_b.npy')))

        if isinstance(network_settings['extract_features'], list): 
            features = extract_features_only_feat(net, network_settings['extract_features'], [im_a, im_b])
        else:
            # prepare for network
            im_a = torch.as_tensor(np.expand_dims(np.moveaxis(im_a,-1,0), axis=0))
            im_b = torch.as_tensor(np.expand_dims(np.moveaxis(im_b,-1,0), axis=0))
            data = [im_a.float(), im_b.float()]
           
            # get features          
            features = net(
                data, 
                n_branches=n_branches, 
                extract_features=network_settings['extract_features'],
                conv_classifier=conv_classifier, 
                use_softmax=True)       
            
            features = features.squeeze().detach().numpy()
        
        # calculate the distancemap
        if conv_classifier == True:
            changemap = np.argmax(features, axis=0)
            #changemap = features[1] > 0.9
            # calculate change detection accuracy
            cm_pos = changemap > 0    
        else:
            if network_settings['extract_features'] == 'joint':
                distmap = calculate_magnitudemap(features)
            elif network_settings['extract_features'] == 'diff':
                distmap = calculate_magnitudemap(features)
            elif isinstance(network_settings['extract_features'], list): 
                distmap = calculate_distancemap(features[0], features[1])
            else:
                 raise Exception('distance map calculation not implemented for these settings of extract_features')
            np.save(os.path.join(directories['results_dir_cd'], save_networkname,'cva','distmap_'+str(idx)+'.npy'), distmap)
    
            for method in threshold_methods:
                changemap, threshold = calculate_changemap(distmap, method=method, plot=True)
                np.save(os.path.join(directories['results_dir_cd'], save_networkname,'cva',
                                     'changemap_threshold_'+method+'_'+str(threshold)+'_'+str(idx)+'.npy'), changemap)
            

    return 
        
        