#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:08:21 2020

@author: M. Leenstra
"""
import numpy as np
import os
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

from train import get_network, determine_branches
from extract_features import  calculate_distancemap, calculate_magnitudemap, calculate_changemap, extract_features_only_feat
from train import determine_branches, get_dataset
from plots import normalize2plot, plot_changemap_colors
from inference import array2raster

def detect_changes(model_settings, directories, dataset_settings, network_settings, 
                   train_settings, threshold_methods=['triangle']):
    """
    Calculate the confusion matrix for specified thresholding methods

    Parameters
    ----------
    model_settings : dict
        Settings used to during training of the model
    directories :  dict
        Directories to load and save data
    dataset_settings : dict
        Settings that specify the used dataset
    network_settings : dict
        Settings that specity the used network
    train_settings : dict
        Settings that specity the training parameters
    threshold_methods : list, optional
        Threshold methods to use. Options: "otsu" | "triangle" | "end-to-end" | "f1" | "AA" 
        The default is ['triangle'].

    Raises
    ------
    Exception
        network_settings['extract_features'] should be one of None | 'joint' | 'diff' | list of layernumbers 

    Returns
    -------
    tp : dict
        True positives of every image in for every thresholding method
    tn : dict
        True negatives of every image in for every thresholding method
    fp : dict
        False positives of every image in for every thresholding method
    fn : dict
        False negatives of every image in for every thresholding method
    th : dict
        Threshold-value applied to get the result
    """
    save_networkname = model_settings['filename'].split('/')[-1]
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname)):
        os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'cva')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'cva'))
    
    # get_network
    if network_settings['extract_features'] != None:
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
    else:
        conv_classifier = False
    
    tp = dict()
    tn = dict()
    fp = dict()
    fn = dict()
    th = dict()

    for method in threshold_methods:       
        tp[method] = dict()
        tn[method] = dict()
        fp[method] = dict()
        fn[method] = dict()
        th[method] = dict()
            
    for idx in dataset_settings['indices_eval']:
        
        im_a = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_a.npy')))
        im_b = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_b.npy')))
        gt = np.load(os.path.join(directories['labels_path'], str(idx)+'.npy'))
        gt -= 1
        gt_pos = gt > 0

        if network_settings['extract_features'] != None:
            if isinstance(network_settings['extract_features'], list): 
                features = extract_features_only_feat(
                    net, 
                    network_settings['extract_features'], 
                    [im_a, im_b])
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
            # calculate change detection accuracy
            cm_pos = changemap > 0    
            tp[method][str(idx)] = np.logical_and(cm_pos, gt_pos).sum()
            tn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), np.logical_not(gt_pos)).sum()
            fp[method][str(idx)] = np.logical_and(cm_pos, np.logical_not(gt_pos)).sum()
            fn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), gt_pos).sum()
        else:
            if network_settings['extract_features'] == None:
                distmap = calculate_distancemap(im_a, im_b)
            elif network_settings['extract_features'] == 'joint':
                distmap = calculate_magnitudemap(features)
            elif network_settings['extract_features'] == 'diff':
                distmap = calculate_magnitudemap(features)
            elif isinstance(network_settings['extract_features'], list): 
                distmap = calculate_distancemap(features[0], features[1])
            else:
                 raise Exception('distance map calculation not implemented for these settings of extract_features')              
            np.save(os.path.join(
                directories['results_dir_cd'],                              
                save_networkname,'cva','distmap_'+str(idx)), 
                distmap)
            
            for method in threshold_methods:
                if method == 'f1':
                    precision, recall, thresholds = precision_recall_curve(gt.ravel(), 
                                                                           distmap.ravel())
                    f1 = (2*precision*recall)/(precision+recall)
                    threshold = thresholds[np.nanargmax(f1)]
                    changemap = distmap>threshold
                elif method == 'AA':
                    fpr, tpr, thresholds = roc_curve(gt.ravel(), distmap.ravel())
                    tnr = 1-fpr
                    avg_acc = (tpr+tnr)/2
                    threshold = thresholds[np.nanargmax(avg_acc)]
                    changemap = distmap>threshold
                else:
                    changemap, threshold = calculate_changemap(distmap, method=method, plot=True)
                np.save(os.path.join(
                    directories['results_dir_cd'], save_networkname,'cva',
                    'changemap_threshold_'+method+'_'+str(threshold)+'_'+str(idx)+'.npy'), 
                    changemap)
                

                fig, ax = plot_changemap_colors(gt, changemap, axis=False, title=None)
                plt.savefig(os.path.join(
                    directories['results_dir_cd'], save_networkname,'cva',
                    'changemap_threshold_'+method+'_'+str(threshold)+'_'+str(idx)+'.png'))
                plt.show()
                
                # calculate change detection accuracy
                cm_pos = changemap > 0    
                tp[method][str(idx)] = np.logical_and(cm_pos, gt_pos).sum()
                tn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), np.logical_not(gt_pos)).sum()
                fp[method][str(idx)] = np.logical_and(cm_pos, np.logical_not(gt_pos)).sum()
                fn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), gt_pos).sum()
                th[method][str(idx)] = threshold

    return (tp, tn, fp, fn, th)
            
def detect_changes_no_gt(model_settings, directories, dataset_settings, network_settings, 
                         train_settings, threshold_methods=['triangle'], 
                         threshold_from_file={'f1':0.8, 'AA':0.5}):
    """
    Calculate the confusion matrix for specified thresholding methods, 
    including inference for images for which no ground truth is available.

    Parameters
    ----------
    model_settings : dict
        Settings used to during training of the model
    directories :  dict
        Directories to load and save data
    dataset_settings : dict
        Settings that specify the used dataset
    network_settings : dict
        Settings that specity the used network
    train_settings : dict
        Settings that specity the training parameters
    threshold_methods : list, optional
        Threshold methods to use. Options: "otsu" | "triangle" | "end-to-end" | "f1" | "AA" 
        The default is ['triangle'].
    threshold_from_file : dict, optional
        specify thresholds to use for AA or f1. The default is {'f1':0.8, 'AA':0.5}.

    Raises
    ------
    Exception
        network_settings['extract_features'] should be one of None | 'joint' | 'diff' | list of layernumbers 

    Returns
    -------
    tp : dict
        True positives of every image in for every thresholding method
    tn : dict
        True negatives of every image in for every thresholding method
    fp : dict
        False positives of every image in for every thresholding method
    fn : dict
        False negatives of every image in for every thresholding method
    th : dict
        Threshold-value applied to get the result
    """
    
    
    save_networkname = model_settings['filename'].split('/')[-1]
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname)):
        os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'testset')):
        os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'testset'))
    if not os.path.exists(os.path.join(
            directories['results_dir_cd'], save_networkname,'testset',
            network_settings['str_extract_features'])):
        os.mkdir(os.path.join(
            directories['results_dir_cd'], save_networkname,'testset', 
            network_settings['str_extract_features']))
    if not os.path.exists(os.path.join(
            directories['results_dir_cd'], save_networkname,'testset',
            network_settings['str_extract_features'],'ENVI')):
        os.mkdir(os.path.join(
            directories['results_dir_cd'], save_networkname,'testset', 
            network_settings['str_extract_features'], 'ENVI'))
            
    # get_network
    if network_settings['extract_features'] != None:
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
    else:
        conv_classifier = False
    
    tp = dict()
    tn = dict()
    fp = dict()
    fn = dict()
    th = dict()

    for method in threshold_methods:       
        tp[method] = dict()
        tn[method] = dict()
        fp[method] = dict()
        fn[method] = dict()
        th[method] = dict()
            
    for idx in dataset_settings['indices_test']:
        
        im_a = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_a.npy')))
        im_b = normalize2plot(np.load(os.path.join(directories['data_path'], str(idx)+'_b.npy')))
        
        if network_settings['extract_features'] != None:
            if isinstance(network_settings['extract_features'], list): 
                features = extract_features_only_feat(
                    net, network_settings['extract_features'], [im_a, im_b])
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
        else:
            if network_settings['extract_features'] == None:
                distmap = calculate_distancemap(im_a, im_b)
            elif network_settings['extract_features'] == 'joint':
                distmap = calculate_magnitudemap(features)
            elif network_settings['extract_features'] == 'diff':
                distmap = calculate_magnitudemap(features)
            elif isinstance(network_settings['extract_features'], list): 
                distmap = calculate_distancemap(features[0], features[1])
            else:
                 raise Exception('distance map calculation not implemented for these settings of extract_features')              
            np.save(os.path.join(
                directories['results_dir_cd'], 
                save_networkname,'cva','distmap_'+str(idx)), 
                distmap)
            
            for method in threshold_methods:
                if method == 'f1':
                    threshold = threshold_from_file['f1']
                    changemap = distmap>threshold
                elif method == 'AA':
                    threshold = threshold_from_file['AA']
                    changemap = distmap>threshold
                else:
                    changemap, threshold = calculate_changemap(
                        distmap, 
                        method=method, plot=True)
                np.save(os.path.join(
                    directories['results_dir_cd'], save_networkname,'cva',
                    'changemap_threshold_'+method+'_'+str(threshold)+'_'+str(idx)+'.npy'), 
                    changemap)
                fig, ax = plt.subplots(figsize=(5,5))
                ax.imshow(changemap, cmap='gray')
                ax.axis('off')
                plt.savefig(os.path.join(
                    directories['results_dir_cd'], save_networkname,'testset', 
                    network_settings['str_extract_features'],
                    str(idx)+'_GRAY_threshold-'+method+'-'+str(threshold)+'.png'))
                plt.show()
        
                # calculate change detection accuracy
                if os.path.exists(os.path.join(directories['labels_path'], str(idx)+'.npy')):
                    gt = np.load(os.path.join(directories['labels_path'], str(idx)+'.npy'))
                    gt -= 1
                    gt_pos = gt > 0
                    cm_pos = changemap > 0  
                    tp[method][str(idx)] = np.logical_and(cm_pos, gt_pos).sum()
                    tn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), np.logical_not(gt_pos)).sum()
                    fp[method][str(idx)] = np.logical_and(cm_pos, np.logical_not(gt_pos)).sum()
                    fn[method][str(idx)] = np.logical_and(np.logical_not(cm_pos), gt_pos).sum()
                    th[method][str(idx)] = threshold
                    
                    fig, ax = plot_changemap_colors(gt, changemap, axis=False, title=None)
                    plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname,'testset', network_settings['str_extract_features'], 
                                             str(idx)+'_COLOR_threshold-'+method+'-'+str(threshold)+'.png'))
                    plt.show()
                    
                else:
                    prediction = changemap.astype(np.int64) + 1
                    array2raster(os.path.join(
                        directories['results_dir_cd'], save_networkname,'testset', 
                        network_settings['str_extract_features'],'ENVI',
                        str(idx)+'_threshold-'+method+'-'+str(threshold)+ '.raw'),(0,0),1,1, prediction) # convert array to raster 

    return (tp, tn, fp, fn, th)
        
        