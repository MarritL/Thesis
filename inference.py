#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:41:53 2020

@author: M. Leenstra
"""
import os
import numpy as np
import torch
import csv
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import gdal, ogr, os, osr


from plots import normalize2plot, plot_changemap_colors
from data_generator import channelsfirst
from train import get_downstream_network, get_network
from extract_features import calculate_changemap, calculate_distancemap_hypercolumn


def inference(directories, dataset_settings, model_settings, train_settings, 
              channels=np.arange(13), percentile=99, extract_features=None, 
              avg_pool=None, use_softmax=False):
    """ Calculate probability maps based on extract feature layers """
    
    # get network
    n_branches = 2
    if model_settings['networkname'].startswith('CD') or model_settings['networkname'].startswith('FINETUNE'):
        network, conv_classifier = get_downstream_network(model_settings, train_settings['gpu'], n_branches)
    else:
        network = get_network(model_settings, train_settings['gpu'])
        conv_classifier = True
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
       
    for q, idx in enumerate(dataset_settings['indices_test']):
        # get filenames
        filenames = [str(idx)+'_a.npy', str(idx)+'_b.npy']
        # prepare images
        images = load_images(directories['data_path'], filenames, channels=channels)    
        for filename in filenames:
            images[filename] = normalize2plot(images[filename], percentile)      
        images = channelsfirst(images)
        images = to_torch(images)
         
        if isinstance(extract_features, list): 
            prob_maps = inference_on_images_hypercolumn(network=network,    
                                                  images=images, 
                                                  conv_classifier=conv_classifier, 
                                                  n_branches=n_branches, 
                                                  gpu=train_settings['gpu'], 
                                                  extract_features=extract_features,
                                                  avg_pool=avg_pool,
                                                  use_softmax=use_softmax)
        else:
            out = inference_on_images(network=network, 
                                      images=images, 
                                      conv_classifier=conv_classifier, 
                                      n_branches=n_branches, 
                                      gpu=train_settings['gpu'], 
                                      extract_features=extract_features,
                                      avg_pool=avg_pool,
                                      use_softmax=use_softmax)
            
            
            #back to numpy
            prob_maps = out.squeeze().detach().cpu().numpy()
        
        save_networkname = model_settings['filename'].split('/')[-1]
        if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname)):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname))
        if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps'))
        
        np.save(os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps',str(idx)), prob_maps)
    
        print('\r {}/{}'.format(q+1, len(dataset_settings['indices_test'])))
        
    print('all probability maps saved!')
  
def inference_on_images(network, images, conv_classifier, 
                        n_branches=2, gpu=None, extract_features=None,
                        avg_pool=None, use_softmax=False):
    """ Calculate probability maps based on extract feature layers """
    
    network.eval() 
    
    inputs = list()
    for im in images: 
        if gpu != None:
            inputs.append(images[im].float().cuda())
        else:
            inputs.append(images[im].float())               
         
    # forward pass 
    with torch.no_grad():
        outputs = network(inputs, n_branches, extract_features=extract_features, 
                      avg_pool=avg_pool,conv_classifier=conv_classifier, 
                      use_softmax=use_softmax)
        
    probabilities = softmax(outputs, dim=1)

    return probabilities

    
def find_best_threshold(directories, indices, model_settings):
    """
    Use precision-recall curve to find best threshold for f1-score and
    Use roc_curve to find best threshold for Average Accuracy

    Parameters
    ----------
    directories :  dict
        Directories to load and save data
    indices : np.array
        indices of images for which to find best threshold
    model_settings : dict
        Settings used to during training of the model

    Returns
    -------
    None. Saves best thresholds in csv file in folder of pretext model

    """
    
    save_networkname = model_settings['filename'].split('/')[-1]
    save_pretaskname = model_settings['pretask_filename'].split('/')[-1]
    prob_dir = os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps')
    
    thresholds_f1 = list()
    f1s = list()
    recalls = list()
    precisions = list()
    thresholds_avg_acc = list()
    tnrs = list()
    tprs = list()
    avg_accs = list()
    
    # init save-file
    fieldnames = ['im_idx', 'kthfold', 'filename','pretask_filename','networkname', \
              'layers_branches', 'layers_joint', 'cfg_classifier_cd',\
              'threshold_f1', 'f1', 'threshold_avg_acc', 'avg_acc']
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname, 'best_thresholds.csv' )):
        with open(os.path.join(directories['results_dir_cd'], save_pretaskname, 'best_thresholds.csv' ), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()

    for q, idx in enumerate(indices):
        filename = str(idx)+'.npy'
        prob_change = np.load(os.path.join(prob_dir, filename))
        if len(prob_change.shape) > 2: 
            prob_change = prob_change[1] 
        
        gt = np.load(os.path.join(directories['labels_path'], filename))
        gt = gt-1
# =============================================================================
#         fig, ax = plt.subplots()
#         ax.imshow(gt, cmap='gray')
#         ax.axis('off')
#         plt.show()
# =============================================================================
        
        # calculate f1 for various thresholds
        precision, recall, thresholds = precision_recall_curve(gt.ravel(), prob_change.ravel())
        f1 = (2*precision*recall)/(precision+recall)
        best_f1 = np.nanmax(f1)
        best_threshold = thresholds[np.nanargmax(f1)]
        best_recall = recall[np.nanargmax(f1)]
        best_precision = precision[np.nanargmax(f1)]
        thresholds_f1.append(thresholds)    
        f1s.append(f1)
        recalls.append(recall)
        precisions.append(precision)

        if not os.path.exists(os.path.join(directories['results_dir_cd'],  save_networkname,'threshold_f1')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_f1'))    
         
        # plots
        fig, ax = plot_changemap_colors(gt, prob_change>best_threshold, axis=False, title=None)
        plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_f1', 
                                 str(idx)+'_COLOR_threshold-'+str(best_threshold)+'_f1-'+str(best_f1)+'.png'))
        
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(prob_change>best_threshold, cmap='gray')
        ax.axis('off')
        plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_f1', 
                                 str(idx)+'_GRAY_threshold-'+str(best_threshold)+'_f1-'+str(best_f1)+'.png'))
        plt.show()
        
        # save also numpy array
        np.save(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_f1',
                             str(idx)+'_threshold-'+str(best_threshold)+'_f1-'+str(best_f1)), 
                prob_change>best_threshold)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_f1',
                             str(idx)+'_recall'), recall)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_f1',
                             str(idx)+'_precision'), precision)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_f1',
                             str(idx)+'_f1'), f1)
   
        
        # calculate average accuracy for various thresholds
        fpr, tpr, thresholds = roc_curve(gt.ravel(), prob_change.ravel())
        tnr = 1-fpr
        avg_acc = (tpr+tnr)/2
        best_avg_acc = avg_acc[np.nanargmax(avg_acc)]
        best_threshold2 = thresholds[np.nanargmax(avg_acc)]
        best_tpr = tpr[np.nanargmax(avg_acc)] # Sensitivity
        best_tnr = tnr[np.nanargmax(avg_acc)] # Specificity
        thresholds_avg_acc.append(thresholds)
        tnrs.append(tnr)
        tprs.append(tpr)
        avg_accs.append(avg_acc)
       
        # SAVE
        if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_avg_acc')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'threshold_avg_acc'))   
        
        # plots
        fig, ax = plot_changemap_colors(gt, prob_change>best_threshold2, axis=False, title=None)
        plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_COLOR_threshold-'+str(best_threshold2)+'_avg_acc-'+str(best_avg_acc)+'.png'))
        
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(prob_change>best_threshold2, cmap='gray')
        ax.axis('off')
        plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_GRAY_threshold-'+str(best_threshold2)+'_avg_acc-'+str(best_avg_acc)+'.png'))
        plt.show()
        
        # np array
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_threshold-'+str(best_threshold2)+'_avg_acc-'+str(best_avg_acc)), 
                prob_change>best_threshold2)
        
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_thresholds'), thresholds)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_fpr'), fpr)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_tpr'), tpr)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_tnr'), tnr)
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_avg_acc',
                             str(idx)+'_avg_acc'), avg_acc)

        # save in csv
        with open(os.path.join(directories['results_dir_cd'], save_pretaskname, 'best_thresholds.csv' ), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
            filewriter.writerow({'im_idx': idx, 
                                 'kthfold': model_settings['kthfold'],
                                 'filename': model_settings['filename'],
                                 'pretask_filename': model_settings['pretask_filename'],
                                 'networkname': model_settings['networkname'], 
                                 'layers_branches': model_settings['layers_branches'], 
                                 'layers_joint': model_settings['layers_joint'], 
                                 'cfg_classifier_cd': model_settings['cfg_classifier_cd'],
                                 'threshold_f1': best_threshold, 
                                 'f1': best_f1, 
                                 'threshold_avg_acc': best_threshold2, 
                                 'avg_acc': best_avg_acc})  

        print('\r {}/{}'.format(q+1, len(indices)))
    return thresholds_f1, f1s, recalls, precisions, thresholds_avg_acc, tnrs, tprs, avg_accs

def apply_best_threshold(directories, indices, model_settings, threshold, threshold_name):
    """
    Applies specified threshold to images specied by indices

    Parameters
    ----------
    directories :  dict
        Directories to load and save data
    indices : np.array
        indices of images for which to find best threshold
    model_settings : dict
        Settings used to during training of the model
    threshold : float
        threshold-value.
    threshold_name : string
        name of threshold for used to save files.

    Returns
    -------
    None. Saves resulting change maps in folder of pretext model

    """
    
    save_networkname = model_settings['filename'].split('/')[-1]
    prob_dir = os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps')
    
    if not os.path.exists(os.path.join(directories['results_dir_cd'],  save_networkname,'result_testset')): 
        os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'result_testset')) 
        
    # init save-file
    fieldnames = ['im_idx', 'kthfold', 'filename','pretask_filename','networkname', \
              'layers_branches', 'layers_joint', 'cfg_classifier_cd',\
              'threshold_name', 'threshold', 'tp', 'tn', 'fp', 'fn', \
               'precision', 'recall', 'f1', 'neg_acc', 'pos_acc', 'avg_acc']
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'result_testset', 'quantitative_results.csv')):
        with open(os.path.join(directories['results_dir_cd'], save_networkname,'result_testset', 'quantitative_results.csv'), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()
    
    for q, idx in enumerate(indices):
        filename = str(idx)+'.npy'
        prob_change = np.load(os.path.join(prob_dir, filename))
        if len(prob_change.shape) > 2: 
            prob_change = prob_change[1]   
        prediction = prob_change>threshold
        
        # plot          
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(prob_change>threshold, cmap='gray')
        ax.axis('off')
        plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'result_testset', 
                                     str(idx)+'_GRAY_threshold-'+threshold_name+'-'+str(threshold)+'.png'))
        plt.show()
        
        # save numpy array
        np.save(os.path.join(directories['results_dir_cd'], save_networkname, 'result_testset', 
                     str(idx)+'_threshold-'+threshold_name+'-'+str(threshold)), 
                        prediction)
        
        # save ENVI raw
        prediction = prediction.astype(np.int64) + 1
        array2raster(os.path.join(directories['results_dir_cd'], save_networkname, 'result_testset', 
                                  str(idx)+'_threshold-'+threshold_name+'-'+str(threshold)+ '.raw'),(0,0),1,1, prediction) # convert array to raster 

        
        
        if os.path.exists(os.path.join(directories['labels_path'], filename)):
            prediction = prob_change>threshold
            gt = np.load(os.path.join(directories['labels_path'], filename))
            gt = gt-1
            fig, ax = plt.subplots()
            ax.imshow(gt, cmap='gray')
            ax.axis('off')
            plt.show()
    

            fig, ax = plot_changemap_colors(gt, prediction, axis=False, title=None)
            plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'result_testset', 
                                     str(idx)+'_COLOR_threshold-'+threshold_name+'-'+str(threshold)+'.png'))
            plt.show()
            
            tp = np.sum((prediction == 1) & (prediction == gt))
            fp = np.sum((prediction == 1) & (prediction != gt))
            tn = np.sum((prediction == 0) & (prediction == gt))
            fn = np.sum((prediction == 0) & (prediction != gt))
            precision = tp / (tp+fp+1e-10)
            recall = tp / (tp+fn+1e-10)
            f1 = 2 * (precision * recall) / (precision + recall+1e-10)
            neg_acc = tn / (tn+fp+1e-10)
            pos_acc = tp / (tp+fn+1e-10)
            avg_acc = (neg_acc+pos_acc)/2
       
            # save in csv
            with open(os.path.join(directories['results_dir_cd'], save_networkname, 'quantitative_results.csv' ), 'a') as file:
                filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
                filewriter.writerow({'im_idx': idx, 
                                 'kthfold': model_settings['kthfold'],
                                 'filename': model_settings['filename'],
                                 'pretask_filename': model_settings['pretask_filename'],
                                 'networkname': model_settings['networkname'], 
                                 'layers_branches': model_settings['layers_branches'], 
                                 'layers_joint': model_settings['layers_joint'], 
                                 'cfg_classifier_cd': model_settings['cfg_classifier_cd'],
                                 'threshold_name': threshold_name, 
                                 'threshold': threshold, 
                                 'tp': tp, 
                                 'tn': tn, 
                                 'fp': fp, 
                                 'fn': fn, 
                                 'precision': precision, 
                                 'recall': recall, 
                                 'f1': f1, 
                                 'neg_acc': neg_acc, 
                                 'pos_acc': pos_acc, 
                                 'avg_acc': avg_acc})  
                               
        print('\r {}/{}'.format(q+1, len(indices))) 
 
def apply_thresholds_from_probmap(model_settings, directories, dataset_settings, 
                                  network_settings, train_settings, 
                                  threshold_methods=['triangle'], 
                                  threshold_from_file={'f1':0.8, 'AA':0.5}):
    """
    Calculate the confusion matrix for specified thresholding methods, 
    including inference for images for which no ground truth is available

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
    save_pretaskname = model_settings['pretask_filename'].split('/')[-1]
    save_layers_branches = str(model_settings['layers_branches'])
    save_layers_joint = str(model_settings['layers_joint'])
    if model_settings['cfg_classifier_cd'] == "['C', '2']":
        classifier = 'C2'
    elif model_settings['cfg_classifier_cd'] == "['C', '32', '2']":
        classifier = 'C322'
    elif model_settings['cfg_classifier_cd'] == 'CVA':
        classifier = 'CVA'
    else:
        raise Exception('Classifier undefined!')
    prob_dir = os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps')
        
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname)):
        os.mkdir(os.path.join(directories['results_dir_cd'], save_pretaskname))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname, dataset_settings['dataset_to_use'])):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use']))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'],save_layers_branches+'_'+save_layers_joint)):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'],save_layers_branches+'_'+save_layers_joint, classifier)):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier))
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'],save_layers_branches+'_'+save_layers_joint,classifier,'ENVI')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint,classifier ,'ENVI'))
                   
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
        filename = str(idx)+'.npy'
        probmap = np.load(os.path.join(prob_dir, filename))
        if len(probmap.shape) > 2: 
            probmap = probmap[1]             

        for method in threshold_methods:
            if method == 'f1':
                threshold = threshold_from_file['f1']
                changemap = probmap>threshold
            elif method == 'AA':
                threshold = threshold_from_file['AA']
                changemap = probmap>threshold
            elif method == 'end-to-end':
                threshold = 0.5
                changemap = probmap>threshold
            else:
                changemap, threshold = calculate_changemap(probmap, method=method, plot=False)
            np.save(os.path.join(directories['results_dir_cd'], save_networkname,'cva',
                                     'changemap_threshold_'+method+'_'+str(threshold)+'_'+str(idx)+'.npy'), changemap)
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(changemap, cmap='gray')
            ax.axis('off')
            plt.savefig(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier,
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
                plt.savefig(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier,
                                         str(idx)+'_COLOR_threshold-'+method+'-'+str(threshold)+'.png'))
                plt.show()
                
            else:
                prediction = changemap.astype(np.int64) + 1
                array2raster(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'],save_layers_branches+'_'+save_layers_joint,classifier,'ENVI',
                              str(idx)+'_threshold-'+method+'-'+str(threshold)+ '.raw'),(0,0),1,1, prediction) # convert array to raster 

    return (tp, tn, fp, fn, th)    
    
def inference_on_images_hypercolumn(network, images, conv_classifier, 
                        n_branches=2, gpu=None, extract_features=None,
                        avg_pool=None, use_softmax=False):
    """ Inference if features from more than 1 layer are used"""
    
    network.eval() 
    if extract_features[-1] == 'joint':  
        extract_features1 = extract_features[:-1]
        extract_features2 = extract_features[-1]
        outputs_columns = list()
    
    inputs = list()
    for im in images:      
        if gpu != None:
            inputs.append(images[im].float().cuda())
        else:
            inputs.append(images[im].float())               
       
    if extract_features[-1] == 'joint': 
        # forward pass 
        with torch.no_grad():
            outputs_columns.append(network(inputs, n_branches, extract_features=extract_features1, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=use_softmax))    
            outputs_columns.append(network([inputs[1], inputs[0]], n_branches, extract_features=extract_features1, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=use_softmax))    
    else:    
        # forward pass 
        with torch.no_grad():
            outputs = network(inputs, n_branches, extract_features=extract_features, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=use_softmax)
    
    if extract_features[-1] == 'joint': 
        outputs2 = network(inputs, n_branches, extract_features=extract_features2, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=use_softmax)
        outputs_im1 = torch.squeeze(torch.cat(outputs_columns[0], dim=1)).detach().numpy()
        outputs_im2 = torch.squeeze(torch.cat(outputs_columns[1], dim=1)).detach().numpy()
        outputs_joint = outputs2.squeeze().detach().numpy()                
        
        distmap = calculate_distancemap_hypercolumn(outputs_im1, outputs_im2, outputs_joint)
    else:
        raise "Not Implemented"
        
        
    return distmap

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    """ Supporting function needed to convert to dataformatot suitable for DASE portal """

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('ENVI')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
def extract_features_vis(directories, dataset_settings, model_settings, train_settings, 
              channels=np.arange(13), percentile=99, extract_features=None, 
              avg_pool=None, use_softmax=False, extract_filter=None):
    
    """ Visualise patches with highest activation """    
    from extract_features import extract_features_only_feat
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    #grid_patches_single = pd.read_csv('/media/cordolo/marrit/results/grid_patches_oscd.csv')
    grid_patches_single = pd.read_csv('grid_patches_oscd.csv')
    grid_patches_single = grid_patches_single[grid_patches_single.variant == 0]
    grid_patches_single = grid_patches_single.reset_index()
    patch_size = 96
    feature_patches = '3'
    
    # get network
    n_branches = 2
    if model_settings['networkname'].startswith('CD') or model_settings['networkname'].startswith('FINETUNE'):
        network, conv_classifier = get_downstream_network(model_settings, train_settings['gpu'], n_branches)
    else:
        network = get_network(model_settings, train_settings['gpu'])
        conv_classifier = True
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
    
    feature_patches = dict()   
    feat_list = list()
    filename_old = 'none'
    
    #del feature_patches
    for i, r in grid_patches_single.iterrows():
        # get filenames
        filename_a = str(r.im_idx)+'_a.npy'
        filename_b = str(r.im_idx)+'_b.npy'
        if filename_a != filename_old: 
            filename_old = filename_a
            # prepare images
            im_a = np.load(os.path.join(directories['data_path'], filename_a))
            im_b = np.load(os.path.join(directories['data_path'], filename_b))
        
            features = extract_features_only_feat(network,extract_features,[im_a, im_b])
            
        for extract_filter in np.arange(32): 
            feat_a = np.expand_dims(features[0][:,:,extract_filter],axis=0)
            feat_b = np.expand_dims(features[1][:,:,extract_filter],axis=0)
                
            patch_start = [r.row, r.col]
            
            patch0 = feat_a[:,patch_start[0]:patch_start[0]+patch_size,
                         patch_start[1]:patch_start[1]+patch_size]
            patch1 = feat_b[:,patch_start[0]:patch_start[0]+patch_size,
                         patch_start[1]:patch_start[1]+patch_size]
            if i == 0:
                feature_patches[str(extract_filter)] = np.concatenate([patch0, patch1], axis=0)
            else:
                feature_patches[str(extract_filter)] = np.concatenate([feature_patches[str(extract_filter)], patch0, patch1], axis=0)
       
        print('\r {}/{}'.format(i+1, len(grid_patches_single)), end='') 
        
        #grid_patches = pd.read_csv('/media/cordolo/marrit/results/grid_patches_oscd.csv')
        grid_patches = pd.read_csv('grid_patches_oscd.csv')
        grid_patches = grid_patches[grid_patches.variant.isin([0,1])]
        grid_patches = grid_patches.reset_index()
        fig = plt.figure(figsize=(5,7))
        n_rows = 12
        n_cols = 6
        plt_idx = 0
        gs = GridSpec(n_rows, n_cols)
        for extract_filter in [0,6,7,9,12,13,14,19,24,26,30,31]: 
            #sorted_means = np.flip(np.argsort(np.mean(feature_patches[str(extract_filter)], axis=(1,2))))
            sorted_l1norm = np.flip(np.argsort(np.sum(np.abs(feature_patches[str(extract_filter)]), axis=(1,2))))
            for q in range(6):
                #print(sorted_means[q])
                print(grid_patches.iloc[sorted_l1norm[int(q)]].im_idx)
                row_idx = sorted_l1norm[int(q)]
                grid_row = grid_patches.iloc[sorted_l1norm[int(q)]]
                if row_idx % 2 == 0:
                    filename = str(grid_row.im_idx)+'_a.npy'       
                else:
                    filename = str(grid_row.im_idx)+'_b.npy'      
                im = np.load(os.path.join(directories['data_path'], filename))
                patch_start = [grid_row.row, grid_row.col]
                patch = im[patch_start[0]:patch_start[0]+patch_size,
                             patch_start[1]:patch_start[1]+patch_size, [3,2,1]]
                
                ax = fig.add_subplot(gs[plt_idx])
                ax.imshow(normalize2plot(patch))
                ax.axis('off')
                plt_idx +=1
            print('---------------NEW FILTER-------------------')
        plt.show()
    print('all probability maps saved!')
    
def load_images(data_dir, filenames, channels=np.arange(13)):
    """ Supporting function to load images"""
    
    images = {}
    for filename in filenames:
        images[filename] = np.load(
            os.path.join(data_dir, filename))[:, :, channels]

        # replace potential nan values 
        images[filename][np.isnan(images[filename])] = 0
                       
    return images

def to_torch(images):
    """ Supporting function to convert to torch tensors"""
    
    for im in images:
        images[im] = torch.from_numpy(images[im]).unsqueeze(0)
        
    return images