#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:41:53 2020

@author: cordolo
"""
import os
import numpy as np
import torch
import csv
from torch.nn.functional import softmax
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from plots import normalize2plot, plot_changemap_colors
from data_generator import channelsfirst
from train import get_downstream_network
from extract_features import calculate_changemap

def load_images(data_dir, filenames, channels=np.arange(13)):
    
    images = {}
    for filename in filenames:
        images[filename] = np.load(
            os.path.join(data_dir, filename))[:, :, channels]

        # replace potential nan values 
        images[filename][np.isnan(images[filename])] = 0
                       
    return images

def to_torch(images):
    
    for im in images:
        images[im] = torch.from_numpy(images[im]).unsqueeze(0)
        
    return images

def inference(directories, dataset_settings, model_settings, train_settings, 
              channels=np.arange(13), percentile=99, extract_features=None, 
              avg_pool=None, use_softmax=False):
    
    # get network
    n_branches = 2
    network, conv_classifier = get_downstream_network(model_settings, train_settings['gpu'], n_branches)
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
    
    save_networkname = model_settings['filename'].split('/')[-1]
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
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname, 'best_thresholds.csv' )):
        with open(os.path.join(directories['results_dir_cd'], save_networkname, 'best_thresholds.csv' ), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()

    for q, idx in enumerate(indices):
        filename = str(idx)+'.npy'
        prob = np.load(os.path.join(prob_dir, filename))
        prob_change = prob[1]
        
        gt = np.load(os.path.join(directories['labels_path'], filename))
        gt = gt-1
        fig, ax = plt.subplots()
        ax.imshow(gt, cmap='gray')
        ax.axis('off')
        plt.show()
        
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
        with open(os.path.join(directories['results_dir_cd'], save_networkname, 'best_thresholds.csv' ), 'a') as file:
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
        prob = np.load(os.path.join(prob_dir, filename))
        prob_change = prob[1]
        
        # plot          
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(prob_change>threshold, cmap='gray')
        ax.axis('off')
        plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'threshold_testset', 
                                     str(idx)+'_GRAY_threshold-'+threshold_name+'-'+str(threshold)+'.png'))
        plt.show()
        
        if os.path.exisits(os.path.join(directories['labels_path'], filename)):
            gt = np.load(os.path.join(directories['labels_path'], filename))
            gt = gt-1
            fig, ax = plt.subplots()
            ax.imshow(gt, cmap='gray')
            ax.axis('off')
            plt.show()
    
            prediction = prob_change>threshold
            fig, ax = plot_changemap_colors(gt, prediction, axis=False, title=None)
            plt.savefig(os.path.join(directories['results_dir_cd'], save_networkname, 'result_testset', 
                                     str(idx)+'_COLOR_threshold-'+threshold_name+'-'+str(threshold)+'.png'))
            plt.show()
            
            tp = np.sum(prediction == 1) & (prediction == gt))
            fp = np.sum((prediction == 1) & (prediction != gt))
            tn = np.sum((prediction == 0) & (prediction == gt))
            fn = np.sum((prediction == 0) & (prediction != gt))
            precision = tp.float() / (tp.float()+fp.float()+1e-10)
            recall = tp.float() / (tp.float()+fn.float()+1e-10)
            f1 = 2 * (precision * recall) / (precision + recall+1e-10)
            neg_acc = tn.float() / (tn.float()+fp.float()+1e-10)
            pos_acc = tp.float() / (tp.float()+fn.float()+1e-10)
            avg_acc = (neg_acc+pos_acc)/2
       
            # save in csv
            with open(os.path.join(directories['results_dir_cd'], save_networkname, 'best_thresholds.csv' ), 'a') as file:
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
   
