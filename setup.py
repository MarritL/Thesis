#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:43:40 2020

@author: M. Leenstra
"""

import numpy as np

def setup(computer='desktop', network='triplet_apn', gpu=None, batch_size=25, 
          disp_iter=25, dataset_type='triplet_apn', loss='bce_sigmoid'):

    computer = computer

    # init
    if computer == 'desktop':
        directories = {
            'intermediate_dir_training': '/media/cordolo/marrit/Intermediate/training_S2',
            'results_dir_training': '/media/cordolo/marrit/results/training_S2',
            'results_dir_cd': '/media/cordolo/marrit/results/CD_OSCD',
            'results_dir': '/media/cordolo/marrit/results',
            'intermediate_dir_cd': '/media/cordolo/marrit/Intermediate/CD_OSCD',
            'intermediate_dir_cd_old': '/media/cordolo/marrit/Intermediate/CD_OSCD_old',
            'intermediate_dir': '/media/cordolo/marrit/Intermediate',
            'source_dir_cd': '/media/cordolo/marrit/source_data/CD_OSCD',
            'source_dir_barrax': '/media/cordolo/marrit/source_data/CD_barrax',
            'csv_file_S21C': 'S21C_dataset.csv',
            'csv_file_S21C_cleaned': 'S21C_dataset_clean.csv',
            'csv_file_oscd': 'OSCD_dataset.csv',
            'csv_file_train_oscd' : 'OSCD_train.csv',
            'csv_file_test_oscd': 'OSCD_test.csv',
            'csv_file_patches_overlap': 'S21C_patches_overlap50-70.csv',
            'csv_models': 'trained_models.csv',
            'csv_models_finetune': 'trained_models_finetune.csv',
            'csv_models_downstream': 'trained_model_downstream.csv',
            'csv_models_results': 'models_results.csv',
            'data_dir_S21C': 'data_S21C',
            'data_dir_barrax': 'data_barrax',
            'data_dir_oscd': 'data_OSCD',
            'data_dir_patches_overlap': 'patches_S21C_overlap50-70',
            'labels_dir_oscd' : 'labels_OSCD',
            'tb_dir': '/media/cordolo/marrit/Intermediate/tensorboard',
            'model_dir': '/media/cordolo/marrit/Intermediate/trained_models',
            'model_dir_finetune': '/media/cordolo/marrit/Intermediate/trained_models_finetune',
            'model_dir_downstream': '/media/cordolo/marrit/Intermediate/trained_models_downstream'}
    elif computer == 'optimus':
        directories = {
            'intermediate_dir_training': '/marrit1/Intermediate/training_S2',
            'results_dir_training': '/marrit1/results/training_S2',
            'results_dir_cd': '/marrit1/results/CD_OSCD',
            'intermediate_dir_cd': '/marrit1/Intermediate/CD_OSCD',
            'intermediate_dir': '/marrit1/Intermediate',
            'csv_file_S21C': 'S21C_dataset.csv',
            'csv_file_S21C_cleaned': 'S21C_dataset_clean.csv',
            'csv_file_oscd': 'OSCD_dataset.csv',
            'csv_file_train_oscd' : 'OSCD_train.csv',
            'csv_file_test_oscd': 'OSCD_test.csv',
            'csv_file_patches_overlap': 'S21C_patches_overlap90-100.csv',
            'csv_models': 'trained_models.csv',
            'csv_models_results': 'models_results.csv',
            'data_dir_S21C': 'data_S21C',
            'data_dir_oscd': 'data_OSCD',
            'labels_dir_oscd' : 'labels_OSCD',
            'data_dir_patches_overlap': 'patches_S21C_overlap90-100',
            'tb_dir': '/marrit1/Intermediate/tensorboard',
            'model_dir': '/marrit1/Intermediate/trained_models'}
    
    network_settings = {
        'network': network,
        'optimizer': 'adam',
        'lr':1e-3,
        'weight_decay':0,
        'loss': loss,
        'n_classes': 2,
        'patch_size': 96,
        'im_size': (96,96),
        'batch_norm': True,
        'weights_file': '' ,
        'extract_features': None,
        'avg_pool': False}
    
    # network cofiguration:
    # number denotes number of output channels of conv layer or linear layer
    # 'M' denotes max-pooling layer (2x2), stride=2
    # 'D' denotes drop-outlayer (only in classifier for siamese/triplet network)
    # classifier in siamese/triplet network uses linear layers
    # classifier in hypercolumn network uses conv layers
    # note: first number of top should be 2* or 3* last conv layer of branch
    cfg = {'branch': np.array([64, 64, 64, 64, 64], dtype='object'), 
           'top': np.array([384], dtype='object'),
           'classifier': np.array([256,128,network_settings['n_classes']])}
    network_settings['cfg'] = cfg
    
    # add layers from which to extract features in hypercolumn network
    if network_settings['network'] == 'hypercolumn':
        branch = network_settings['cfg']['branch']
        # extract the features from all activation layers
        network_settings['extract_features'] = list()
        layer_i = -1
        if network_settings['batch_norm']:
            for layer in branch:
                if layer == 'M':
                    layer_i += 1
                else:
                    layer_i += 3
                    network_settings['extract_features'].append(layer_i)
        else:
            for layer in branch:
                if layer == 'M':
                    layer_i += 1
                else:
                    layer_i += 2
                    network_settings['extract_features'].append(layer_i)
    
    train_settings = {
        'start_epoch': 0,
        'num_epoch': 25,
        'batch_size': batch_size,
        'disp_iter': disp_iter,
        'gpu': gpu}
    
    dataset_settings = {
        'dataset_type' : dataset_type,
        'perc_train': 0.85,
        'channels': np.arange(13),
        'min_overlap': 1, 
        'max_overlap': 1,
        'patches_per_image': 25}
    
    return directories, network_settings, train_settings, dataset_settings