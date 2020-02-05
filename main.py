#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:22:05 2020

@author: M. Leenstra
"""

import os
import numpy as np
import pandas as pd
from train import train

# init
directories = {
    'intermediate_dir_training': '/media/cordolo/elements/Intermediate/training_S2',
    'csv_file_S21C_cleaned': 'S21C_dataset_cleaned.csv',
    'data_dir_S21C': 'data_S21C',
    'tb_dir': '/media/cordolo/elements/Intermediate/tensorboard'}

directories['data_path'] = os.path.join(
    directories['intermediate_dir_training'], 
    directories['data_dir_S21C'])

network_settings = {
    'network': 'siamese',
    'optimizer': 'adam',
    'lr': 0.001,
    'loss': 'bce_sigmoid',
    'n_classes': 2}

train_settings = {
    'start_epoch': 0,
    'num_epoch': 10,
    'batch_size': 10,
    'disp_iter': 2}

dataset_settings = {
    'dataset_type' : 'triplet',
    'perc_train': 0.8,
    'channels': np.arange(13)}

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_training'],directories['csv_file_S21C_cleaned']))
dataset = np.random.choice(dataset['im_idx'], 1, replace=False)
dataset_settings['indices_train'] = np.repeat(dataset, 80)
dataset_settings['indices_val'] = np.repeat(dataset, 20)
# =============================================================================
# dataset = np.random.choice(dataset['im_idx'], 50, replace=False)
# dataset_settings['indices_train'] = dataset[:int(dataset_settings['perc_train']*len(dataset))]
# dataset_settings['indices_val'] = dataset[int(dataset_settings['perc_train']*len(dataset)):]
# =============================================================================

#%% 
""" Train """
train(directories, dataset_settings, network_settings, train_settings)

#%%