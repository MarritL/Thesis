#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:22:05 2020

@author: M. Leenstra
"""

import os
import numpy as np
import pandas as pd

# init
directories = {
    'intermediate_dir_training': '/media/cordolo/elements/Intermediate/training_S2',
    'intermediate_dir_cd': '/media/cordolo/elements/Intermediate/CD_OSCD',
    'intermediate_dir': '/media/cordolo/elements/Intermediate',
    'csv_file_S21C_cleaned': 'S21C_dataset_cleaned.csv',
    'csv_file_oscd': 'OSCD_dataset.csv',
    'csv_file_train_oscd' : 'OSCD_train.csv',
    'csv_file_test_oscd': 'OSCD_test.csv',
    'csv_models': 'trained_models.csv',
    'data_dir_S21C': 'data_S21C',
    'data_dir_oscd': 'data_OSCD',
    'tb_dir': '/media/cordolo/elements/Intermediate/tensorboard',
    'model_dir': '/media/cordolo/elements/Intermediate/trained_models'}

directories['data_path'] = os.path.join(
    directories['intermediate_dir_training'], 
    directories['data_dir_S21C'])

# network cofiguration:
# number denotes number of output channels of 3x3 conv layer
# 'M' denotes max-pooling layer (2x2), stride=2
# note: first number of top should be 2* lasy conv layer of branch 
cfg = {
       'branch': np.array([64,'M'], dtype='object'), 
       'top': np.array([192], dtype='object')}




network_settings = {
    'network': 'triplet',
    'cfg': cfg,
    'optimizer': 'adam',
    'lr': 0.001,
    'weight_decay':0,
    'loss': 'bce_sigmoid',
    'n_classes': 2,
    'patch_size': 96,
    'batch_norm': False}

train_settings = {
    'start_epoch': 0,
    'num_epoch': 2,
    'batch_size': 10,
    'disp_iter': 2}

dataset_settings = {
    'dataset_type' : 'triplet',
    'perc_train': 0.9,
    'channels': np.arange(13)}

#%% 
""" Train """
from train import train

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_training'],directories['csv_file_S21C_cleaned']))
# =============================================================================
# dataset = np.random.choice(dataset['im_idx'], 1, replace=False)
# dataset_settings['indices_train'] = np.repeat(dataset, 80)
# dataset_settings['indices_val'] = np.repeat(dataset, 20)
# =============================================================================
dataset = np.random.choice(dataset['im_idx'], 100, replace=False)
#dataset = dataset['im_idx'].values
dataset_settings['indices_train'] = dataset[:int(dataset_settings['perc_train']*len(dataset))]
dataset_settings['indices_val'] = dataset[int(dataset_settings['perc_train']*len(dataset)):]

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Extract Features """

from extract_features import extract_features, calculate_distancemap, calculate_changemap

# get csv files
oscd_train = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_train_oscd']))
oscd_test = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_test_oscd']))
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], directories['csv_models']))

#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1]

# get image
randim = np.random.choice(oscd_train.im_idx.values)
randim_a = str(randim)+'_a.npy'
randim_b = str(randim)+'_b.npy'
im_a = np.load(os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd'],randim_a))
im_b = np.load(os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd'],randim_b))

# extract features
features = extract_features(directories=directories, imagelist=[im_a, im_b], model_settings=model_settings,  layers=[1])
assert(features[0].shape == features[1].shape)

# calculate change map
distmap = calculate_distancemap(features[0], features[1])
changemap = calculate_changemap(distmap, plot=True)





