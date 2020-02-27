#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:22:05 2020

@author: M. Leenstra
"""

import os
import numpy as np
import pandas as pd

computer = 'desktop'


# init
if computer == 'desktop':
    directories = {
        #'intermediate_dir_training': '/media/cordolo/elements/Intermediate/training_S2',
        'results_dir_training': '/media/cordolo/elements/results/training_S2',
        'intermediate_dir_cd': '/media/cordolo/elements/Intermediate/CD_OSCD',
        'intermediate_dir': '/media/cordolo/elements/Intermediate',
        'csv_file_S21C': 'S21C_dataset.csv',
        'csv_file_S21C_cleaned': 'S21C_dataset_clean.csv',
        'csv_file_oscd': 'OSCD_dataset.csv',
        'csv_file_train_oscd' : 'OSCD_train.csv',
        'csv_file_test_oscd': 'OSCD_test.csv',
        'csv_file_patches90-100': 'S21C_patches_overlap90-100.csv',
        'csv_models': 'trained_models.csv',
        'data_dir_S21C': 'data_S21C',
        'data_dir_oscd': 'data_OSCD',
        'data_dir_patches90-100': 'patches_S21C_overlap90-100',
        'labels_dir_oscd' : 'labels_OSCD',
        'tb_dir': '/media/cordolo/elements/Intermediate/tensorboard',
        'model_dir': '/media/cordolo/elements/Intermediate/trained_models'}
elif computer == 'optimus':
    directories = {
        #'intermediate_dir_training': '/media/cordolo/elements/Intermediate/training_S2',
        'results_dir_training': '/media/marrit/results/training_S2',
        'intermediate_dir_cd': '/media/marrit/Intermediate/CD_OSCD',
        'intermediate_dir': '/media/marrit/Intermediate',
        'csv_file_S21C': 'S21C_dataset.csv',
        'csv_file_S21C_cleaned': 'S21C_dataset_clean.csv',
        'csv_file_oscd': 'OSCD_dataset.csv',
        'csv_file_train_oscd' : 'OSCD_train.csv',
        'csv_file_test_oscd': 'OSCD_test.csv',
        'csv_models': 'trained_models.csv',
        'data_dir_S21C': 'data_S21C',
        'data_dir_oscd': 'data_OSCD',
        'labels_dir_oscd' : 'labels_OSCD',
        'tb_dir': '/media/marrit/Intermediate/tensorboard',
        'model_dir': '/media/marrit/Intermediate/trained_models'}

# network cofiguration:
# number denotes number of output channels of 3x3 conv layer
# 'M' denotes max-pooling layer (2x2), stride=2
# note: first number of top should be 2* or 3* las4 conv layer of branch 
cfg = {
       'branch': np.array([32,'M',64,'M',128,'M'], dtype='object'), 
       'top': np.array([384], dtype='object')}


network_settings = {
    'network': 'triplet',
    'cfg': cfg,
    'optimizer': 'adam',
    'lr':1e-3,
    'weight_decay':1e-4,
    'loss': 'bce_sigmoid',
    'n_classes': 2,
    'patch_size': 96,
    'batch_norm': False}

train_settings = {
    'start_epoch': 0,
    'num_epoch': 20,
    'batch_size': 50,
    'disp_iter': 5,
    'gpu': None}

dataset_settings = {
    'dataset_type' : 'triplet',
    'perc_train': 0.8,
    'channels': np.arange(13),
    'min_overlap': 0.9, 
    'max_overlap':  1}

#%%
""" Train on S21C presaved dataset 90-100% overlap """
from train import train

directories['data_path'] = os.path.join(
    directories['intermediate_dir'], 
    'training_S2',
    directories['data_dir_patches90-100'])

dataset = pd.read_csv(os.path.join(directories['intermediate_dir'], 'training_S2',directories['csv_file_patches90-100']))
dataset = dataset.loc[dataset['impair_idx'] == 'a']
dataset = dataset[dataset.duplicated(['im_patch_idx'], keep=False)]
dataset = dataset.loc[dataset['patchpair_idx'] == 0]
# ====================================================================

# np.random.seed(234)
# dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
# =============================================================================

np.random.seed(234)
random_image = np.random.choice(dataset['im_idx'], 1, replace=False)
dataset = dataset[dataset['im_idx'] == random_image[0]]['im_patch_idx'].values

#random_pair = [0]




# =============================================================================
# np.random.seed(234)
# dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
# =============================================================================

dataset_settings['indices_train'] = np.repeat(dataset, 25)
dataset_settings['indices_val'] = dataset
dataset_settings['indices_test'] = dataset
dataset_settings['indices_patch2'] = dataset
#dataset = np.random.choice(dataset['im_idx'], 1000, replace=False)
#dataset = dataset['im_idx'].values

# =============================================================================
# dataset_settings['indices_train'] = dataset[:int(dataset_settings['perc_train']*len(dataset))]
# dataset_settings['indices_val'] = dataset[int(dataset_settings['perc_train']*len(dataset)):-100]
# dataset_settings['indices_test'] = dataset[-100:]
# =============================================================================

# train
train(directories, dataset_settings, network_settings, train_settings)

#%% 
""" Train on S21C dataset"""
from train import train

directories['data_path'] = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
#dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

dataset = np.random.choice(dataset['im_idx'], 1, replace=False)
dataset_settings['indices_train'] = np.repeat(dataset[0], 1000)
dataset_settings['indices_val'] = np.repeat(dataset[0], 25)
dataset_settings['indices_test'] = np.repeat(dataset[0], 25)

# =============================================================================
# #dataset = np.random.choice(dataset['im_idx'], 1000, replace=False)
# #dataset = dataset['im_idx'].values
# 
# dataset_settings['indices_train'] = dataset[:int(dataset_settings['perc_train']*len(dataset))]
# dataset_settings['indices_val'] = dataset[int(dataset_settings['perc_train']*len(dataset)):-100]
# dataset_settings['indices_test'] = dataset[-100:]
# =============================================================================

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Train on image of OSCD dataset """
from train import train

directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
# =============================================================================
# np.random.seed(234)
# dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
# =============================================================================
np.random.seed(234)
#dataset = np.random.choice(dataset['im_idx'], 1, replace=False)
dataset = [0] # select one specific image
dataset_settings['indices_train'] = np.repeat(dataset[0], 1000)
dataset_settings['indices_val'] = np.repeat(dataset[0], 25)
dataset_settings['indices_test'] = np.repeat(dataset[0], 25)

# train
train(directories, dataset_settings, network_settings, train_settings)


#%%
""" Extract Features """

from extract_features import extract_features, calculate_distancemap, calculate_changemap
from metrics import compute_confusion_matrix, compute_matthews_corrcoef
import matplotlib.pyplot as plt
from plots import normalize2plot

# get csv files
oscd_train = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_train_oscd']))
oscd_test = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_test_oscd']))
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], directories['csv_models']))

#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1]
extract_layers = [1,4,7]

# get image
np.random.seed(789)
randim = np.random.choice(oscd_train.im_idx.values)
#randim = dataset[0]
randim_a = str(randim)+'_a.npy'
randim_b = str(randim)+'_b.npy'
im_a = np.load(os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd'],randim_a))
im_b = np.load(os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd'],randim_b))

# extract features
features = extract_features(directories=directories, imagelist=[im_a, im_b], model_settings=model_settings,  layers=extract_layers)
assert(features[0].shape == features[1].shape)

# calculate change map
distmap = calculate_distancemap(features[0], features[1])
changemap = calculate_changemap(distmap, plot=True)

# get labels
label = str(randim)+'.npy'
gt = np.load(os.path.join(directories['intermediate_dir_cd'], directories['labels_dir_oscd'],label))

fig1, axes1 = plt.subplots(ncols=3, figsize=(10, 5))
ax = axes1.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3)

ax[0].imshow(normalize2plot(im_a[:,:,[3,2,1]]))
ax[0].set_title('Image a')
ax[0].axis('off')

ax[1].imshow(normalize2plot(im_b[:,:,[3,2,1]]))
ax[1].set_title('Image b')
ax[1].axis('off')

ax[2].imshow(gt, cmap=plt.cm.gray)
ax[2].set_title('Ground truth')
ax[2].axis('off')
plt.show() 

fig2, axes2 = plt.subplots(ncols=2, figsize=(10, 5))
ax = axes2.ravel()
ax[0] = plt.subplot(1, 2, 1)
ax[1] = plt.subplot(1, 2, 2)

ax[0].imshow(changemap, cmap=plt.cm.gray)
ax[0].set_title('Change map')
#ax[0].axis('off')

ax[1].imshow(gt, cmap=plt.cm.gray)
ax[1].set_title('Ground truth')
#ax[1].axis('off')
plt.show() 

cm, fig3, axes3 = compute_confusion_matrix(gt, changemap, normalize=False)
mcc = compute_matthews_corrcoef(gt, changemap)
print("mcc: ", mcc)

tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]

sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
precision = tp/(tp+fp) # prop of changed pixels that are truly changed

print("sensitivity: ", sensitivity)
print("specificity: ", specificity)
print("recall: ", sensitivity)
print("precision: ", precision)