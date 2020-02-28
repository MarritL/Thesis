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
    'weight_decay':0,
    'loss': 'bce_sigmoid',
    'n_classes': 2,
    'patch_size': 96,
    'batch_norm': True,
    'weights_file': '' }

train_settings = {
    'start_epoch': 0,
    'num_epoch': 20,
    'batch_size': 25,
    'disp_iter': 10,
    'gpu': None}

dataset_settings = {
    'dataset_type' : 'triplet_saved',
    'perc_train': 0.7,
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
#dataset = dataset.loc[dataset['patchpair_idx'] == 0]

# train on the complete dataset
# =============================================================================
# np.random.seed(234)
# dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
# =============================================================================

# train on only 1 image
# =============================================================================
# np.random.seed(234)
# random_image = np.random.choice(dataset['im_idx'], 1, replace=False)
# dataset = dataset[dataset['im_idx'] == random_image[0]]['im_patch_idx'].values
# 
# =============================================================================

# train on image 0 only
#random_pair = [0]

# for 1 image only
# =============================================================================
# dataset_settings['indices_train'] = np.repeat(dataset, 25)
# dataset_settings['indices_val'] = dataset
# dataset_settings['indices_test'] = dataset
# dataset_settings['indices_patch2'] = dataset
# =============================================================================

# train on n images (part kept apart for val and test)
n = 25
n_test = 4
np.random.seed(234)
random_im_idx = np.random.choice(dataset['im_idx'], n, replace=False)
# defide over train val and test
train_im_idx = random_im_idx[:int(dataset_settings['perc_train']*len(random_im_idx))]
val_im_idx = random_im_idx[int(dataset_settings['perc_train']*len(random_im_idx)):-n_test]
test_im_idx = random_im_idx[-n_test:]
# get the patch_indices
dataset_settings['indices_train'] = dataset[dataset['im_idx'].isin(train_im_idx)]['im_patch_idx'].values
dataset_settings['indices_val'] = dataset[dataset['im_idx'].isin(val_im_idx)]['im_patch_idx'].values
dataset_settings['indices_test'] = dataset[dataset['im_idx'].isin(test_im_idx)]['im_patch_idx'].values

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

# continue training
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], directories['csv_models']))
model_settings = trained_models.iloc[24]
network_settings['weights_file'] = model_settings['filename']

# train
train(directories, dataset_settings, network_settings, train_settings)


#%%
""" Extract Features """

from extract_features import extract_features, calculate_distancemap, calculate_changemap
from metrics import compute_confusion_matrix, compute_matthews_corrcoef
from plots import plot_imagepair_plus_gt, plot_changemap_plus_gt

# get csv files
oscd_train = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_train_oscd']))
oscd_test = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_test_oscd']))
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], directories['csv_models']))

#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[24]
print("MODEL SETTINGS: \n", model_settings)
extract_layers = None # for manual setting which layers to use

if extract_layers == None:
    # cfgs are saved as strings, cast back to list
    branch = model_settings['cfg_branch'].split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    branch = branch.replace(" ", "")
    branch = branch.split(",")
    
    # extract the features from all activation layers
    extract_layers = list()
    layer_i = -1
    if model_settings['batch_norm']:
        for layer in branch:
            if layer == 'M':
                layer_i += 1
            else:
                layer_i += 3
                extract_layers.append(layer_i)
    else:
        for layer in branch:
            if layer == 'M':
                layer_i += 1
            else:
                layer_i += 2
                extract_layers.append(layer_i)
                   

# get image
np.random.seed(789)
randim = np.random.choice(oscd_train.im_idx.values)
randim = 0
randim_a = str(randim)+'_a.npy'
randim_b = str(randim)+'_b.npy'
im_a = np.load(os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd'],randim_a))
im_b = np.load(os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd'],randim_b))

# extract features
features = extract_features(
    directories=directories, 
    imagelist=[im_a, im_b], 
    model_settings=model_settings,  
    layers=extract_layers)

assert(features[0].shape == features[1].shape)
#plt.imshow(features[0][:,:,40], cmap=plt.cm.gray)

# calculate change map
distmap = calculate_distancemap(features[0], features[1])
changemap = calculate_changemap(distmap, plot=True)

# get labels
label = str(randim)+'.npy'
gt = np.load(os.path.join(directories['intermediate_dir_cd'], directories['labels_dir_oscd'],label))

#plots
fig1, ax1 = plot_imagepair_plus_gt(im_a, im_b, gt)  
fig2, ax2 = plot_changemap_plus_gt(changemap, gt, axis=True)

# confusion matrix
cm, fig3, axes3 = compute_confusion_matrix(gt, changemap, normalize=False)

# compute metrics
mcc = compute_matthews_corrcoef(gt, changemap)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
precision = tp/(tp+fp) # prop of changed pixels that are truly changed

# print metrics
print("mcc: ", mcc)
print("sensitivity: ", sensitivity)
print("specificity: ", specificity)
print("recall: ", sensitivity)
print("precision: ", precision)