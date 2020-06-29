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
        'intermediate_dir_training': '/media/cordolo/marrit/Intermediate/training_S2',
        'results_dir_training': '/media/cordolo/marrit/results/training_S2',
        'results_dir_cd': '/media/cordolo/marrit/results/CD_OSCD',
        'results_dir': '/media/cordolo/marrit/results',
        'intermediate_dir_cd': '/media/cordolo/marrit/Intermediate/CD_OSCD',
        'intermediate_dir': '/media/cordolo/marrit/Intermediate',
        'source_dir_cd': '/media/cordolo/marrit/source_data/CD_OSCD',
        'csv_file_S21C': 'S21C_dataset.csv',
        'csv_file_oscd': 'OSCD_dataset.csv',
        'csv_file_train_oscd' : 'OSCD_train.csv',
        'csv_file_test_oscd': 'OSCD_test.csv',
        'csv_models': 'trained_models.csv',
        'csv_models_finetune': 'trained_models_finetune.csv',
        'csv_models_downstream': 'trained_model_downstream.csv',
        'csv_models_results': 'models_results.csv',
        'data_dir_S21C': 'data_S21C',
        'data_dir_oscd': 'data_OSCD',
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
        'results_dir': '/marrit1/results',
        'intermediate_dir_cd': '/marrit1/Intermediate/CD_OSCD',
        'intermediate_dir': '/marrit1/Intermediate',
        'csv_file_S21C': 'S21C_dataset.csv',
        'csv_file_oscd': 'OSCD_dataset.csv',
        'csv_file_train_oscd' : 'OSCD_train.csv',
        'csv_file_test_oscd': 'OSCD_test.csv',
        'csv_models': 'trained_models.csv',
        'csv_models_finetune': 'trained_models_finetune.csv',
        'csv_models_downstream': 'trained_model_downstream.csv',
        'csv_models_results': 'models_results.csv',
        'data_dir_S21C': 'data_S21C',
        'data_dir_oscd': 'data_OSCD',
        'labels_dir_oscd' : 'labels_OSCD',
        'tb_dir': '/marrit1/Intermediate/tensorboard',
        'model_dir': '/marrit1/Intermediate/trained_models',
        'model_dir_finetune': '/marrit1/Intermediate/trained_models_finetune',
        'model_dir_downstream': '/marrit1/Intermediate/trained_models_downstream'}

network_settings = {
    'network': 'triplet_apn', # siamese" | "triplet" | "hypercolumn" | "triplet_apn" | siamese_unet | "siamese_dilated" | "triplet_unet" | "siamese_concat" | "single"| "logistic_regression"
    'optimizer': 'adam',
    'lr':1e-5,
    'weight_decay': 1e-4,
    'loss': 'nll_finetune', # "cross_entropy" | "bce_sigmoid" | "nll" | "nll_finetune" | "triplet" | "l1+triplet" | "l1+triplet+bce" | "mse" 
    'n_classes': 2,
    'patch_size': 96,
    'im_size': (96,96),
    'batch_norm': True,
    'weights_file': '',
    'extract_features': None,
    'avg_pool': False}


### network cofiguration ###
# number denotes number of output channels of conv layer or linear layer
# 'M' denotes max-pooling layer (2x2), stride=2
# 'D' denotes drop-outlayer (only in classifier for siamese/triplet network)
# classifier in siamese/triplet network uses linear layers as default. If
#    you want conv layers, start classifier array with 'C'.
#    e.g. 'classifier': np.array(['C',network_settings['n_classes']])
# classifier in hypercolumn network uses conv layers

cfg = {'branch': np.array([32,32,32], dtype='object'), 
       'top': np.array([32], dtype='object'),
       'classifier': np.array([128,64,network_settings['n_classes']])}

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
    'num_epoch':300,
    'batch_size': 25,
    'disp_iter': 25,
    'gpu': 0,
    'early_stopping': 50}

dataset_settings = {
    'dataset_type' : 'triplet_apn', # "pair" | "triplet" | "triplet_saved" | "overlap" | "triplet_apn" | "overlap_regression" | "pair_hard_neg" | "triplet_apn_hard_neg" | "supervised" | "supervised_from_file"
    'perc_train': 0.85,
    'channels': np.arange(13),
    'min_overlap': 1, 
    'max_overlap': 1,
    'stride': int(network_settings['patch_size']),
    'patches_per_image': 100}


#%% 
""" Train on S21C dataset"""
from train import train

directories['data_path'] = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

# datasplits
dataset_settings['indices_train'] = np.repeat(dataset[:-250], dataset_settings['patches_per_image'])
dataset_settings['indices_val'] = np.repeat(dataset[-250:-100],dataset_settings['patches_per_image'])
dataset_settings['indices_test'] = np.repeat(dataset[-100:],dataset_settings['patches_per_image'])
dataset_settings['indices_eval'] = np.repeat(dataset[-250:-100][-20:],dataset_settings['patches_per_image'])

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Train on image of OSCD dataset """
from train import train

directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

directories['labels_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['labels_dir_oscd'])

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(567)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

k = 2 # 'full' for complete training set
if k != 'full':
    folds = 3
    assert k < folds
    ims_per_fold = int(len(dataset[:-2])/folds)
    # train / val split  
    val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
    train_indices = dataset[np.isin(dataset, val_indices) == False][:-2]
else:
    train_indices = dataset[:-2]
    val_indices = dataset[:-2]

dataset_settings['indices_train'] = np.repeat(train_indices,dataset_settings['patches_per_image'])
dataset_settings['indices_val'] = np.repeat(val_indices, dataset_settings['patches_per_image'])
dataset_settings['indices_test'] = np.repeat(dataset[-2:],dataset_settings['patches_per_image'])
dataset_settings['dataset_type'] = 'supervised'
dataset_settings['kthfold'] = k

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Train on OSCD-dataset, fixed patches """
from train import train
directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

directories['labels_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['labels_dir_oscd'])

dataset_train = pd.read_csv(os.path.join(directories['results_dir'],'train_patches_oscd.csv'))
indices = np.unique(dataset_train['im_idx'].values)

# shuffle
np.random.seed(567)
dataset = np.random.choice(indices, len(indices), replace=False)

# splits for cross-validation
k = 2 # 'full' for complete training set | 0, ... , n
folds = 3
if k != 'full':
    assert k < folds
    ims_per_fold = int(len(dataset)/folds)
    # train / val split  
    val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
    train_indices = dataset[np.isin(dataset, val_indices) == False]
else:
    train_indices = dataset
    val_indices = dataset

dataset_settings['dataset_train_df'] = dataset_train[np.isin(dataset_train['im_idx'], train_indices)]
dataset_val_df = dataset_train[np.isin(dataset_train['im_idx'], val_indices)]
dataset_settings['dataset_val_df'] = dataset_val_df[dataset_val_df.variant == 0]
dataset_settings['indices_train'] = dataset_settings['dataset_train_df'].im_idx.values
dataset_settings['indices_val'] = dataset_settings['dataset_val_df'].im_idx.values
dataset_settings['dataset_type'] = 'supervised_from_file'
dataset_settings['kthfold'] = k

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Supervised finetune on OSCD-dataset, random patches """
from train import finetune
# network
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
# get model
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

directories['labels_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['labels_dir_oscd'])

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(567)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

k = 2 # 'full' for complete training set
if k != 'full':
    folds = 3
    assert k < folds
    ims_per_fold = int(len(dataset[:-2])/folds)
    # train / val split  
    val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
    train_indices = dataset[np.isin(dataset, val_indices) == False][:-2]
else:
    train_indices = dataset[:-2]
    val_indices = dataset[:-2]

dataset_settings['indices_train'] = np.repeat(train_indices,dataset_settings['patches_per_image'])
dataset_settings['indices_val'] = np.repeat(val_indices, dataset_settings['patches_per_image'])
dataset_settings['indices_test'] = np.repeat(dataset[-2:],dataset_settings['patches_per_image'])
dataset_settings['dataset_type'] = 'supervised_from_file'
dataset_settings['kthfold'] = k

# change learning rate
network_settings['lr'] = 1e-2

finetune(model_settings, directories, dataset_settings, network_settings, train_settings)

#%%
""" Supervised finetuning on OSCD-dataset, fixed patches"""
from train import use_features_downstream
# network
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
# get model
model_idx = 0
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

directories['labels_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['labels_dir_oscd'])

dataset_train = pd.read_csv(os.path.join(directories['results_dir'],'train_patches_oscd.csv'))
indices = np.unique(dataset_train['im_idx'].values)

# shuffle
np.random.seed(567)
dataset = np.random.choice(indices, len(indices), replace=False)

# splits for cross-validation
k = 0 # 'full' for complete training set | 0, ... , n
folds = 3
if k != 'full':
    assert k < folds
    ims_per_fold = int(len(dataset)/folds)
    # train / val split  
    val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
    train_indices = dataset[np.isin(dataset, val_indices) == False]
else:
    train_indices = dataset
    val_indices = dataset

dataset_settings['dataset_train_df'] = dataset_train[np.isin(dataset_train['im_idx'], train_indices)]
dataset_val_df = dataset_train[np.isin(dataset_train['im_idx'], val_indices)]
dataset_settings['dataset_val_df'] = dataset_val_df[dataset_val_df.variant == 0]
dataset_settings['indices_train'] = dataset_settings['dataset_train_df'].im_idx.values
dataset_settings['indices_val'] = dataset_settings['dataset_val_df'].im_idx.values
dataset_settings['dataset_type'] = 'supervised_from_file'
dataset_settings['kthfold'] = k

network_settings['layers_branches'] = 6
network_settings['layers_joint'] = 0
network_settings['cfg']['classifier_cd'] = np.array(['C', network_settings['n_classes']])

use_features_downstream(model_settings, directories, dataset_settings, network_settings, train_settings, finetune=True)

#%%
""" Train supervised classifier on OSCD-dataset, fixed patches"""
from train import use_features_downstream
# network
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
# get model
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

directories['labels_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['labels_dir_oscd'])

dataset_train = pd.read_csv(os.path.join(directories['results_dir'],'train_patches_oscd.csv'))
indices = np.unique(dataset_train['im_idx'].values)

# shuffle
np.random.seed(567)
dataset = np.random.choice(indices, len(indices), replace=False)

# splits for cross-validation
for k in [1,2,3]: # 'full' for complete training set | 0, ... , n
    folds = 3
    if k != 'full':
        assert k < folds
        ims_per_fold = int(len(dataset)/folds)
        # train / val split  
        val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
        train_indices = dataset[np.isin(dataset, val_indices) == False]
    else:
        train_indices = dataset
        val_indices = dataset
    
    dataset_settings['dataset_train_df'] = dataset_train[np.isin(dataset_train['im_idx'], train_indices)]
    dataset_settings['dataset_train_df'] = dataset_settings['dataset_train_df'][dataset_settings['dataset_train_df'].variant == 0]
    dataset_val_df = dataset_train[np.isin(dataset_train['im_idx'], val_indices)]
    dataset_settings['dataset_val_df'] = dataset_val_df[dataset_val_df.variant == 0]
    dataset_settings['indices_train'] = dataset_settings['dataset_train_df'].im_idx.values
    dataset_settings['indices_val'] = dataset_settings['dataset_val_df'].im_idx.values
    dataset_settings['dataset_type'] = 'supervised_from_file'
    dataset_settings['kthfold'] = k
    
    network_settings['cfg']['classifier_cd'] = np.array(['C',network_settings['n_classes']])
    network_settings['concat'] = False
    network_settings['layers_branches'] = 6    
    network_settings['layers_joint'] = 0
            
            
    use_features_downstream(model_settings, directories, dataset_settings, network_settings, train_settings, finetune=False)

#%%
""" Plot training history """
import torch
import matplotlib.pyplot as plt

# get model files
models_key = 'csv_models_downstream' # 'csv_models' |'csv_models_finetune' | 'csv_models_downstream' 
models_dir_key = 'model_dir_downstream'
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]))
# get model info
model_idx = -1  # TODO: change here the model
model_settings = trained_models.iloc[model_idx] 
print("MODEL SETTINGS: \n", model_settings)

# get history
history_file = os.path.join(directories[models_dir_key], 'history_' + model_settings['filename'].split('/')[-1])
history = torch.load(history_file)

history_loss = pd.DataFrame(history['train']['loss'], columns= ['train'], index=history['train']['epoch'])
history_loss['val'] = history['val']['loss']
history_acc = pd.DataFrame(history['train']['acc'], columns= ['train'], index=history['train']['epoch'])
history_acc['val'] = history['val']['acc']

fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
history_loss.plot(ax=ax[1])
history_acc.plot(ax=ax[0])
ax[1].set_title('loss')
ax[1].set_xlabel('epoch')
ax[0].set_title('accuracy')
ax[0].set_xlabel('epoch')
fig.suptitle(model_settings['filename'].split('/')[-1], fontsize=18)
plt.show()
#%%
""" Plot training history multiple cross-validation models"""
import torch
import matplotlib.pyplot as plt

# get model files
models_key = 'csv_models_downstream' # 'csv_models' |'csv_models_finetune' | 'csv_models_downstream' 
models_dir_key = 'model_dir_downstream'
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]))
# get model info
history_loss = pd.DataFrame(data=None, index=None, columns=[['k=0', 'k=1', 'k=2']])
history_acc = pd.DataFrame(data=None, index=None, columns=[['k=0', 'k=1', 'k=2']])
for i, idx in enumerate([-3,-2,-1]):
    model_idx = idx # TODO: change here the model
    model_settings = trained_models.iloc[model_idx] 
    print("MODEL SETTINGS: \n", model_settings)
    
    # get history
    history_file = os.path.join(directories[models_dir_key], 'history_' + model_settings['filename'].split('/')[-1])
    history = torch.load(history_file)
    
    if i == 0:
        history_loss = pd.DataFrame(history['train']['loss'], columns= ['k='+str(i)], index=history['train']['epoch'])
        history_acc = pd.DataFrame(history['train']['acc'], columns= ['k='+str(i)], index=history['train']['epoch'])
    else:
        history_loss['k='+str(i)] = pd.DataFrame(history['train']['loss'], index=history['train']['epoch'])
        history_acc['k='+str(i)] = pd.DataFrame(history['train']['acc'], index=history['train']['epoch'])
    #history_loss['train k='+str(i)] = history['train']['loss']    
    #history_acc['train k='+str(i)] = history['train']['acc']

fig, ax = plt.subplots(ncols=2, figsize=(25, 10))
plt.rcParams.update( {'legend.fontsize': 24})
lo = history_loss.iloc[:].plot(ax=ax[1], fontsize=24)
history_acc.iloc[:].plot(ax=ax[0], fontsize=24)
ax[1].set_ylabel('Loss',fontsize=24)
ax[1].set_xlabel('Epoch',fontsize=24)
ax[0].set_ylabel('AA (%)',fontsize=24)
ax[0].set_xlabel('Epoch',fontsize=24)
#ax.set_ylim((-4,4))
for tick in ax[0].xaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
for tick in ax[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
#add major gridlines
ax[0].grid(color='grey', linestyle='-', linewidth=0.25)
ax[1].grid(color='grey', linestyle='-', linewidth=0.25)
#fig.suptitle(model_settings['filename'].split('/')[-1], fontsize=18)
plt.show()

#%%
""" Evaluate (preliminary) network on pretext task (overlap yes/no) S21C dataset """
from train import evaluate

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
# get model info
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get dataset
directories['data_path'] = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset_eval = pd.read_csv(os.path.join(directories['results_dir'],'eval_patches.csv'))
dataset_settings['dataset_type'] = 'triplet_from_file'
dataset_settings['indices_eval'] = dataset_eval['im_idx'].values
dataset_settings['dataset_eval_df'] = dataset_eval

# evaluate
acc, loss, probability = evaluate(model_settings, directories, dataset_settings, network_settings, train_settings)
print('accuracy: ', acc)
print('loss: ', loss)
print('probability: ', probability)

# save
trained_models.loc[trained_models.index[model_idx],'n_eval_patches'] = len(dataset_settings['indices_eval'])
trained_models.loc[trained_models.index[model_idx],'eval_acc'] = acc
trained_models.loc[trained_models.index[model_idx],'eval_loss'] = loss
trained_models.loc[trained_models.index[model_idx],'eval_prob'] = probability
trained_models.to_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']),index=False)
#%%
""" Evaluate (preliminary) network on pretext task (overlap yes/no) OSCD dataset """
from train import evaluate

models_key = 'csv_models' # "csv_models" | "csv_models_finetune"
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]))
# get model info
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get dataset
directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

dataset_eval = pd.read_csv(os.path.join(directories['results_dir'],'eval_patches_oscd.csv'))
dataset_settings['dataset_type'] = 'triplet_from_file'
dataset_settings['indices_eval'] = dataset_eval['im_idx'].values
dataset_settings['dataset_eval_df'] = dataset_eval

# evaluate
acc, loss, probability = evaluate(model_settings, directories, dataset_settings, network_settings, train_settings)
print('accuracy: ', acc)
print('loss: ', loss)
print('probability: ', probability)

# save
trained_models.loc[trained_models.index[model_idx],'n_eval_patches_oscd'] = len(dataset_settings['indices_eval'])
trained_models.loc[trained_models.index[model_idx],'eval_acc_oscd'] = acc
trained_models.loc[trained_models.index[model_idx],'eval_loss_oscd'] = loss
trained_models.loc[trained_models.index[model_idx],'eval_prob_oscd'] = probability
trained_models.to_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]),index=False)

#%%
""" Evaluate (preliminary) network on pretext task (APN) S21C dataset """
from train import evaluate_apn

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
# get model info
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get dataset
directories['data_path'] = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset_eval = pd.read_csv(os.path.join(directories['results_dir'],'eval_patches.csv'))
dataset_settings['dataset_type'] = 'triplet_apn_from_file'
dataset_settings['indices_eval'] = dataset_eval['im_idx'].values
dataset_settings['dataset_eval_df'] = dataset_eval

# evaluate
acc, loss = evaluate_apn(model_settings, directories, dataset_settings, network_settings, train_settings)
print('accuracy: ', acc)
print('loss: ', loss)


# save
trained_models.loc[trained_models.index[model_idx],'n_eval_patches'] = len(dataset_settings['indices_eval'])
trained_models.loc[trained_models.index[model_idx],'eval_acc'] = acc
trained_models.loc[trained_models.index[model_idx],'eval_loss'] = loss
trained_models.to_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']),index=False)
#%%
""" Evaluate (preliminary) network on pretext task (APN) OSCD dataset """
from train import evaluate_apn

models_key = 'csv_models' # "csv_models" | "csv_models_finetune"
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]))
# get model info
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get dataset
directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

dataset_eval = pd.read_csv(os.path.join(directories['results_dir'],'eval_patches_oscd.csv'))
dataset_settings['dataset_type'] = 'triplet_apn_from_file'
dataset_settings['indices_eval'] = dataset_eval['im_idx'].values
dataset_settings['dataset_eval_df'] = dataset_eval

# evaluate
acc, loss = evaluate_apn(model_settings, directories, dataset_settings, network_settings, train_settings)
print('accuracy: ', acc)
print('loss: ', loss)

# save
trained_models.loc[trained_models.index[model_idx],'n_eval_patches_oscd'] = len(dataset_settings['indices_eval'])
trained_models.loc[trained_models.index[model_idx],'eval_acc_oscd'] = acc
trained_models.loc[trained_models.index[model_idx],'eval_loss_oscd'] = loss
trained_models.to_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]),index=False)


#%%
""" Evaluate (preliminary) network on change detection task, using CVA classifier """
from change_detection import detect_changes, detect_changes_no_gt
import csv

models_key = 'csv_models' # "csv_models" | "csv_models_finetune"
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]))
# get model info
model_idx = -1
model_settings = trained_models.iloc[model_idx] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get dataset
directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

directories['labels_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['labels_dir_oscd'])

dataset_to_use = 'test'
if dataset_to_use == 'train':
    dataset_train = pd.read_csv(os.path.join(directories['results_dir'],'train_patches_oscd.csv'))
    indices = np.unique(dataset_train['im_idx'].values)
    # shuffle
    np.random.seed(567)
    dataset = np.random.choice(indices, len(indices), replace=False)
    dataset_settings['indices_eval'] = dataset
else:
    dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
    dataset = dataset.loc[dataset['pair_idx'] == 'a']
    np.random.seed(567)
    dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
    dataset_settings['indices_test'] = dataset[-2:]
    dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_test_oscd']))
    dataset = dataset.loc[dataset['pair_idx'] == 'a']
    np.random.seed(567)
    dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
    dataset_settings['indices_test'] = np.concatenate((dataset_settings['indices_test'],dataset))

# set extraction layer
network_settings['extract_features'] = [5]
network_settings['str_extract_features'] = '5'

methods = ['triangle'] # "otsu" | "triangle" 

if dataset_to_use == 'train':
    tp, tn, fp, fn, thresholds = detect_changes(
        model_settings, 
        directories, 
        dataset_settings, 
        network_settings, 
        train_settings, 
        threshold_methods=methods)
    fieldnames = ['metric','method','extract_features','19', '17', '10', '14', '2', '1', '18', '21', '0', '15', '5', '8']
else:
    thresholds_list1 = {}
    thresholds_list3 = {}
    tp, tn, fp, fn, thresholds = detect_changes_no_gt(
        model_settings, 
        directories, 
        dataset_settings, 
        network_settings, 
        train_settings, 
        threshold_methods=methods, 
        threshold_from_file=thresholds_list1[network_settings['str_extract_features']])
    fieldnames = ['metric','method','extract_features','3', '4', '13', '9', '22', '6', '7','23', '16', '20', '11', '12']
# init save-file

save_networkname = model_settings['filename'].split('/')[-1]
if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname, 'cva_results_'+dataset_to_use+'.csv' )):
    with open(os.path.join(directories['results_dir_cd'], save_networkname, 'cva_results_'+dataset_to_use+'.csv' ), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()
        
fieldnames2 = ['method','images','extract_features', 'sensitivity', 'specificity', 'recall', 'precision', 'F1', 'avg_acc', 'acc', 'avg_threshold']
save_networkname = model_settings['filename'].split('/')[-1]
if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname, 'cva_results_avg_'+dataset_to_use+'.csv' )):
    with open(os.path.join(directories['results_dir_cd'], save_networkname, 'cva_results_avg_'+dataset_to_use+'.csv' ), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames2, delimiter = ",")
        filewriter.writeheader()

for method in methods:
    tp_temp = sum(tp[method].values())
    tn_temp = sum(tn[method].values())
    fp_temp = sum(fp[method].values())
    fn_temp = sum(fn[method].values())
   
    sensitivity = tp_temp/(tp_temp+fn_temp) # prop of correctly identified changed pixels = recall
    recall = sensitivity
    specificity = tn_temp/(tn_temp+fp_temp) # prop of correctly identified unchanged pixels
    precision = tp_temp/(tp_temp+fp_temp) # prop of changed pixels that are truly changed
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp_temp+tn_temp)/(tp_temp+tn_temp+fp_temp+tn_temp)
    avg_acc = (sensitivity+specificity)/2
    avg_threshold = sum(thresholds[method].values())/len(thresholds[method].keys())
    print('------------------------------')
    print("RESULTS METHOD: {}:".format(method))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)   
    print("avg_acc", avg_acc)
    print("avg thresholds", avg_threshold)
    print('------------------------------')
    
    if network_settings['extract_features'] == None:
        network_settings['extract_features'] = 'None'
    tp[method]['metric'] = 'tp'
    tp[method]['method'] = method
    tp[method]['extract_features'] = str(network_settings['extract_features'])
    tn[method]['metric'] = 'tn'
    tn[method]['method'] = method
    tn[method]['extract_features'] = str(network_settings['extract_features'])
    fp[method]['metric'] = 'fp'
    fp[method]['method'] = method
    fp[method]['extract_features'] = str(network_settings['extract_features'])
    fn[method]['metric'] = 'fn'
    fn[method]['method'] = method
    fn[method]['extract_features'] = str(network_settings['extract_features'])
    thresholds[method]['metric'] = 'threshold'
    thresholds[method]['method'] = method
    thresholds[method]['extract_features'] = str(network_settings['extract_features'])
    
    with open(os.path.join(directories['results_dir_cd'], save_networkname, 'cva_results_'+dataset_to_use+'.csv' ), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writerow(tp[method])  
            filewriter.writerow(tn[method])  
            filewriter.writerow(fp[method])  
            filewriter.writerow(fn[method])  
            filewriter.writerow(thresholds[method])

    with open(os.path.join(directories['results_dir_cd'], save_networkname, 'cva_results_avg_'+dataset_to_use+'.csv' ), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames2, delimiter = ",")
        filewriter.writerow({'method': method,
                             'images':str(list(tp[method].keys())[:-2]), 
                             'sensitivity': sensitivity, 
                             'specificity':specificity, 
                             'recall':recall, 
                             'precision':precision, 
                             'F1':F1, 
                             'avg_acc':avg_acc, 
                             'acc':acc, 
                             'avg_threshold':avg_threshold,
                             'extract_features': str(network_settings['extract_features'])})


#%%
""" Evalutate downstream model on OSCD dataset, using 1-layer conv. classifier or 2-layer conv. classifier """
from inference import inference, find_best_threshold, apply_thresholds_from_probmap
import csv

extract_features = None
# get model files
models_key = 'csv_models_downstream' # 'csv_models' |'csv_models_finetune' | 'csv_models_downstream' 
models_dir_key = 'model_dir_downstream'
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories[models_key]))
#model_idx = 40 # TODO: change here the model
model_indices = [24]
for model_idx in model_indices:
    model_settings = trained_models.iloc[model_idx] 
    print("MODEL SETTINGS: \n", model_settings.filename)
# =============================================================================
#     # =============================================================================
#     model_settings['kthfold'] = 'full'
#     model_settings['pretask_filename'] = 'from_scratch'
#     model_settings['layers_branches'] = 'None'
#     model_settings['layers_joint'] = 'None'
#     model_settings['cfg_classifier_cd'] = 'CVA'
#     # =============================================================================
# =============================================================================
    
    # get dataset
    directories['data_path'] = os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'])
    
    directories['labels_path'] = os.path.join(
        directories['intermediate_dir_cd'], 
        directories['labels_dir_oscd'])
    
    # splits used in cross-validation
    k = model_settings['kthfold'] # 'full' for complete training set | 0, ... , n
    folds = 3
    if k != 'full': 
        k = int(k)
        assert k < folds
        # get data
        dataset_train = pd.read_csv(os.path.join(directories['results_dir'],'train_patches_oscd.csv'))
        indices = np.unique(dataset_train['im_idx'].values)
        # shuffle
        np.random.seed(567)
        dataset = np.random.choice(indices, len(indices), replace=False)
        ims_per_fold = int(len(dataset)/folds)
        # train / val split
        val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
        train_indices = dataset[np.isin(dataset, val_indices) == False]
        dataset_settings['indices_test'] = val_indices
        dataset_settings['dataset_to_use'] = 'trainset'
        fieldnames = ['metric','method','extract_features','19', '17', '10', '14', '2', '1', '18', '21', '0', '15', '5', '8']
    else:
        dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
        dataset = dataset.loc[dataset['pair_idx'] == 'a']
        np.random.seed(567)
        dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
        dataset_settings['indices_test'] = dataset[-2:]
        dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_test_oscd']))
        dataset = dataset.loc[dataset['pair_idx'] == 'a']
        np.random.seed(567)
        dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
        dataset_settings['indices_test'] = np.concatenate((dataset_settings['indices_test'],dataset))
        dataset_settings['dataset_to_use'] = 'testset'
        fieldnames = ['metric','method','extract_features','3', '4', '13', '9', '22', '6', '7','23', '16', '20', '11', '12']
        
    
    inference(directories, dataset_settings, model_settings, train_settings,extract_features=extract_features)
    if dataset_settings['dataset_to_use'] == 'trainset':
        thresholds_f1, f1s, recalls, precisions, thresholds_avg_acc, tnrs, tprs, avg_accs = find_best_threshold(
            directories, dataset_settings['indices_test'], model_settings)
 
for model_idx in model_indices:
    model_settings = trained_models.iloc[model_idx] 
    print("MODEL SETTINGS: \n", model_settings.filename)
# =============================================================================
#     # =============================================================================
#     model_settings['kthfold'] = 'full'
#     model_settings['pretask_filename'] = 'from_scratch'
#     model_settings['layers_branches'] = 'None'
#     model_settings['layers_joint'] = 'None'
#     model_settings['cfg_classifier_cd'] = 'CVA'
#     # =============================================================================
# =============================================================================
    
    # get dataset
    directories['data_path'] = os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'])
    
    directories['labels_path'] = os.path.join(
        directories['intermediate_dir_cd'], 
        directories['labels_dir_oscd'])
    
    # splits used in cross-validation
    k = model_settings['kthfold'] # 'full' for complete training set | 0, ... , n
    folds = 3
    if k != 'full': 
        k = int(k)
        assert k < folds
        # get data
        dataset_train = pd.read_csv(os.path.join(directories['results_dir'],'train_patches_oscd.csv'))
        indices = np.unique(dataset_train['im_idx'].values)
        # shuffle
        np.random.seed(567)
        dataset = np.random.choice(indices, len(indices), replace=False)
        ims_per_fold = int(len(dataset)/folds)
        # train / val split
        val_indices = dataset[k*ims_per_fold:(k+1)*ims_per_fold]
        train_indices = dataset[np.isin(dataset, val_indices) == False]
        dataset_settings['indices_test'] = val_indices
        dataset_settings['dataset_to_use'] = 'trainset'
        fieldnames = ['metric','method','extract_features','19', '17', '10', '14', '2', '1', '18', '21', '0', '15', '5', '8']
    else:
        dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
        dataset = dataset.loc[dataset['pair_idx'] == 'a']
        np.random.seed(567)
        dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
        dataset_settings['indices_test'] = dataset[-2:]
        dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_test_oscd']))
        dataset = dataset.loc[dataset['pair_idx'] == 'a']
        np.random.seed(567)
        dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
        dataset_settings['indices_test'] = np.concatenate((dataset_settings['indices_test'],dataset))
        dataset_settings['dataset_to_use'] = 'testset'
        fieldnames = ['metric','method','extract_features','3', '4', '13', '9', '22', '6', '7','23', '16', '20', '11', '12']
    # set extraction layer
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
    if os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname, 'best_thresholds.csv')): 
        threshold_save_file = pd.read_csv(os.path.join(directories['results_dir_cd'], save_pretaskname, 'best_thresholds.csv'))
        subset = threshold_save_file[threshold_save_file.layers_branches == model_settings.layers_branches]
        subset2 = subset[subset.layers_joint == model_settings.layers_joint]
        other_folds = subset2[subset2.kthfold != model_settings.kthfold]
        thres_f1 = np.mean(other_folds.threshold_f1)
        thres_aa = np.mean(other_folds.threshold_avg_acc)
    else:
        thres_f1 = 0
        thres_aa = 0
    
    methods = ['otsu', 'triangle','end-to-end', 'f1', 'AA'] # "otsu" | "triangle" | "end-to-end" | "f1" | "AA"
    
    tp, tn, fp, fn, thresholds = apply_thresholds_from_probmap(
        model_settings, 
        directories, 
        dataset_settings, 
        network_settings, 
        train_settings, 
        threshold_methods=methods, 
        threshold_from_file={'f1': thres_f1, 'AA': thres_aa})
    # init save-file
    
    save_networkname = model_settings['filename'].split('/')[-1]
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier, 'results.csv' )):
        with open(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier, 'results.csv'), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()
            
    fieldnames2 = ['method','images','extract_features', 'sensitivity', 'specificity', 'recall', 'precision', 'F1', 'avg_acc', 'acc', 'avg_threshold']
    save_networkname = model_settings['filename'].split('/')[-1]
    if not os.path.exists(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier, 'results_avg.csv' )):
        with open(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier, 'results_avg.csv' ), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames2, delimiter = ",")
            filewriter.writeheader()
    
    for method in methods:
        tp_temp = sum(tp[method].values())
        tn_temp = sum(tn[method].values())
        fp_temp = sum(fp[method].values())
        fn_temp = sum(fn[method].values())
       
        sensitivity = tp_temp/(tp_temp+fn_temp) # prop of correctly identified changed pixels = recall
        recall = sensitivity
        specificity = tn_temp/(tn_temp+fp_temp) # prop of correctly identified unchanged pixels
        precision = tp_temp/(tp_temp+fp_temp) # prop of changed pixels that are truly changed
        F1 = 2 * (precision * recall) / (precision + recall)
        acc = (tp_temp+tn_temp)/(tp_temp+tn_temp+fp_temp+tn_temp)
        avg_acc = (sensitivity+specificity)/2
        avg_threshold = sum(thresholds[method].values())/len(thresholds[method].keys())
        print('------------------------------')
        print("RESULTS METHOD: {}:".format(method))
        print("sensitivity: ", sensitivity)
        print("specificity: ", specificity)
        print("recall: ", sensitivity)
        print("precision: ", precision)
        print("F1: ", F1)
        print("acc: ", acc)   
        print("avg_acc", avg_acc)
        print("avg thresholds", avg_threshold)
        print('------------------------------')
        
        if network_settings['extract_features'] == None:
            network_settings['extract_features'] = 'None'
        tp[method]['metric'] = 'tp'
        tp[method]['method'] = method
        tp[method]['extract_features'] = str(network_settings['extract_features'])
        tn[method]['metric'] = 'tn'
        tn[method]['method'] = method
        tn[method]['extract_features'] = str(network_settings['extract_features'])
        fp[method]['metric'] = 'fp'
        fp[method]['method'] = method
        fp[method]['extract_features'] = str(network_settings['extract_features'])
        fn[method]['metric'] = 'fn'
        fn[method]['method'] = method
        fn[method]['extract_features'] = str(network_settings['extract_features'])
        thresholds[method]['metric'] = 'threshold'
        thresholds[method]['method'] = method
        thresholds[method]['extract_features'] = str(network_settings['extract_features'])
        
        with open(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier, 'results.csv'), 'a') as file:
                filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
                filewriter.writerow(tp[method])  
                filewriter.writerow(tn[method])  
                filewriter.writerow(fp[method])  
                filewriter.writerow(fn[method])  
                filewriter.writerow(thresholds[method])
    
        with open(os.path.join(directories['results_dir_cd'], save_pretaskname,dataset_settings['dataset_to_use'], save_layers_branches+'_'+save_layers_joint, classifier, 'results_avg.csv' ), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames2, delimiter = ",")
            filewriter.writerow({'method': method,
                                 'images':str(list(tp[method].keys())[:-2]), 
                                 'sensitivity': sensitivity, 
                                 'specificity':specificity, 
                                 'recall':recall, 
                                 'precision':precision, 
                                 'F1':F1, 
                                 'avg_acc':avg_acc, 
                                 'acc':acc, 
                                 'avg_threshold':avg_threshold,
                                 'extract_features': str(network_settings['extract_features'])})

