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
        'intermediate_dir_cd': '/media/cordolo/marrit/Intermediate/CD_OSCD',
        'intermediate_dir': '/media/cordolo/marrit/Intermediate',
        'source_dir_cd': '/media/cordolo/marrit/source_data/CD_OSCD',
        'source_dir_barrax': '/media/cordolo/marrit/source_data/CD_barrax',
        'csv_file_S21C': 'S21C_dataset.csv',
        'csv_file_S21C_cleaned': 'S21C_dataset_clean.csv',
        'csv_file_oscd': 'OSCD_dataset.csv',
        'csv_file_train_oscd' : 'OSCD_train.csv',
        'csv_file_test_oscd': 'OSCD_test.csv',
        'csv_file_patches_overlap': 'S21C_patches_overlap50-70.csv',
        'csv_models': 'trained_models_preliminary.csv',
        'csv_models_results': 'models_results.csv',
        'data_dir_S21C': 'data_S21C',
        'data_dir_barrax': 'data_barrax',
        'data_dir_oscd': 'data_OSCD',
        'data_dir_patches_overlap': 'patches_S21C_overlap50-70',
        'labels_dir_oscd' : 'labels_OSCD',
        'tb_dir': '/media/cordolo/marrit/Intermediate/tensorboard',
        'model_dir': '/media/cordolo/marrit/Intermediate/trained_models_preliminary'}
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
        'csv_models': 'trained_models_preliminary.csv',
        'csv_models_results': 'models_results.csv',
        'data_dir_S21C': 'data_S21C',
        'data_dir_oscd': 'data_OSCD',
        'labels_dir_oscd' : 'labels_OSCD',
        'data_dir_patches_overlap': 'patches_S21C_overlap90-100',
        'tb_dir': '/marrit1/Intermediate/tensorboard',
        'model_dir': '/marrit1/Intermediate/trained_models_preliminary'}

network_settings = {
    'network': 'siamese', # siamese" | "triplet" | "hypercolumn" | "triplet_apn" | siamese_unet | "siamese_dilated" | "triplet_unet" | "siamese_concat"
    'optimizer': 'adam',
    'lr':1e-3,
    'weight_decay':1e-4,
    'loss': 'bce_sigmoid', # "cross_entropy" | "bce_sigmoid" | "l1+triplet" | "nll" | "nll_finetune" | "triplet" | "l1+triplet" | "l1+triplet+bce" | "mse" 
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
cfg = {'branch': np.array([32,32,32,32,32], dtype='object'), 
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
    'num_epoch': 100,
    'batch_size': 25, # 1 =  25 in triplet_apn_dataset
    'disp_iter': 5,
    'gpu': None}

dataset_settings = {
    'dataset_type' : 'triplet', # "pair" | "triplet" | "triplet_saved" | "overlap" | "triplet_apn" | "triplet_oscd" | "overlap_regression" | "pair_hard_neg" | "triplet_apn_hard_neg"
    'perc_train': 0.85,
    'channels': np.arange(13),
    'min_overlap': 1, 
    'max_overlap': 1,
    'stride': int(network_settings['patch_size']),
    'patches_per_image': 25}

#%%
""" Train on S21C presaved dataset overlap """
from utils import cv_test_split
from train import train

directories['data_path'] = os.path.join(
    directories['intermediate_dir'], 
    'training_S2',
    directories['data_dir_patches_overlap'])

# use all images
dataset_settings['indices_train'],\
    dataset_settings['indices_val'],\
        dataset_settings['indices_test'] = cv_test_split(
            csv_file=os.path.join(
                directories['intermediate_dir'], 
                'training_S2',
                directories['csv_file_patches_overlap']), 
            n_test_ims=10,
            folds=5,
            k=0)

# subset of all images
#dataset_settings['indices_train'] = dataset_settings['indices_train'][:20000]
#dataset_settings['indices_val'] = dataset_settings['indices_val'][:2500]

t = [int(x.split('_')[0]) for x in dataset_settings['indices_train']]
unique, counts = np.unique(t, return_counts=True)
assert len(counts[counts < 2]) == 0, \
    "Training set: number of patches per images may not be smaller than 2"

v = [int(x.split('_')[0]) for x in dataset_settings['indices_val']]
unique, counts = np.unique(v, return_counts=True)
assert len(counts[counts < 2]) == 0, \
    "Validation set: number of patches per images may not be smaller than 2"

# train on only 1 image
# =============================================================================
# np.random.seed(234)
# random_image = np.random.choice(dataset['im_idx'], 1, replace=False)
# dataset = dataset[dataset['im_idx'] == random_image[0]]['im_patch_idx'].values
# 
# # train on image 0 only
# #random_pair = [0]
# 
# # for 1 image only
# dataset_settings['indices_train'] = np.repeat(dataset, 25)
# dataset_settings['indices_val'] = dataset
# dataset_settings['indices_test'] = dataset
# 
# =============================================================================
# train on n images (part kept apart for val and test)
# =============================================================================
# n = 25
# n_test = 4
# np.random.seed(234)
# random_im_idx = np.random.choice(dataset['im_idx'], n, replace=False)
# # defide over train val and test
# train_im_idx = random_im_idx[:int(dataset_settings['perc_train']*len(random_im_idx))]
# val_im_idx = random_im_idx[int(dataset_settings['perc_train']*len(random_im_idx)):-n_test]
# test_im_idx = random_im_idx[-n_test:]
# # get the patch_indices
# dataset_settings['indices_train'] = dataset[dataset['im_idx'].isin(train_im_idx)]['im_patch_idx'].values
# dataset_settings['indices_val'] = dataset[dataset['im_idx'].isin(val_im_idx)]['im_patch_idx'].values
# dataset_settings['indices_test'] = dataset[dataset['im_idx'].isin(test_im_idx)]['im_patch_idx'].values
# 
# =============================================================================
#dataset = dataset['im_idx'].values

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Train on presaved dataset, only patches from other image as pairs """
from train import train

directories['data_path'] = os.path.join(
    directories['intermediate_dir'], 
    'training_S2',
    directories['data_dir_patches_overlap'])

folds = 5
k = 0
n_ims_test = 100

# only patches from image a or image b
dataset = pd.read_csv(os.path.join(
                 directories['intermediate_dir'], 
                'training_S2',
                 directories['csv_file_patches_overlap']), sep=',')
patch0 = dataset[dataset['patchpair_idx'] == 0]
patch1 = dataset[dataset['patchpair_idx'] == 1]
patchpairs = pd.merge(patch0, patch1, on='im_patch_idx')
patchpairs_diff = patchpairs[patchpairs['impair_idx_x'] != patchpairs['impair_idx_y']]

# train / test split
dataset_ims = np.unique(patchpairs_diff['im_idx_x'])
np.random.seed(234)
test_ims = np.random.choice(dataset_ims, n_ims_test, replace=False)

# remove test images
dataset_ims = dataset_ims[np.isin(dataset_ims, test_ims) == False]

# shuffle and save train-part
import random
random.seed(890)
random.shuffle(dataset_ims)

ims_per_fold = int(len(dataset_ims)/folds)

# train / val split  
val_ims = dataset_ims[k*ims_per_fold:(k+1)*ims_per_fold]
train_ims = dataset_ims[np.isin(dataset_ims, val_ims) == False]

# find indices
dataset_val = patchpairs_diff[np.isin(patchpairs_diff['im_idx_x'], val_ims)]
dataset_val = dataset_val[dataset_val['impair_idx_x'] == 'a']
dataset_train = patchpairs_diff[np.isin(patchpairs_diff['im_idx_x'], train_ims)]
dataset_train = dataset_train[dataset_train['impair_idx_x'] == 'a']
dataset_test = patchpairs_diff[np.isin(patchpairs_diff['im_idx_x'], test_ims)]

indices_val = np.array(dataset_val['im_patch_idx'])
indices_train = np.array(dataset_train['im_patch_idx'])
indices_test = np.array(dataset_test['im_patch_idx'])

random.seed(123)
random.shuffle(indices_val)
random.shuffle(indices_train)
random.shuffle(indices_test)

dataset_settings['indices_train'] = indices_train
dataset_settings['indices_val'] = indices_val
dataset_settings['indices_test'] = indices_test

t = [int(x.split('_')[0]) for x in dataset_settings['indices_train']]
unique, counts = np.unique(t, return_counts=True)
assert len(counts[counts < 2]) == 0, \
     "Training set: number of patches per images may not be smaller than 2"
 
v = [int(x.split('_')[0]) for x in dataset_settings['indices_val']]
unique, counts = np.unique(v, return_counts=True)
assert len(counts[counts < 2]) == 0, \
    "Validation set: number of patches per images may not be smaller than 2"
    
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
#np.random.seed(432)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

dataset_settings['indices_train'] = np.repeat(dataset[:-250][:20],dataset_settings['patches_per_image'])
dataset_settings['indices_val'] = np.repeat(dataset[-250:-100][:20],dataset_settings['patches_per_image'])
dataset_settings['indices_test'] = np.repeat(dataset[-100:],dataset_settings['patches_per_image'])

# =============================================================================
# dataset = np.random.choice(dataset['im_idx'], 2, replace=False)
# dataset_settings['indices_train'] = np.repeat(dataset[0], 250)
# dataset_settings['indices_val'] = np.repeat(dataset[0], 50)
# dataset_settings['indices_test'] = np.repeat(dataset, 25)
# =============================================================================

# =============================================================================
# dataset = np.random.choice(dataset['im_idx'], 125, replace=False)
# dataset_settings['indices_train'] = dataset[:75]
# dataset_settings['indices_val'] = dataset[75:-25]
# dataset_settings['indices_test'] = dataset[-25:]
# =============================================================================
#dataset = dataset['im_idx'].values

# =============================================================================
# dataset_settings['indices_train'] = np.repeat(
#     dataset[:int(dataset_settings['perc_train']*len(dataset))],5)
# dataset_settings['indices_val'] = np.repeat(
#     dataset[int(dataset_settings['perc_train']*len(dataset)):-25],5)
# dataset_settings['indices_test'] = np.repeat(
#     dataset[-25:],5)
# =============================================================================

# =============================================================================
# dataset_settings['indices_train'] = np.repeat(dataset[:int(dataset_settings['perc_train']*len(dataset))],5)
# dataset_settings['indices_val'] = np.repeat(dataset[int(dataset_settings['perc_train']*len(dataset)):-25],5)
# dataset_settings['indices_test'] = dataset[-25:]
# 
# =============================================================================
# continue training
# =============================================================================
# trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], directories['csv_models']))
# model_settings = trained_models.iloc[-1]
# print(model_settings)
# network_settings['weights_file'] = model_settings['filename']
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

np.random.seed(234)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
# =============================================================================
# np.random.seed(234)
# dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
# =============================================================================
#dataset = np.array([8]) # select one specific image
#dataset = dataset[1:]

### USE THIS ###
dataset_settings['indices_train'] = np.repeat(dataset[:-2], dataset_settings['patches_per_image'])
dataset_settings['indices_val'] = np.repeat(dataset[-2:], dataset_settings['patches_per_image'])
# =============================================================================
# dataset_settings['indices_train'] = np.repeat(dataset[1], dataset_settings['patches_per_image'])
# dataset_settings['indices_val'] = np.repeat(dataset[2], dataset_settings['patches_per_image'])
# =============================================================================


#dataset_settings['indices_test'] = np.repeat(dataset[0], 25)

# continue training
# =============================================================================
# trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], directories['csv_models']))
# model_settings = trained_models.iloc[-1]
# network_settings['weights_file'] = model_settings['filename']
# =============================================================================

# train
train(directories, dataset_settings, network_settings, train_settings)

# %%
""" Train on images of OSCD dataset saving in memory"""
from train import train

directories['data_path'] = os.path.join(
    directories['source_dir_cd'],
    directories['data_dir_oscd'])

train_data = pd.read_csv(os.path.join(
    directories['source_dir_cd'],
    directories['data_dir_oscd'],'train.txt')).columns

np.random.seed(234)
val_data = np.random.choice(train_data.values, 2, replace=False)
train_data = train_data[np.isin(train_data, val_data) == False]

print('validation cities: ', val_data)
print('train cities: ', train_data.values)

dataset_settings['indices_train'] = train_data
dataset_settings['indices_val'] = val_data

# train
train(directories, dataset_settings, network_settings, train_settings)

#%%
""" Finetune APN network on OSCD-dataset """
from extract_features import get_network    
from train import finetune
# network
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

# get model
#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get dataset
dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
indices = np.random.choice(dataset['im_idx'], size = int(np.ceil(len(dataset)/2)), replace=False)
indices_val= dataset['im_idx'][np.isin(dataset['im_idx'].values,indices) == False].values 
dataset_settings['indices_train'] = indices
dataset_settings['indices_val'] = indices_val

finetune(model_settings, directories, dataset_settings, network_settings, train_settings)
#%%
""" Plot training history """
import torch
import matplotlib.pyplot as plt

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
# get model info
model_settings = trained_models.iloc[-1] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get history
history_file = os.path.join(directories['model_dir'], 'history_' + model_settings['filename'].split('/')[-1])
history = torch.load(history_file)

history_loss = pd.DataFrame(history['train']['loss'][:-1], columns= ['train'], index=history['train']['epoch'][:-1])
history_loss['val'] = history['val']['loss']
history_acc = pd.DataFrame(history['train']['acc'][:-1], columns= ['train'], index=history['train']['epoch'][:-1])
history_acc['val'] = history['val']['acc']

fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
history_loss.plot(ax=ax[1])
history_acc.plot(ax=ax[0])
ax[1].set_title('loss')
ax[1].set_xlabel('epoch')
ax[0].set_title('accuracy')
ax[0].set_xlabel('epoch')
#ax[0].set_ylim([0.5,1])
#ax[1].set_ylim([0,1.5])
fig.suptitle(model_settings['filename'].split('/')[-1], fontsize=18)
plt.show()

#%%
""" Evaluate (preliminary) network on training task"""
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

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
dataset_settings['indices_eval'] = np.repeat(dataset[-250:-100][-20:],dataset_settings['patches_per_image'])

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
""" Evaluate (preliminary) network on change detection task"""

from change_detection import detect_changes

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
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

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(571)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
dataset_settings['indices_eval'] = dataset[:2]

# set extraction layer
network_settings['extract_features'] = 'joint'

methods = ['triangle']
tp, tn, fp, fn = detect_changes(
    model_settings, 
    directories, 
    dataset_settings, 
    network_settings, 
    train_settings, 
    threshold_methods=methods)

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
    print('------------------------------')
    print("RESULTS METHOD: {}:".format(method))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)   
    print('------------------------------')

trained_models.loc[trained_models.index[model_idx],'tp_oscd_eval_triangle'] = sum(tp[method].values())
trained_models.loc[trained_models.index[model_idx],'tn_oscd_eval_triangle'] = sum(tn[method].values())
trained_models.loc[trained_models.index[model_idx],'fp_oscd_eval_triangle'] = sum(fp[method].values())
trained_models.loc[trained_models.index[model_idx],'fn_oscd_eval_triangle'] = fn_temp = sum(fn[method].values())
trained_models.loc[trained_models.index[model_idx],'f1_oscd_eval_triangle'] = F1
trained_models.to_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']),index=False)  













#%%
""" Check predictions of network """
from extract_features import get_network
from train import get_dataset
from torch.utils.data import DataLoader

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

train_settings['gpu'] = None
# get model info
model_settings = trained_models.iloc[-2] # TODO: change here the model
n_branches = 2
print("MODEL SETTINGS: \n", model_settings)
network = get_network(model_settings)


# Datasets
directories['data_path'] = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
#np.random.seed(432)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

dataset_settings['indices_train'] = np.repeat(dataset[:-150][:5],dataset_settings['patches_per_image'])
dataset_settings['indices_val'] = np.repeat(dataset[-150:-10][:5],dataset_settings['patches_per_image'])
dataset_settings['indices_test'] = np.repeat(dataset[-10:][:5],dataset_settings['patches_per_image'])
dataset_test = get_dataset(
    data_path=directories['data_path'], 
    indices=dataset_settings['indices_train'], 
    channels=dataset_settings['channels'], 
    one_hot=False, 
    dataset_settings=dataset_settings, 
    network_settings=network_settings)

# Data loaders
dataloader = DataLoader(
    dataset_test, 
    batch_size=min(train_settings['batch_size'],len(dataset_test)), 
    shuffle=True,
    num_workers = 1)

     
network.eval() 
  
iterator = iter(dataloader)

batch_data = next(iterator)

for item in batch_data: 
    batch_data[item] = torch.squeeze(batch_data[item])

# get the inputs
if n_branches == 2:
    if train_settings['gpu'] != None:
        inputs = [batch_data['patch0'].float().cuda(), 
                  batch_data['patch1'].float().cuda()] 
    else:
        inputs = [batch_data['patch0'].float(), 
                  batch_data['patch1'].float()] 

elif n_branches == 3:
    if train_settings['gpu'] != None:
        inputs = [batch_data['patch0'].float().cuda(), 
                  batch_data['patch1'].float().cuda(),
                  batch_data['patch2'].float().cuda()] 
    else:
        inputs = [batch_data['patch0'].float(), 
                  batch_data['patch1'].float(),
                  batch_data['patch2'].float()] 
labels = batch_data['label']

# to gpu
if train_settings['gpu'] != None:
    labels = labels.cuda()
    
# forward pass
with torch.no_grad():
    outputs_full = network(inputs, n_branches)
    
loss_func = torch.nn.MSELoss()
loss = loss_func(outputs_full, labels)
print('loss: ', loss)

#%% 
""" Check accuracy model on OSCD dataset """
from train import validate

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

# get model
#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1] # TODO: change here the model
print("MODEL SETTINGS: \n", model_settings)

# get oscd dataset
eval_settings = dict()
eval_settings['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])
dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
#dataset = np.array([21]) # select one specific image
eval_settings['indices_eval'] = np.repeat(dataset, 75)

eval_settings['gpu'] = train_settings['gpu']
eval_settings['batch_size'] = 25

# evaluate network
acc, loss = validate(model_settings, eval_settings)
print("accuracy: ", acc)
print("loss: ", loss)

#%%


from change_detection import detect_changes

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))
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

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(571)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
dataset_settings['indices_eval'] = dataset[:2]

# set extraction layer
network_settings['extract_features'] = 'joint'

methods = ['triangle']
tp, tn, fp, fn = detect_changes(
    model_settings, 
    directories, 
    dataset_settings, 
    network_settings, 
    train_settings, 
    threshold_methods=methods)

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
    print('------------------------------')
    print("RESULTS METHOD: {}:".format(method))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)   
    print('------------------------------')

trained_models.loc[trained_models.index[model_idx],'tp_oscd_eval_triangle'] = sum(tp[method].values())
trained_models.loc[trained_models.index[model_idx],'tn_oscd_eval_triangle'] = sum(tn[method].values())
trained_models.loc[trained_models.index[model_idx],'fp_oscd_eval_triangle'] = sum(fp[method].values())
trained_models.loc[trained_models.index[model_idx],'fn_oscd_eval_triangle'] = fn_temp = sum(fn[method].values())
trained_models.loc[trained_models.index[model_idx],'f1_oscd_eval_triangle'] = F1
trained_models.to_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']),index=False)  

#%%
""" Extract Features """

from extract_features import get_network, extract_features, get_dcv
from extract_features import calculate_magnitudemap, calculate_changemap
from metrics import compute_confusion_matrix, compute_mcc
from plots import plot_changemap_colors, plot_image, plot_distance_decision
from matplotlib import pyplot as plt
import csv

# init save-file
fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
              'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
              'n_channels','patch_size','batch_norm','dataset','min_overlap',\
              'max_overlap', 'best_acc','best_epoch', 'best_loss', 'cd_dataset', \
              'extract_layers', 'prop_layers', 'tp', 'tn', 'fp', 'fn', \
              'sensitivity', 'specificity', 'recall', 'precision', 'F1', 'accuracy',\
              'threshold']   
if not os.path.isdir(directories['results_dir_cd']):
    os.makedirs(directories['results_dir_cd'])     
if not os.path.exists(os.path.join(directories['results_dir_cd'], 
                                   directories['csv_models_results'])):
    with open(os.path.join(directories['results_dir_cd'], 
                           directories['csv_models_results']), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

# get model
#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1] # TODO: change here the model
extract_layers = None # for manual setting which layers to use
which_layers = 'activations' # TODO: change here which layers (conv / activations)

if extract_layers == None:
    # cfgs are saved as strings, cast back to list
    branch = model_settings['cfg_branch'].split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    branch = branch.replace(" ", "")
    branch = branch.split(",")
    
    if which_layers == 'conv':
        # extract the features from all conv layers
        extract_layers = list()
        layer_i = 0
        if model_settings['batch_norm']:
            for layer in branch:
                if layer == 'M':
                    layer_i += 1
                else:
                    extract_layers.append(layer_i)
                    layer_i += 3
                    
        else:
            for layer in branch:
                if layer == 'M':
                    layer_i += 1
                else:
                    extract_layers.append(layer_i)
                    layer_i += 2
    elif which_layers == 'activations':
        # extract the features from all conv layers
        extract_layers = list()
        layer_i = 0
        if model_settings['batch_norm']:
            for layer in branch:
                if layer == 'M':
                    layer_i += 1
                else:
                    layer_i += 2
                    extract_layers.append(layer_i)
                    layer_i += 1
                    
        else:
            for layer in branch:
                if layer == 'M':
                    layer_i += 1
                else:
                    layer_i += 1
                    extract_layers.append(layer_i)
                    layer_i += 1

model_settings['extract_features'] = extract_layers
model_settings['threshold'] = 'triangle' # TODO: change here the threshold method
print("MODEL SETTINGS: \n", model_settings)
net = get_network(model_settings)

# get cd dataset
model_settings['cd_dataset'] = 'train_oscd'
oscd_dataset = pd.read_csv(os.path.join(
    directories['intermediate_dir_cd'], 
    directories['csv_file_' + model_settings['cd_dataset']]))

# get indices
oscd_indices = np.unique(oscd_dataset.im_idx.values)
#oscd_indices = np.array([3])

# initiate confusion matrix
cm = np.zeros((2,2), dtype=np.int)
if not os.path.isdir(os.path.join(directories['results_dir_cd'],
                     model_settings['filename'].split('/')[-1])+'_'+model_settings['threshold']):
     os.makedirs(os.path.join(directories['results_dir_cd'],
                              model_settings['filename'].split('/')[-1])+'_'+model_settings['threshold'])

for idx in oscd_indices:
    
    cityname = oscd_dataset[oscd_dataset['im_idx'] == idx].iloc[0]['city']
    # load images
    im_a = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'], 
        str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'],
        str(idx)+'_b.npy'))
    gt = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['labels_dir_oscd'],
        str(idx)+'.npy'))
  
    # plot images
    plot_image(
        os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd']),
        [str(idx)+'_a.npy', str(idx)+'_b.npy'], [3,2,1], titles=[str(idx)+'_a.npy', str(idx)+'_b.npy'])
    
    # get features
    features = extract_features(net, model_settings['extract_features'], [im_a, im_b])
    
    # calculate difference maps per layer
    start = 1
    layers_diffmap = range(start,len(features[0])) # TODO: change here which layers to extract from
    # find height and width of S splits # for now in 4 splits, may be suoptimal
    height = im_a.shape[0] // 2
    width = im_a.shape[1] // 2
    prop=0.5 # TODO: change here proportion of features to keep
    dcv = get_dcv(features, layers_diffmap, height, width, prop)
    
    # calculate distance map
    distmap = calculate_magnitudemap(dcv)
    
    # calculate change map
    changemap, threshold = calculate_changemap(distmap, method=model_settings['threshold'])
    plot_distance_decision(distmap, changemap, gt, threshold)
    np.save(os.path.join(
        directories['results_dir_cd'],
        model_settings['filename'].split('/')[-1]+'_'+model_settings['threshold'],
        str(idx)+'.npy'), changemap)
  
    #plot_changemap_colors(changemap, gt, title='Change map '+cityname, axis=False)

    # confusion matrix
    conf_matrix, fig3, axes3 = compute_confusion_matrix(gt, changemap, normalize=False)
    cm += conf_matrix
       
    # compute metrix
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
    recall = sensitivity
    specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
    precision = tp/(tp+fp) # prop of changed pixels that are truly changed
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp+tn)/sum(sum(conf_matrix))
    # print metrics
    print('------------------------------')
    print("RESULTS OSCD "+str(idx))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)    
    

    
# compute metrics
#mcc = compute_mcc(cm)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
recall = sensitivity
specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
precision = tp/(tp+fp) # prop of changed pixels that are truly changed
F1 = 2 * (precision * recall) / (precision + recall)
acc = (tp+tn)/sum(sum(cm))

# print metrics
#print("mcc: ", mcc)
print('------------------------------')
print("RESULTS DATASET:")
print("sensitivity: ", sensitivity)
print("specificity: ", specificity)
print("recall: ", sensitivity)
print("precision: ", precision)
print("F1: ", F1)
print("acc: ", acc)    

model_settings['extract_layers'] = model_settings['extract_features'][start:]
model_settings['prop_layers'] = prop 
model_settings['tp'] = tp
model_settings['tn'] = tn
model_settings['fp'] = fp
model_settings['fn'] = fn
model_settings['sensitivity'] = sensitivity
model_settings['specificity'] = specificity
model_settings[ 'recall'] = recall
model_settings['precision'] = precision
model_settings['F1'] = F1
model_settings['accuracy'] = acc

with open(os.path.join(directories['results_dir_cd'], 
                       directories['csv_models_results']), 'a') as file:
    filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
    filewriter.writerow(model_settings)  
    
#%%
""" Test on OSCD dataset """
from extract_features import get_network, extract_features, get_dcv
from extract_features import calculate_magnitudemap, calculate_changemap
from metrics import compute_confusion_matrix, compute_mcc
from plots import plot_changemap_colors, plot_image, plot_distance_decision
from matplotlib import pyplot as plt
import csv
from models.netbuilder import NetBuilder

# init save-file
# =============================================================================
# fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
#               'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
#               'n_channels','patch_size','batch_norm','dataset','min_overlap',\
#               'max_overlap', 'best_acc','best_epoch', 'best_loss', 'cd_dataset', \
#               'extract_layers', 'prop_layers', 'tp', 'tn', 'fp', 'fn', \
#               'sensitivity', 'specificity', 'recall', 'precision', 'F1', 'accuracy',\
#               'threshold']   
# if not os.path.isdir(directories['results_dir_cd']):
#     os.makedirs(directories['results_dir_cd'])     
# if not os.path.exists(os.path.join(directories['results_dir_cd'], 
#                                    directories['csv_models_results'])):
#     with open(os.path.join(directories['results_dir_cd'], 
#                            directories['csv_models_results']), 'a') as file:
#         filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
#         filewriter.writeheader()
# =============================================================================

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

# get model
#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1] # TODO: change here the model
model_settings['threshold'] = 'triangle' # TODO: change here the threshold method
print("MODEL SETTINGS: \n", model_settings)
net = get_network(model_settings)
       
# get cd dataset
model_settings['cd_dataset'] = 'train_oscd'
oscd_dataset = pd.read_csv(os.path.join(
    directories['intermediate_dir_cd'], 
    directories['csv_file_' + model_settings['cd_dataset']]))

# get indices
oscd_indices = np.unique(oscd_dataset.im_idx.values)
oscd_indices = np.array([8])

# initiate confusion matrix
cm = np.zeros((2,2), dtype=np.int)
if not os.path.isdir(os.path.join(directories['results_dir_cd'],
                     model_settings['filename'].split('/')[-1])+'_'+model_settings['threshold']):
     os.makedirs(os.path.join(directories['results_dir_cd'],
                              model_settings['filename'].split('/')[-1])+'_'+model_settings['threshold'])

for idx in oscd_indices:
    
    cityname = oscd_dataset[oscd_dataset['im_idx'] == idx].iloc[0]['city']
    # load images
    im_a = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'], 
        str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'],
        str(idx)+'_b.npy'))
    gt = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['labels_dir_oscd'],
        str(idx)+'.npy'))
  
    # plot images
    plot_image(
        os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd']),
        [str(idx)+'_a.npy', str(idx)+'_b.npy'], [3,2,1], titles=[str(idx)+'_a.npy', str(idx)+'_b.npy'])
    
    # prepare images
    from plots import normalize2plot
    import torch
    im_n_a = normalize2plot(im_a)
    im_n_a = np.moveaxis(im_n_a, -1, 0)
    im_n_a = np.expand_dims(im_n_a, axis=0)
    im_t_a = torch.as_tensor(im_n_a)
    im_n_b = normalize2plot(im_b)
    im_n_b = np.moveaxis(im_n_b, -1, 0)
    im_n_b = np.expand_dims(im_n_b, axis=0)
    im_t_b = torch.as_tensor(im_n_b)
            
    # get distance map
    distmap = net([im_t_a.float(),im_t_b.float()], 2, extract_features='last')
    distmap = distmap[0][0].detach().numpy()  

    # calculate change map
    changemap, threshold = calculate_changemap(distmap, method=model_settings['threshold'])
    plot_distance_decision(distmap, changemap, gt, threshold)
    np.save(os.path.join(
        directories['results_dir_cd'],
        model_settings['filename'].split('/')[-1]+'_'+model_settings['threshold'],
        str(idx)+'.npy'), changemap)
  
    #plot_changemap_colors(changemap, gt, title='Change map '+cityname, axis=False)

    # confusion matrix
    conf_matrix, fig3, axes3 = compute_confusion_matrix(gt, changemap, normalize=False)
    cm += conf_matrix
       
    # compute metrix
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
    recall = sensitivity
    specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
    precision = tp/(tp+fp) # prop of changed pixels that are truly changed
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp+tn)/sum(sum(conf_matrix))
    # print metrics
    print('------------------------------')
    print("RESULTS OSCD "+str(idx))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)    
    

    
# compute metrics
#mcc = compute_mcc(cm)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
recall = sensitivity
specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
precision = tp/(tp+fp) # prop of changed pixels that are truly changed
F1 = 2 * (precision * recall) / (precision + recall)
acc = (tp+tn)/sum(sum(cm))

# print metrics
#print("mcc: ", mcc)
print('------------------------------')
print("RESULTS DATASET:")
print("sensitivity: ", sensitivity)
print("specificity: ", specificity)
print("recall: ", sensitivity)
print("precision: ", precision)
print("F1: ", F1)
print("acc: ", acc)    

#model_settings['extract_layers'] = model_settings['extract_features'][start:]
#model_settings['prop_layers'] = prop 
model_settings['tp'] = tp
model_settings['tn'] = tn
model_settings['fp'] = fp
model_settings['fn'] = fn
model_settings['sensitivity'] = sensitivity
model_settings['specificity'] = specificity
model_settings[ 'recall'] = recall
model_settings['precision'] = precision
model_settings['F1'] = F1
model_settings['accuracy'] = acc

with open(os.path.join(directories['results_dir_cd'], 
                       directories['csv_models_results']), 'a') as file:
    filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
    filewriter.writerow(model_settings)  
    
#%%
""" Test on OSCD dataset APN model with output: anchor, positive, negative"""
from extract_features import get_network, extract_features, get_dcv
from extract_features import calculate_magnitudemap, calculate_changemap
from metrics import compute_confusion_matrix, compute_mcc
from plots import plot_changemap_colors, plot_image, plot_distance_decision
from matplotlib import pyplot as plt
import csv
from models.netbuilder import NetBuilder

# init save-file
fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
              'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
              'n_channels','patch_size','batch_norm','dataset','min_overlap',\
              'max_overlap', 'best_acc','best_epoch', 'best_loss', 'cd_dataset', \
              'extract_layers', 'prop_layers', 'tp', 'tn', 'fp', 'fn', \
              'sensitivity', 'specificity', 'recall', 'precision', 'F1', 'accuracy',\
              'threshold']   
if not os.path.isdir(directories['results_dir_cd']):
    os.makedirs(directories['results_dir_cd'])     
if not os.path.exists(os.path.join(directories['results_dir_cd'], 
                                   directories['csv_models_results'])):
    with open(os.path.join(directories['results_dir_cd'], 
                           directories['csv_models_results']), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

# get model
#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1] # TODO: change here the model
model_settings['threshold'] = 'triangle' # TODO: change here the threshold method
print("MODEL SETTINGS: \n", model_settings)
net = get_network(model_settings)
       
# get cd dataset
model_settings['cd_dataset'] = 'train_oscd'
oscd_dataset = pd.read_csv(os.path.join(
    directories['intermediate_dir_cd'], 
    directories['csv_file_' + model_settings['cd_dataset']]))

# get indices
oscd_indices = np.unique(oscd_dataset.im_idx.values)
#oscd_indices = np.array([17])

# initiate confusion matrix
cm = np.zeros((2,2), dtype=np.int)
if not os.path.isdir(os.path.join(directories['results_dir_cd'],
                     model_settings['filename'].split('/')[-1])+'_'+model_settings['threshold']):
     os.makedirs(os.path.join(directories['results_dir_cd'],
                              model_settings['filename'].split('/')[-1])+'_'+model_settings['threshold'])

for idx in oscd_indices:
    
    cityname = oscd_dataset[oscd_dataset['im_idx'] == idx].iloc[0]['city']
    # load images
    im_a = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'], 
        str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'],
        str(idx)+'_b.npy'))
    gt = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['labels_dir_oscd'],
        str(idx)+'.npy'))
  
    # plot images
    plot_image(
        os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd']),
        [str(idx)+'_a.npy', str(idx)+'_b.npy'], [3,2,1], titles=[str(idx)+'_a.npy', str(idx)+'_b.npy'])
    
    # prepare images
    from plots import normalize2plot
    import torch
    im_n_a = normalize2plot(im_a)
    im_n_a = np.moveaxis(im_n_a, -1, 0)
    im_n_a = np.expand_dims(im_n_a, axis=0)
    im_t_a = torch.as_tensor(im_n_a)
    im_n_b = normalize2plot(im_b)
    im_n_b = np.moveaxis(im_n_b, -1, 0)
    im_n_b = np.expand_dims(im_n_b, axis=0)
    im_t_b = torch.as_tensor(im_n_b)
            
    # get distance map
    features = net([im_t_a.float(),im_t_b.float()], 2, extract_features=None)
    features = [np.moveaxis(f.detach().numpy(), 1,-1)for f in features]
    # calculate difference maps per layer
    start = 0
    layers_diffmap = range(start,len(features[0])) # TODO: change here which layers to extract from
    # find height and width of S splits # for now in 4 splits, may be suoptimal
    height = im_a.shape[0] // 2
    width = im_a.shape[1] // 2
    prop=1 # TODO: change here proportion of features to keep
    dcv = get_dcv(features, layers_diffmap, height, width, prop)
    
    # calculate distance map
    distmap = calculate_magnitudemap(dcv)

    # calculate change map
    changemap, threshold = calculate_changemap(distmap, method=model_settings['threshold'])
    plot_distance_decision(distmap, changemap, gt, threshold)
    np.save(os.path.join(
        directories['results_dir_cd'],
        model_settings['filename'].split('/')[-1]+'_'+model_settings['threshold'],
        str(idx)+'.npy'), changemap)
  
    #plot_changemap_colors(changemap, gt, title='Change map '+cityname, axis=False)

    # confusion matrix
    conf_matrix, fig3, axes3 = compute_confusion_matrix(gt, changemap, normalize=False)
    cm += conf_matrix
       
    # compute metrix
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
    recall = sensitivity
    specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
    precision = tp/(tp+fp) # prop of changed pixels that are truly changed
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp+tn)/sum(sum(conf_matrix))
    # print metrics
    print('------------------------------')
    print("RESULTS OSCD "+str(idx))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)    
    

    
# compute metrics
#mcc = compute_mcc(cm)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
recall = sensitivity
specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
precision = tp/(tp+fp) # prop of changed pixels that are truly changed
F1 = 2 * (precision * recall) / (precision + recall)
acc = (tp+tn)/sum(sum(cm))

# print metrics
#print("mcc: ", mcc)
print('------------------------------')
print("RESULTS DATASET:")
print("sensitivity: ", sensitivity)
print("specificity: ", specificity)
print("recall: ", sensitivity)
print("precision: ", precision)
print("F1: ", F1)
print("acc: ", acc)    

#model_settings['extract_layers'] = model_settings['extract_features'][start:]
#model_settings['prop_layers'] = prop 
model_settings['tp'] = tp
model_settings['tn'] = tn
model_settings['fp'] = fp
model_settings['fn'] = fn
model_settings['sensitivity'] = sensitivity
model_settings['specificity'] = specificity
model_settings[ 'recall'] = recall
model_settings['precision'] = precision
model_settings['F1'] = F1
model_settings['accuracy'] = acc

with open(os.path.join(directories['results_dir_cd'], 
                       directories['csv_models_results']), 'a') as file:
    filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
    filewriter.writerow(model_settings)  
    

#%%
""" Test finetuned APN model on OSCD dataset """
from extract_features import get_network, extract_features, get_dcv
from extract_features import calculate_magnitudemap, calculate_changemap
from metrics import compute_confusion_matrix, compute_mcc
from plots import plot_image, plot_changemap_colors_gt, normalize2plot
from matplotlib import pyplot as plt
import csv
from models.netbuilder import NetBuilder
from models.siameseNetAPNFinetune import siamese_net_apn_finetune
import torch

# init save-file
fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
              'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
              'n_channels','patch_size','batch_norm','dataset','min_overlap',\
              'max_overlap', 'best_acc','best_epoch', 'best_loss', 'cd_dataset', \
              'extract_layers', 'prop_layers', 'tp', 'tn', 'fp', 'fn', \
              'sensitivity', 'specificity', 'recall', 'precision', 'F1', 'accuracy',\
              'threshold']   
if not os.path.isdir(directories['results_dir_cd']):
    os.makedirs(directories['results_dir_cd'])     
if not os.path.exists(os.path.join(directories['results_dir_cd'], 
                                   directories['csv_models_results'])):
    with open(os.path.join(directories['results_dir_cd'], 
                           directories['csv_models_results']), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()

# get model files
trained_models = pd.read_csv(os.path.join(directories['intermediate_dir'], 
                                          directories['csv_models']))

# get model
#model_settings = trained_models.sort_values('best_acc',ascending=False).iloc[0]
model_settings = trained_models.iloc[-1] # TODO: change here the model
#model_settings['threshold'] = 'triangle' # TODO: change here the threshold method
print("MODEL SETTINGS: \n", model_settings)
#model_settings['networkname'] = 'triplet_apn'
#net = get_network(model_settings)  

# =============================================================================
# model_settings = trained_models.iloc[-1]
# # find size last layer branch
# branch = model_settings['cfg_branch'].split("[" )[1]
# branch = branch.split("]" )[0]
# branch = branch.replace("'", "")
# branch = branch.replace(" ", "")
# branch = branch.split(",")
# branch = np.array(branch)
# last_in = int(branch[-1])
# =============================================================================
net = siamese_net_apn_finetune(
    model_settings['cfg_branch'], 
    model_settings['batch_norm'], 
    model_settings['n_channels'], 
    model_settings['n_classes'])  
net.load_state_dict(torch.load(model_settings['filename']), strict=False)
       
# get cd dataset
model_settings['cd_dataset'] = 'train_oscd'
oscd_dataset = pd.read_csv(os.path.join(
    directories['intermediate_dir_cd'], 
    directories['csv_file_' + model_settings['cd_dataset']]))

# get indices
oscd_indices = np.unique(oscd_dataset.im_idx.values)
#oscd_indices = np.array([17])

# initiate confusion matrix
cm = np.zeros((2,2), dtype=np.int)
if not os.path.isdir(os.path.join(directories['results_dir_cd'],
                     model_settings['filename'].split('/')[-1])+'_finetunedAPN'):
     os.makedirs(os.path.join(directories['results_dir_cd'],
                              model_settings['filename'].split('/')[-1])+'_finetunedAPN')

for idx in oscd_indices:
    
    cityname = oscd_dataset[oscd_dataset['im_idx'] == idx].iloc[0]['city']
    # load images
    im_a = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'], 
        str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['data_dir_oscd'],
        str(idx)+'_b.npy'))
    gt = np.load(os.path.join(
        directories['intermediate_dir_cd'], 
        directories['labels_dir_oscd'],
        str(idx)+'.npy'))
  
    # plot images
    plot_image(
        os.path.join(directories['intermediate_dir_cd'], directories['data_dir_oscd']),
        [str(idx)+'_a.npy', str(idx)+'_b.npy'], [3,2,1], titles=[str(idx)+'_a.npy', str(idx)+'_b.npy'])
    
    # prepare images
    im_n_a = normalize2plot(im_a)
    im_n_a = np.moveaxis(im_n_a, -1, 0)
    im_n_a = np.expand_dims(im_n_a, axis=0)
    im_t_a = torch.as_tensor(im_n_a)
    im_n_b = normalize2plot(im_b)
    im_n_b = np.moveaxis(im_n_b, -1, 0)
    im_n_b = np.expand_dims(im_n_b, axis=0)
    im_t_b = torch.as_tensor(im_n_b)
            
    # get distance map
    output = net([im_t_a.float(),im_t_b.float()], 2, extract_features='joint')
    changemap = np.argmax(np.squeeze(output.detach().numpy()),axis = 0)
    #plt.imshow(changemap, cmap=cm.gray)
    plot_changemap_colors_gt(changemap, gt, title='Change map', axis=False)
    np.save(os.path.join(    
        directories['results_dir_cd'],
        model_settings['filename'].split('/')[-1]+'_finetunedAPN',
        str(idx)+'.npy'), changemap)
  
    #plot_changemap_colors(changemap, gt, title='Change map '+cityname, axis=False)

    # confusion matrix
    conf_matrix, fig3, axes3 = compute_confusion_matrix(gt, changemap.astype(bool), normalize=False)
    cm += conf_matrix
       
    # compute metrix
    tp = conf_matrix[1,1]
    tn = conf_matrix[0,0]
    fp = conf_matrix[0,1]
    fn = conf_matrix[1,0]
    sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
    recall = sensitivity
    specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
    precision = tp/(tp+fp) # prop of changed pixels that are truly changed
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp+tn)/sum(sum(conf_matrix))
    # print metrics
    print('------------------------------')
    print("RESULTS OSCD "+str(idx))
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("recall: ", sensitivity)
    print("precision: ", precision)
    print("F1: ", F1)
    print("acc: ", acc)    
    

    
# compute metrics
#mcc = compute_mcc(cm)
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]
sensitivity = tp/(tp+fn) # prop of correctly identified changed pixels = recall
recall = sensitivity
specificity = tn/(tn+fp) # prop of correctly identified unchanged pixels
precision = tp/(tp+fp) # prop of changed pixels that are truly changed
F1 = 2 * (precision * recall) / (precision + recall)
acc = (tp+tn)/sum(sum(cm))

# print metrics
#print("mcc: ", mcc)
print('------------------------------')
print("RESULTS DATASET:")
print("sensitivity: ", sensitivity)
print("specificity: ", specificity)
print("recall: ", sensitivity)
print("precision: ", precision)
print("F1: ", F1)
print("acc: ", acc)    

#model_settings['extract_layers'] = model_settings['extract_features'][start:]
#model_settings['prop_layers'] = prop 
model_settings['tp'] = tp
model_settings['tn'] = tn
model_settings['fp'] = fp
model_settings['fn'] = fn
model_settings['sensitivity'] = sensitivity
model_settings['specificity'] = specificity
model_settings[ 'recall'] = recall
model_settings['precision'] = precision
model_settings['F1'] = F1
model_settings['accuracy'] = acc

with open(os.path.join(directories['results_dir_cd'], 
                       directories['csv_models_results']), 'a') as file:
    filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
    filewriter.writerow(model_settings)  



#%%
lbl_bottle = np.array([0,1,0,1,1])
inputs = [np.zeros((5,3,3,3)), np.ones((5,3,3,3)), np.ones((5,3,3,3))*2]

positives = np.zeros((5,3,3,3))
negatives = np.zeros((5,3,3,3))
for i, label in enumerate(lbl_bottle):
    if label == 0:
        positives[i] = inputs[1][i]
        negatives[i] = inputs[2][i]
    elif label == 1:
        positives[i] = inputs[2][i]
        negatives[i] = inputs[1][i]

#%%
""" Evaluate network on feature level """
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

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']
np.random.seed(234)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)
dataset_settings['indices_eval'] = np.repeat(dataset[-250:-100][20:40],dataset_settings['patches_per_image'])

# set extraction layer
network_settings['extract_features'] = 'diff'