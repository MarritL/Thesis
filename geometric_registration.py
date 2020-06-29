#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:19:44 2020

@author: M. Leenstra
"""
import os
import csv
import pandas as pd
import numpy as np
from skimage.feature import register_translation
import cv2 

from setup import setup
from plots import normalize2plot

# init
directories, network_settings, train_settings, dataset_settings = setup()

# get dataset
data_path = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))
dataset = dataset.loc[dataset['pair_idx'] == 'a']

# init save-files for radiometric difference
fieldnames = ['idx','offset_x', 'offset_y', 'NRMSE', 'diffphase']
if not os.path.exists(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference',
                    'phase_cross_correlation.csv')):
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference',
                    'phase_cross_correlation.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()

# loop over dataset
for i, idx in enumerate(dataset['im_idx']):
    # load images
    im_a = np.load(os.path.join(directories['results_dir_training'],directories['data_dir_S21C'], str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(directories['results_dir_training'],directories['data_dir_S21C'], str(idx)+'_b.npy'))
   
    # normalize 
    im_a_rgb = (normalize2plot(im_a[:,:,[3,2,1]])*255).astype(np.uint8)
    im_b_rgb = (normalize2plot(im_b[:,:,[3,2,1]])*255).astype(np.uint8)
    # conver to gray scale
    im_a_gray= cv2.cvtColor(im_a_rgb,cv2.COLOR_RGB2GRAY)
    im_b_gray= cv2.cvtColor(im_b_rgb,cv2.COLOR_RGB2GRAY)
    
    # calculate shift needed for registration (upsample factor 100 for subpixel registration)
    shift, error, diffphase = register_translation(im_a_gray, im_b_gray, upsample_factor=100)  
    offset_x = shift[1]
    offset_y = shift[0]
    
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference',
                    'phase_cross_correlation.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
        filewriter.writerow({'idx':idx,
                             'offset_x': offset_x,
                             'offset_y': offset_y,
                             'NRMSE': error,
                             'diffphase': diffphase})  
        
    if (i+1) % 5 == 0:
        print('\r Image pair: [{}/{}]'.format(i+1, len(dataset)), end='')

#%%
from matplotlib import pyplot as plt   
from plots import plot_image     
cc_file = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference',
                    'phase_cross_correlation.csv'))

cc_file.rename(columns={'offset_x':'x', 'offset_y':'y'}, inplace=True)
cc_file.iloc[:,1] = np.abs(cc_file.iloc[:,1])
cc_file.iloc[:,2] = np.abs(cc_file.iloc[:,2])

# plot 
fig, ax = plt.subplots(figsize=(10,10))
cc_file.iloc[:,1:3].boxplot(ax=ax)
ax.set_xlabel('Direction',fontsize=24)
ax.set_ylabel('Abs. shift (pixels)',fontsize=24)
#ax.set_ylim((-4,4))
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25)

# calculate statistics
mean = np.mean(cc_file.iloc[:,1:3])
std = np.std(cc_file.iloc[:,1:3])
print('raw mean: ', mean)
print('raw std: ', std)


# find results with large misregistration
large_shift_x = cc_file[cc_file.iloc[:,1] > 10]
large_shift_y = cc_file[cc_file.iloc[:,2] > 10]
large_shift = large_shift_x.merge(large_shift_y, 'outer')
large_shift_x = cc_file[cc_file.iloc[:,1] < -10]
large_shift = large_shift.merge(large_shift_x, 'outer')
large_shift_y = cc_file[cc_file.iloc[:,2] < -10]
large_shift = large_shift.merge(large_shift_y, 'outer')

# plot images where registration error was > 10 or < -10
for idx, row in large_shift.iterrows():
    fig, ax = plot_image(data_path, [str(int(row.idx))+'_a.npy'], bands=[3,2,1], normalize=True)
    plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/geometric_misregistration',
                             str(int(row.idx))+'_a.png'))
    fig, ax = plot_image(data_path, [str(int(row.idx))+'_b.npy'], bands=[3,2,1], normalize=True)
    plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/geometric_misregistration',
                         str(int(row.idx))+'_b.png'))
    
# remove all results with misregistration < -10 and > 10
small_shift = cc_file[~cc_file['idx'].isin(large_shift['idx'])]
# n small_shift = 1490

# plot again
fig, ax = plt.subplots(figsize=(10,10))
small_shift.iloc[:,1:3].boxplot(ax=ax)
ax.set_xlabel('Direction',fontsize=24)
ax.set_ylabel('Abs. shift (pixels)',fontsize=24)
ax.set_ylim((-0.1,4))
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25)

# calculate statistics
mean = np.mean(small_shift.iloc[:,1:3])
std = np.std(small_shift.iloc[:,1:3])
print('mean: ', mean)
print('std: ', std)
shift_larger1_x = small_shift[small_shift.iloc[:,1] > 1]
print('# image pairs with shift larger than 1 in the x direction: ', len(shift_larger1_x))
shift_larger1_y = small_shift[small_shift.iloc[:,2] > 1]
print('# image pairs with shift larger than 1 in the y direction: ', len(shift_larger1_y))

#%%
""" Experiment geometric registration OSCD """

# get dataset
directories['data_path'] = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])

dataset = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_train_oscd']))
np.random.seed(567)
dataset = np.random.choice(dataset['im_idx'], len(dataset), replace=False)

dataset_test = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_test_oscd']))
np.random.seed(567)
dataset_test = np.random.choice(dataset_test['im_idx'], len(dataset_test), replace=False)
dataset_settings['indices'] = np.unique(np.concatenate((dataset_test,dataset)))


# init save-files for radiometric difference
fieldnames = ['idx','offset_x', 'offset_y', 'NRMSE', 'diffphase']
if not os.path.exists(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference_oscd',
                    'phase_cross_correlation_oscd.csv')):
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference_oscd',
                    'phase_cross_correlation_oscd.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()

# loop over dataset
for i, idx in enumerate(dataset_settings['indices']):
    if idx == 21:
        im_a = np.load(os.path.join(directories['data_path'], str(idx)+'_a_resampled.npy'))
        im_b = np.load(os.path.join(directories['data_path'], str(idx)+'_b_resampled.npy'))
    elif idx == 22:
        im_a = np.load(os.path.join(directories['data_path'], str(idx)+'_a_resampled.npy'))
        im_b = np.load(os.path.join(directories['data_path'], str(idx)+'_b_resampled.npy'))
    else:
        # load images
        im_a = np.load(os.path.join(directories['data_path'], str(idx)+'_a.npy'))
        im_b = np.load(os.path.join(directories['data_path'], str(idx)+'_b.npy'))
    
    # normalize 
    im_a_rgb = (normalize2plot(im_a[:,:,[3,2,1]])*255).astype(np.uint8)
    im_b_rgb = (normalize2plot(im_b[:,:,[3,2,1]])*255).astype(np.uint8)
    # convert to gray scale
    im_a_gray= cv2.cvtColor(im_a_rgb,cv2.COLOR_RGB2GRAY)
    im_b_gray= cv2.cvtColor(im_b_rgb,cv2.COLOR_RGB2GRAY)
    
    # calculate shift needed for registration (upsample factor 100 for subpixel registration)
    shift, error, diffphase = register_translation(im_a_gray, im_b_gray, upsample_factor=100)  
    offset_x = shift[1]
    offset_y = shift[0]
    
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference_oscd',
                    'phase_cross_correlation_oscd.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
        filewriter.writerow({'idx':idx,
                             'offset_x': offset_x,
                             'offset_y': offset_y,
                             'NRMSE': error,
                             'diffphase': diffphase})  
        
    if (i+1) % 5 == 0:
        print('\r Image pair: [{}/{}]'.format(i+1, len(dataset)), end='')
        
#%%
from matplotlib import pyplot as plt   
from plots import plot_image     
cc_file = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference_oscd',
                    'phase_cross_correlation_oscd.csv'))

cc_file.rename(columns={'offset_x':'x', 'offset_y':'y'}, inplace=True)
cc_file.iloc[:,1] = np.abs(cc_file.iloc[:,1])
cc_file.iloc[:,2] = np.abs(cc_file.iloc[:,2])


# plot 
fig, ax = plt.subplots(figsize=(10,10))
cc_file.iloc[:,1:3].boxplot(ax=ax)
ax.set_xlabel('Direction',fontsize=24)
ax.set_ylabel('Abs. shift (pixels)',fontsize=24)
#ax.set_ylim((-4,4))
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25)

# calculate statistics
mean = np.mean(cc_file.iloc[:,1:3])
std = np.std(cc_file.iloc[:,1:3])
print('raw mean: ', mean)
print('raw std: ', std)


# find results with large misregistration
large_shift_x = cc_file[cc_file.iloc[:,1] > 10]
large_shift_y = cc_file[cc_file.iloc[:,2] > 10]
large_shift = large_shift_x.merge(large_shift_y, 'outer')

data_path = os.path.join(
    directories['intermediate_dir_cd'], 
    directories['data_dir_oscd'])
# plot images where registration error was > 10 or < -10
for idx, row in large_shift.iterrows():
    fig, ax = plot_image(data_path, [str(int(row.idx))+'_a.npy'], bands=[3,2,1], normalize=True)
    plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/geometric_misregistration',
                             str(int(row.idx))+'_a.png'))
    fig, ax = plot_image(data_path, [str(int(row.idx))+'_b.npy'], bands=[3,2,1], normalize=True)
    plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/geometric_misregistration',
                         str(int(row.idx))+'_b.png'))
    
# remove all results with misregistration < -10 and > 10
small_shift = cc_file[~cc_file['idx'].isin(large_shift['idx'])]
# n small_shift = 1490

# plot again
fig, ax = plt.subplots(figsize=(10,10))
small_shift.iloc[:,1:3].boxplot(ax=ax)
ax.set_xlabel('Direction',fontsize=24)
ax.set_ylabel('Shift (pixels)',fontsize=24)
ax.set_ylim((-4,4))
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25)

# calculate statistics
mean = np.mean(small_shift.iloc[:,1:3])
std = np.std(small_shift.iloc[:,1:3])
print('mean: ', mean)
print('std: ', std)
shift_larger1_x = small_shift[small_shift.iloc[:,1] > 1].merge(small_shift[small_shift.iloc[:,1] < -1], 'outer')
print('# image pairs with shift larger than 1 in the x direction: ', len(shift_larger1_x))
shift_larger1_y = small_shift[small_shift.iloc[:,2] > 1].merge(small_shift[small_shift.iloc[:,2] < -1], 'outer')
print('# image pairs with shift larger than 1 in the y direction: ', len(shift_larger1_y))