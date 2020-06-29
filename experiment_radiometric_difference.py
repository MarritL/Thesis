#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:23:51 2020

@author: M. Leenstra
"""

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#np.random.seed(321)
#dataset = np.random.choice(dataset['im_idx'], 10, replace=False)

# init save-files for radiometric difference
fieldnames = ['idx','B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60']
if not os.path.exists(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                    'rmse_pixellevel.csv')):
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                    'rmse_pixellevel.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()
if not os.path.exists(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                    'diff_band_mean.csv')):
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                    'diff_band_mean.csv'), 'a') as file2:
        filewriter = csv.DictWriter(file2, fieldnames, delimiter = ",")
        filewriter.writeheader()

# loop over dataset
for i, idx in enumerate(dataset['im_idx']):
    # load images
    im_a = np.load(os.path.join(directories['results_dir_training'],directories['data_dir_S21C'], str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(directories['results_dir_training'],directories['data_dir_S21C'], str(idx)+'_b.npy'))
   
    # calculate pixelwise radiometric difference
    rmse_pixelwise = np.sqrt(np.mean(np.square(im_a-im_b), axis=(0,1)))
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                    'rmse_pixellevel.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
        filewriter.writerow({'idx':idx,
                             'B1':rmse_pixelwise[0],
                             'B2':rmse_pixelwise[1],
                             'B3':rmse_pixelwise[2],
                             'B4':rmse_pixelwise[3],
                             'B5':rmse_pixelwise[4],
                             'B6':rmse_pixelwise[5],
                             'B7':rmse_pixelwise[6],
                             'B8':rmse_pixelwise[7],
                             'B8A':rmse_pixelwise[8],
                             'B9':rmse_pixelwise[9],
                             'B10':rmse_pixelwise[10],
                             'B11':rmse_pixelwise[11],
                             'B12':rmse_pixelwise[12],
                             'QA60':rmse_pixelwise[13]})  
    
    # calculate band means (if geometric difference is present, pixelwise radiometric difference not correct)
    band_mean_im_a = np.mean(im_a,axis=(0,1))
    band_mean_im_b = np.mean(im_b, axis=(0,1))
    diff_band_mean = band_mean_im_a-band_mean_im_b
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                'diff_band_mean.csv'), 'a') as file2:
        filewriter = csv.DictWriter(file2, fieldnames, delimiter = ",", extrasaction='ignore')
        filewriter.writerow({'idx':idx,
                             'B1':diff_band_mean[0],
                             'B2':diff_band_mean[1],
                             'B3':diff_band_mean[2],
                             'B4':diff_band_mean[3],
                             'B5':diff_band_mean[4],
                             'B6':diff_band_mean[5],
                             'B7':diff_band_mean[6],
                             'B8':diff_band_mean[7],
                             'B8A':diff_band_mean[8],
                             'B9':diff_band_mean[9],
                             'B10':diff_band_mean[10],
                             'B11':diff_band_mean[11],
                             'B12':diff_band_mean[12],
                             'QA60':diff_band_mean[13]})  
    
    if (i+1) % 5 == 0:
        print('\r Image pair: [{}/{}]'.format(i+1, len(dataset)), end='')

#%%
from plots import plot_image
rmse_pixelwise = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                    'rmse_pixellevel.csv'))
# remove entries with non zero difference in QA60 (clouds)
rmse_pixelwise_cleaned = rmse_pixelwise[rmse_pixelwise.iloc[:,-1] == 0]


diff_band_mean = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference',
                'diff_band_mean.csv'))
mean_rmse_pixelwise = np.mean(rmse_pixelwise)
mean_rmse_bandwise = np.sqrt(np.mean(np.square(diff_band_mean.iloc[:,1:]),axis=0))

# plot of RMSE without cleaning
fig, ax = plt.subplots(figsize=(10,10))
rmse_pixelwise.iloc[:,1:-1].boxplot(ax=ax)
ax.set_xlabel('Band', fontsize=24)
ax.set_ylabel('RMSE (DN)', fontsize=24)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-200,7000)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# calculate statistics
mean = np.mean(rmse_pixelwise.iloc[:,1:-1], axis=0)
std = np.std(rmse_pixelwise.iloc[:,1:-1], axis=0)
print('mean: ', mean/100)
print('std: ', std/100)


# now remove images 
cc_file = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference',
                    'phase_cross_correlation.csv'))

cc_file.rename(columns={'offset_x':'x', 'offset_y':'y'}, inplace=True)
cc_file.iloc[:,1] = np.abs(cc_file.iloc[:,1])
cc_file.iloc[:,2] = np.abs(cc_file.iloc[:,2])

# find results with large misregistration
large_shift_x = cc_file[cc_file.iloc[:,1] > 10]
large_shift_y = cc_file[cc_file.iloc[:,2] > 10]
large_shift = large_shift_x.merge(large_shift_y, 'outer') 
# remove all results with misregistration < -10 and > 10
small_shift = cc_file[~cc_file['idx'].isin(large_shift['idx'])]
# n small_shift = 1490

shift_larger1_x = small_shift[small_shift.iloc[:,1] > 1]
shift_larger1_y = small_shift[small_shift.iloc[:,2] > 1]
smaller_shift = small_shift[~small_shift['idx'].isin(shift_larger1_x['idx'])]
smaller_shift = smaller_shift[~smaller_shift['idx'].isin(shift_larger1_y['idx'])]
# n smaller_shift = 1329

 
# remove shift larger than 1 from assessment
rmse_pixel_registered = rmse_pixelwise[rmse_pixelwise['idx'].isin(smaller_shift['idx'])]

# plot again 
fig, ax = plt.subplots(figsize=(10,10))
rmse_pixel_registered.iloc[:,1:-1].boxplot(ax=ax)
ax.set_xlabel('Band', fontsize=24)
ax.set_ylabel('RMSE (DN)', fontsize=24)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-200,7000)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# calculate statistics
mean2 = np.mean(rmse_pixel_registered.iloc[:,1:-1], axis=0)
std2 = np.std(rmse_pixel_registered.iloc[:,1:-1], axis=0)
print('mean: ', mean2/100)
print('std: ', std2/100)

# diff
diff_mean = mean - mean2
diff_std = std - std2
print('mean mean diff: ', np.mean(diff_mean))
print('mean std diff: ', np.mean(diff_std))




#%%
# get dataset
data_path = os.path.join(
    directories['results_dir_training'], 
    directories['data_dir_S21C'])

dataset = pd.read_csv(os.path.join(directories['results_dir_training'],directories['csv_file_S21C']))

# check some images with more than rmse>5000
rmse_pixel_ge_1000 =  rmse_pixel_registered[rmse_pixel_registered.iloc[:,11] > 500]
rmse_pixel_1000 = rmse_pixel_ge_1000[rmse_pixel_ge_1000.iloc[:,10] < 2000]
rmse_pixel_ge_2000 =  rmse_pixel_registered[rmse_pixel_registered.iloc[:,10] > 2000]
rmse_pixel_2000 = rmse_pixel_ge_2000[rmse_pixel_ge_2000.iloc[:,4] < 3000]
rmse_pixel_ge_3000 =  rmse_pixel_registered[rmse_pixel_registered.iloc[:,4] > 3000]
rmse_pixel_3000 = rmse_pixel_ge_3000[rmse_pixel_ge_3000.iloc[:,4] < 4000]
rmse_pixel_ge_4000 =  rmse_pixel_registered[rmse_pixel_registered.iloc[:,4] > 4000]
rmse_pixel_4000 = rmse_pixel_ge_4000[rmse_pixel_ge_4000.iloc[:,4] < 5000]
rmse_pixel_5000 =  rmse_pixel_registered[rmse_pixel_registered.iloc[:,3] > 5000]


# samples >1000, <2000: #1325
idx = str(1325)
fig, ax = plot_image(data_path, [idx+'_a.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_a.png'))
fig, ax = plot_image(data_path, [idx+'_b.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_b.png'))
arr = rmse_pixel_registered[rmse_pixel_registered['idx'] == int(idx)].values
print('idx: ', arr[0][0], '\n',
      'B1: ', arr[0][1], '\n',
      'B2: ', arr[0][2], '\n',
      'B3: ', arr[0][3], '\n',
      'B4: ', arr[0][4], '\n',
      'B5: ', arr[0][5], '\n',
      'B6: ', arr[0][6], '\n',
      'B7: ', arr[0][7], '\n',
      'B8: ', arr[0][8], '\n',
      'B8A: ', arr[0][9], '\n',
      'B9: ', arr[0][10], '\n',
      'B10: ', arr[0][11], '\n',
      'B11: ', arr[0][12], '\n',
      'B12: ', arr[0][13], '\n',
      'QA60: ', arr[0][14], '\n')
print('mean: ', np.mean(arr[0][1:-1]))
print('city: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].city)
print('country: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].country)
print('dates: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].date)

# samples >2000, <3000: #724
idx = str(724)
fig, ax = plot_image(data_path, [idx+'_a.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_a.png'))
fig, ax = plot_image(data_path, [idx+'_b.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_b.png'))
arr = rmse_pixel_registered[rmse_pixel_registered['idx'] == int(idx)].values
print('idx: ', arr[0][0], '\n',
      'B1: ', arr[0][1], '\n',
      'B2: ', arr[0][2], '\n',
      'B3: ', arr[0][3], '\n',
      'B4: ', arr[0][4], '\n',
      'B5: ', arr[0][5], '\n',
      'B6: ', arr[0][6], '\n',
      'B7: ', arr[0][7], '\n',
      'B8: ', arr[0][8], '\n',
      'B8A: ', arr[0][9], '\n',
      'B9: ', arr[0][10], '\n',
      'B10: ', arr[0][11], '\n',
      'B11: ', arr[0][12], '\n',
      'B12: ', arr[0][13], '\n',
      'QA60: ', arr[0][14], '\n')
print('mean: ', np.mean(arr[0][1:-1]))
print('city: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].city)
print('country: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].country)
print('dates: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].date)

# samples >3000, <4000: #354
idx = str(354)
fig, ax = plot_image(data_path, [idx+'_a.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_a.png'))
fig, ax = plot_image(data_path, [idx+'_b.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_b.png'))
arr = rmse_pixel_registered[rmse_pixel_registered['idx'] == int(idx)].values
print('idx: ', arr[0][0], '\n',
      'B1: ', arr[0][1], '\n',
      'B2: ', arr[0][2], '\n',
      'B3: ', arr[0][3], '\n',
      'B4: ', arr[0][4], '\n',
      'B5: ', arr[0][5], '\n',
      'B6: ', arr[0][6], '\n',
      'B7: ', arr[0][7], '\n',
      'B8: ', arr[0][8], '\n',
      'B8A: ', arr[0][9], '\n',
      'B9: ', arr[0][10], '\n',
      'B10: ', arr[0][11], '\n',
      'B11: ', arr[0][12], '\n',
      'B12: ', arr[0][13], '\n',
      'QA60: ', arr[0][14], '\n')



# samples >4000, <5000: #1675
idx = str(1675)
fig, ax = plot_image(data_path, [idx+'_a.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_a.png'))
fig, ax = plot_image(data_path, [idx+'_b.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_b.png'))
arr = rmse_pixel_registered[rmse_pixel_registered['idx'] == int(idx)].values
print('idx: ', arr[0][0], '\n',
      'B1: ', arr[0][1], '\n',
      'B2: ', arr[0][2], '\n',
      'B3: ', arr[0][3], '\n',
      'B4: ', arr[0][4], '\n',
      'B5: ', arr[0][5], '\n',
      'B6: ', arr[0][6], '\n',
      'B7: ', arr[0][7], '\n',
      'B8: ', arr[0][8], '\n',
      'B8A: ', arr[0][9], '\n',
      'B9: ', arr[0][10], '\n',
      'B10: ', arr[0][11], '\n',
      'B11: ', arr[0][12], '\n',
      'B12: ', arr[0][13], '\n',
      'QA60: ', arr[0][14], '\n')
print('mean: ', np.mean(arr[0][1:-1]))
print('city: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].city)
print('country: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].country)
print('dates: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].date)

# samples >5000: #751
idx = str(751)
fig, ax = plot_image(data_path, [idx+'_a.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_a.png'))
fig, ax = plot_image(data_path, [idx+'_b.npy'], bands=[3,2,1], normalize=True) 
plt.savefig(os.path.join('/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/images_final/radiometric_large_diff',
                         idx+'_b.png'))
arr = rmse_pixel_registered[rmse_pixel_registered['idx'] == int(idx)].values
print('idx: ', arr[0][0], '\n',
      'B1: ', arr[0][1], '\n',
      'B2: ', arr[0][2], '\n',
      'B3: ', arr[0][3], '\n',
      'B4: ', arr[0][4], '\n',
      'B5: ', arr[0][5], '\n',
      'B6: ', arr[0][6], '\n',
      'B7: ', arr[0][7], '\n',
      'B8: ', arr[0][8], '\n',
      'B8A: ', arr[0][9], '\n',
      'B9: ', arr[0][10], '\n',
      'B10: ', arr[0][11], '\n',
      'B11: ', arr[0][12], '\n',
      'B12: ', arr[0][13], '\n')
print('mean: ', np.mean(arr[0][1:-1]))
print('city: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].city)
print('country: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].country)
print('dates: ', dataset[dataset.loc[:,'im_idx'] == int(idx)].date)


#%%
""" Experiment radiometric difference OSCD dataset """
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
fieldnames = ['idx','B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
if not os.path.exists(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                    'rmse_pixellevel_oscd.csv')):
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                    'rmse_pixellevel_oscd.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
        filewriter.writeheader()
if not os.path.exists(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                    'diff_band_mean_oscd.csv')):
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                    'diff_band_mean_oscd.csv'), 'a') as file2:
        filewriter = csv.DictWriter(file2, fieldnames, delimiter = ",")
        filewriter.writeheader()

# loop over dataset
for i, idx in enumerate(dataset_settings['indices']):
    # load image 
    im_a = np.load(os.path.join(directories['data_path'], str(idx)+'_a.npy'))
    im_b = np.load(os.path.join(directories['data_path'], str(idx)+'_b.npy'))
   
    # calculate pixelwise radiometric difference
    rmse_pixelwise = np.sqrt(np.mean(np.square(im_a-im_b), axis=(0,1)))
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                    'rmse_pixellevel_oscd.csv'), 'a') as file:
        filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
        filewriter.writerow({'idx':idx,
                             'B1':rmse_pixelwise[0],
                             'B2':rmse_pixelwise[1],
                             'B3':rmse_pixelwise[2],
                             'B4':rmse_pixelwise[3],
                             'B5':rmse_pixelwise[4],
                             'B6':rmse_pixelwise[5],
                             'B7':rmse_pixelwise[6],
                             'B8':rmse_pixelwise[7],
                             'B8A':rmse_pixelwise[8],
                             'B9':rmse_pixelwise[9],
                             'B10':rmse_pixelwise[10],
                             'B11':rmse_pixelwise[11],
                             'B12':rmse_pixelwise[12]})  
    
    # calculate band means (if geometric difference is present, pixelwise radiometric difference not correct)
    band_mean_im_a = np.mean(im_a,axis=(0,1))
    band_mean_im_b = np.mean(im_b, axis=(0,1))
    diff_band_mean = band_mean_im_a-band_mean_im_b
    with open(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                'diff_band_mean_oscd.csv'), 'a') as file2:
        filewriter = csv.DictWriter(file2, fieldnames, delimiter = ",", extrasaction='ignore')
        filewriter.writerow({'idx':idx,
                             'B1':diff_band_mean[0],
                             'B2':diff_band_mean[1],
                             'B3':diff_band_mean[2],
                             'B4':diff_band_mean[3],
                             'B5':diff_band_mean[4],
                             'B6':diff_band_mean[5],
                             'B7':diff_band_mean[6],
                             'B8':diff_band_mean[7],
                             'B8A':diff_band_mean[8],
                             'B9':diff_band_mean[9],
                             'B10':diff_band_mean[10],
                             'B11':diff_band_mean[11],
                             'B12':diff_band_mean[12]})  
    
    if (i+1) % 5 == 0:
        print('\r Image pair: [{}/{}]'.format(i+1, len(dataset_settings['indices'])), end='')

#%%
from plots import plot_image
rmse_pixelwise = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                    'rmse_pixellevel_oscd.csv'))
# remove entries with non zero difference in QA60 (clouds)
rmse_pixelwise_cleaned = rmse_pixelwise[rmse_pixelwise.iloc[:,-1] == 0]


diff_band_mean = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_radiometric_difference_oscd',
                'diff_band_mean_oscd.csv'))
mean_rmse_pixelwise = np.mean(rmse_pixelwise)
mean_rmse_bandwise = np.sqrt(np.mean(np.square(diff_band_mean.iloc[:,1:]),axis=0))

# plot of RMSE without cleaning
fig, ax = plt.subplots(figsize=(10,10))
rmse_pixelwise.iloc[:,1:].boxplot(ax=ax)
ax.set_xlabel('Band', fontsize=24)
ax.set_ylabel('RMSE (DN)', fontsize=24)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-200,7000)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# calculate statistics
mean = np.mean(rmse_pixelwise.iloc[:,1:], axis=0)
std = np.std(rmse_pixelwise.iloc[:,1:], axis=0)
print('mean: ', mean/100)
print('std: ', std/100)


# now remove images 
cc_file = pd.read_csv(os.path.join('/media/cordolo/marrit/results/experiments_dataset/experiment_geometric_difference_oscd',
                    'phase_cross_correlation_oscd.csv'))

cc_file.rename(columns={'offset_x':'x', 'offset_y':'y'}, inplace=True)
cc_file.iloc[:,1] = np.abs(cc_file.iloc[:,1])
cc_file.iloc[:,2] = np.abs(cc_file.iloc[:,2])

# find results with large misregistration
large_shift_x = cc_file[cc_file.iloc[:,1] > 10]
large_shift_y = cc_file[cc_file.iloc[:,2] > 10]
large_shift = large_shift_x.merge(large_shift_y, 'outer')
large_shift_x = cc_file[cc_file.iloc[:,1] < -10]
large_shift = large_shift.merge(large_shift_x, 'outer')
large_shift_y = cc_file[cc_file.iloc[:,2] < -10]
large_shift = large_shift.merge(large_shift_y, 'outer')    
# remove all results with misregistration < -10 and > 10
small_shift = cc_file[~cc_file['idx'].isin(large_shift['idx'])]
# n small_shift = 1490

shift_larger1_x = small_shift[small_shift.iloc[:,1] > 1].merge(small_shift[small_shift.iloc[:,1] < -1], 'outer')
shift_larger1_y = small_shift[small_shift.iloc[:,2] > 1].merge(small_shift[small_shift.iloc[:,2] < -1], 'outer')
smaller_shift = small_shift[~small_shift['idx'].isin(shift_larger1_x['idx'])]
smaller_shift = smaller_shift[~smaller_shift['idx'].isin(shift_larger1_y['idx'])]
# n smaller_shift = 1329

 
# remove shift larger than 1 from assessment
rmse_pixel_registered = rmse_pixelwise[rmse_pixelwise['idx'].isin(smaller_shift['idx'])]

# plot again 
fig, ax = plt.subplots(figsize=(10,10))
rmse_pixel_registered.iloc[:,1:].boxplot(ax=ax)
ax.set_xlabel('Band', fontsize=24)
ax.set_ylabel('RMSE (DN)', fontsize=24)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(24) 
#removing top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-200,7000)
#add major gridlines
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# calculate statistics
mean2 = np.mean(rmse_pixel_registered.iloc[:,1:], axis=0)
std2 = np.std(rmse_pixel_registered.iloc[:,1:], axis=0)
print('mean: ', mean2/100)
print('std: ', std2/100)

# diff
diff_mean = mean - mean2
diff_std = std - std2
print('mean mean diff: ', np.mean(diff_mean))
print('mean std diff: ', np.mean(diff_std))

