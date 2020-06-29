#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:10:41 2020

@author: M. Leenstra
"""

# imports
import ee
import numpy as np
import os
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import csv
import math

# variabels
size = 600 #image size in pixels
#im = 0 # image for testing
n_im = 2 # number of images per city
ntry = 20 # number of times to try to get a sentinel band
bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA60']

# directories cities
source_dir_cities = '/media/cordolo/marrit/source_data/worldcities'
results_dir_cities = '/media/cordolo/marrit/results/cities'
cities_csv = 'r_cities.csv'
# directories training
intermediate_dir_training = '/media/cordolo/marrit/Intermediate/training_S2'
results_dir_training = '/media/cordolo/marrit/results/training_S2'
data_dir_S21C = 'data_S21C'
data_dir_S21C_repeat = 'data_S21C_repeat'
data_dir_S21C_repeat2 = 'data_S21C_repeat2'
csv_file_S21C = 'S21C_dataset.csv'
csv_file_S21C_repeat = 'S21C_dataset_repeat.csv'
csv_file_S21C_repeat2 = 'S21C_dataset_repeat2.csv'
# directories test
source_dir_test = '/media/cordolo/marrit/source_data/test_OSCD'
intermediate_dir_test = '/media/cordolo/marrit/Intermediate/CD_OSCD'
data_dir_OSCD = 'data_OSCD'
csv_file_OSCD = 'OSCD_dataset.csv'

#%% 
"""create and preprocess world-cities dataset from geonames"""

import datapackage
import geocoder
import random

# variables 
minPop = 200000 #min population per city
# data from geonames.org via https://datahub.io/core/world-cities
data_url = 'https://datahub.io/core/world-cities/datapackage.json'

# data package
package = datapackage.Package(data_url)

# load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])

data = pd.read_csv(os.path.join(source_dir_cities,'worldcities_geonames.csv'), header=0) 
# combine with worldcity data from simple maps
# worldcities.csv downloaded at 06/01/2020 from https://simplemaps.com/data/world-cities
cities = pd.read_csv(os.path.join(source_dir_cities,'worldcities_simplemaps.csv'), header=0)

# data cleaning
cities_cleaned = cities.drop_duplicates(subset = ['lat', 'lng'])
cities_cleaned = cities_cleaned.dropna(subset = ['lat', 'lng', 'population'])
cities_cleaned = cities_cleaned[cities_cleaned['population'] >= minPop]

# merge
cities_merge = cities_cleaned.merge(data, left_on = ['city','country'], right_on = ['name','country'])
cities_merge = cities_merge.drop('name', axis=1)

# add alternative lat, lng coordinates
cities_merge['Lng_alt'] = np.nan
cities_merge['Lat_alt'] = np.nan

for idx, row in cities_merge.iterrows():
    #time.sleep(1)
    print('\r {}'.format(idx), end='')
    place = geocoder.geonames(row.geonameid, method='details', key='<api_key>')
    if place == None:
        continue
    if place.feature_class != 'P':
        continue
    
    cities_merge.loc[idx,'Lat_alt'] = float(place.lat)
    cities_merge.loc[idx,'Lng_alt'] = float(place.lng)

# check for na's
cities_na = cities_merge[cities_merge['Lat_alt'].isnull()]
# check if na's can be filled with other method
for idx, row in cities_na.iterrows():
    #time.sleep(1)
    print('\r {}'.format(idx), end='')
    place = geocoder.geonames(row.city_ascii+' ('+row.subcountry+')', key='<api_key>')
    if place == None:
        continue
    if place.feature_class != 'P':
        continue
    
    cities_merge.loc[idx,'Lat_alt'] = float(place.lat)
    cities_merge.loc[idx,'Lng_alt'] = float(place.lng)

# if there are still na's drop these
cities_merge = cities_merge.dropna(subset = ['Lng_alt', 'Lat_alt'])

# drop duplicates
cities_merge = cities_merge.drop_duplicates(subset='geonameid')
cities_merge = cities_merge.drop_duplicates(subset=['Lat_alt', 'Lng_alt'])

# save
cities_merge = cities_merge.sample(frac=1).reset_index(drop=True) #shuffle
cities_merge.to_csv(os.path.join(results_dir_cities,'r_cities.csv'), index=False)

#%%
""" Download sentinal 2A 1C training dataset """

from data_download_functions import filterOnCities, eeToNumpy, \
create_featurecollection

if not os.path.isdir(results_dir_training):
    os.makedirs(results_dir_training) 
if not os.path.isdir(os.path.join(results_dir_training, data_dir_S21C)):
    os.makedirs(os.path.join(results_dir_training, data_dir_S21C)) 

# init gee session
#ee.Authenticate()
ee.Initialize()

cities = pd.read_csv(os.path.join(results_dir_cities, cities_csv))
cities = cities.iloc[1722:]

# dict of properties to add to images
properties = {'city_idx': None, 'city': 'city', 'city_ascii': 'city_ascii', \
           'country':'country', 'geonameid': 'geonameid', 'lng': 'Lng_alt', \
           'lat': 'Lat_alt'}
property_keys = list(properties.keys())

# create features
cities_fc = create_featurecollection(cities, coords = {'lng': 'Lng_alt', \
                                    'lat': 'Lat_alt'}, properties = properties)

# get image collection
S21C = ee.ImageCollection('COPERNICUS/S2').filterBounds(cities_fc) \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1) \
    .select(bands)
    
# filter image collection to get 2 random images per city
city_collection = cities_fc.map(filterOnCities(S21C, property_keys, size, n=n_im)) #featurecollection of imagecollection per city
city_list = city_collection.toList(city_collection.size().getInfo()) #List of imagecollection per city

n_cities = len(city_list.getInfo())
print('cityCollection size: ', n_cities)

# prepare csv file
fieldnames = ['im_idx', 'pair_idx', 'im_id', 'filename', 'system_idx', 'city',\
              'city_ascii', 'country', 'geonameid', 'lng', 'lat', 'date',\
              'time', 'error']
if not os.path.exists(os.path.join(results_dir_training, csv_file_S21C)):   
    with open(os.path.join(results_dir_training, csv_file_S21C), 'a') as file:
        writer = csv.DictWriter(file, fieldnames, delimiter = ",")
        writer.writeheader()

# save individual images
tot_time = 0
for im in range(n_cities): 
    
    start_city = time.time()
    imagePair = ee.ImageCollection(city_list.get(im))
    pairsize = imagePair.size().getInfo()
    if pairsize != n_im:
        continue
    images = imagePair.toList(pairsize)
    area = imagePair.geometry()    
    
    for i in range(n_im):
        start_im = time.time()
        imx = ee.Image(images.get(i))
        
        # get some metadata
        metadata = dict()
        metadata['im_idx'] = imx.get('city_idx').getInfo() 
        metadata['pair_idx'] = chr(ord('a')+i)
        metadata['im_id'] = metadata['im_idx']+'_'+ metadata['pair_idx']
        metadata['filename'] = metadata['im_id']+'.npy'
        metadata['system_idx'] = imx.get('system:index').getInfo()
        for prop in property_keys[1:]:
            metadata[prop] = imx.get(prop).getInfo()
        datestr = metadata['system_idx'].split('_')[0]
        metadata['date'] = datestr.split('T')[0]
        metadata['time'] = datestr.split('T')[1]
        metadata['error'] = 'False'

        B = ee.ImageCollection([imx])
        B = ee.Image(B.median())
        
        # covert to numpy array
        try: 
            np_image = eeToNumpy(B, area, bands, ntry=ntry)
            # save
            np.save(os.path.join(results_dir_training,data_dir_S21C, metadata['filename']), np_image)
        except:
            metadata['error'] = 'ERROR'
        
        # save some info in csv-file
        with open(os.path.join(results_dir_training, csv_file_S21C), 'a') as file:
            writer = csv.DictWriter(file, fieldnames, delimiter = ",")
            writer.writerow(metadata)   
        

        end_im= time.time() - start_im
        tot_time = tot_time + end_im
        avg_time = tot_time / ((im*2)+i+1)

        print("\r image {}/{} for city {}/{}. Avg time image (sec): {}".format(i+1,n_im,im+1,n_cities, avg_time), end='')



#%%
""" Preprocess OSCD test dataset """

from data_download_functions import tif_to_numpy
from utils import list_directories

if not os.path.isdir(intermediate_dir_test):
    os.makedirs(intermediate_dir_test)
if not os.path.isdir(os.path.join(intermediate_dir_test, data_dir_OSCD)):
    os.makedirs(os.path.join(intermediate_dir_test, data_dir_OSCD))

# get info about which images are used for training and which for test
source_dir = '/media/cordolo/marrit/source_data/CD_OSCD/data_OSCD'
train_cities = pd.read_csv(os.path.join(source_dir,'train.txt'),header =None).values
test_cities = pd.read_csv(os.path.join(source_dir,'test.txt'),header =None).values

# specify directories 
dirs = list_directories(source_dir)
image_dirs = ['imgs_1','imgs_2']
systemid_dirs = ['imgs_1', 'imgs_2']

# get the images and metadata
for i_city, city in enumerate(dirs): 
    
    # check if training or test image
    if city in train_cities:
        traintest = 'train'
    elif city in test_cities:
        traintest = 'test'
        
    # get dates
    dates = (pd.read_csv(os.path.join(source_dir,city,'dates.txt'),header =None).values).squeeze()
    
    # read and save the images
    for i_dir, image_dir in enumerate(image_dirs): 
        image_path = os.path.join(source_dir, city, image_dir)
        im_bands = os.listdir(image_path)
        n_bands = len(im_bands)
        
        # read images to numpy array
        im = tif_to_numpy(image_path, im_bands, n_bands)
        #plt.imshow(normalize2plot(im[:,:,[3,2,1]]))
        #plt.show()
        #print(im.shape)

        im_id = str(i_city)+'_'+chr(ord('a')+i_dir)
        
        # save
        np.save(os.path.join(intermediate_dir_test,data_dir_OSCD, im_id+'.npy'), im)

        # get systemid
        system_idx= os.listdir(os.path.join(source_dir_test, city, systemid_dirs[i_dir]))[0][:-8]
        
        # save some info in csv-file
        with open(os.path.join(intermediate_dir_test, csv_file_OSCD), 'a') as file:
            writer = csv.writer(file, delimiter = ",")
            writer.writerow([im_id, system_idx, city, dates[i_dir].split(' ')[1], traintest])  
  
# directories dictionary was added at later point in project          
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

# load oscd dataset
oscd = pd.read_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_oscd']))

# add some more columns
im_ids = oscd.loc[:,'im_id'].values
oscd['im_idx'] = [image.split('_')[0] for image in im_ids]
oscd['pair_idx'] = [image.split('_')[1] for image in im_ids]
oscd['filename'] = [image + '.npy' for image in im_ids]
oscd.to_csv(os.path.join(directories['intermediate_dir_cd'],directories['csv_file_oscd']), index=False)

# split train and test and save in separate csv files
oscd_train = oscd[oscd.loc[:,'traintest'] == 'train']
oscd_test = oscd[oscd.loc[:,'traintest'] == 'test']
oscd_train.to_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_train_oscd']), index=False)
oscd_test.to_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_test_oscd']), index=False)

#%%
""" Check the downloaded Sentinel-2 1C images and remove possible errors """
from data_download_functions import filter_errors
from plots import plot_image, plot_random_images
from random import sample
import math

# read csv
dataset = pd.read_csv(os.path.join(results_dir_training, csv_file_S21C), header=0)
   
# filter out the images that did not lead to a download
errors = dataset.loc[dataset.loc[:,'error'] == 'ERROR']
dataset = dataset.loc[dataset.loc[:,'error'] != 'ERROR']

# find the entries with only one image
counts = dataset.groupby(['im_idx']).count()
onescounts = counts[counts.loc[:,'city'] < 2]
onesdataset = dataset.loc[dataset['im_idx'].isin(onescounts.index)]
dataset = dataset.drop(onesdataset.index, axis=0)

# remove duplicates
dataset = dataset.loc[~dataset.duplicated(subset = 'im_id',keep='last')]

# find the entries with only one image
counts = dataset.groupby(['im_idx']).count()
onescounts = counts[counts.loc[:,'city'] < 2]
multicounts = counts[counts.loc[:,'city'] > 2]
onesdataset = dataset.loc[dataset['im_idx'].isin(onescounts.index)]
multidataset = dataset.loc[dataset['im_idx'].isin(multicounts.index)]
dataset = dataset.drop(onesdataset.index, axis=0)
dataset = dataset.drop(multidataset.index, axis=0)

# find corrupt images at threshold 0.5
corrupts = filter_errors(os.path.join(results_dir_training, data_dir_S21C), dataset['filename'].values, threshold=0.4, plot = False)

# find the entries with only one image
corrupts_df = pd.DataFrame(corrupts, columns=['filename'])
corrupt_filenames = corrupts_df.loc[:,'filename'].values
corrupts_df['im_idx'] = [image.split('_')[0] for image in corrupt_filenames]
corrupts_df['pair_idx'] = [image.split('_')[1].split('.')[0] for image in corrupt_filenames]
corrupts_df['im_id'] = [image.split('.')[0] for image in corrupt_filenames]
corrupts_count = corrupts_df.groupby(['im_idx']).count()
corrupts_onescounts = corrupts_count[corrupts_count.loc[:,'im_id'] != 2]
corrupts_onesdataset = corrupts_df.loc[corrupts_df['im_idx'].isin(corrupts_onescounts.index)]
corrupts_twodataset = corrupts_df.drop(corrupts_onesdataset.index, axis=0)

# visually checking the corrupt twos dataset: 
for i in range(math.ceil(len(corrupts_twodataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C), corrupts_twodataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_twodataset['im_id'].values[i*8:(i+1)*8]))
not_corrupts = ['41','114','223','456','540','577','725','874','967','1009','1093','1452','1484','1559','1572','1674']
not_corr = [i+'_a.npy' for i in not_corrupts]
not_corr.extend([i+'_b.npy' for i in not_corrupts])

# visually checking the corrupt ones dataset:
for i in range(math.ceil(len(corrupts_onesdataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C), corrupts_onesdataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_onesdataset['im_id'].values[i*8:(i+1)*8]))
# none of the corrupts_onesdataset is truly corrupt.

# remove the not corrupt images from the corrupts dataset
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(not_corr)]
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(corrupts_onesdataset['filename'])]
# remove corrupt images from datast
dataset = dataset[~dataset['filename'].isin(corrupts_df['filename'])]

# take a random sample to check
rand_im = sample(list(dataset['filename'].values), 8)
fig,ax = plot_image(os.path.join(results_dir_training, data_dir_S21C), rand_im, [3,2,1], axis=False, normalize=True, titles = None)

# look only at 1 image per pair to find duplicate pairs
dataset_a = dataset[dataset['pair_idx'] == 'a']
duplicated = dataset_a[dataset_a.duplicated(subset = ['lng', 'lat'])]['im_idx'].values
# there are still 2 duplicates, even though the geonameid is different. --> remove duplicates
dataset = dataset[~dataset['im_idx'].isin(duplicated)]
dataset.to_csv(os.path.join(results_dir_training, 'S21C_dataset_clean.csv'),index=False)

#%%
"""Repeat the download for the images that were not downloaded correctly """
from data_download_functions import filterOnCities, eeToNumpy, \
create_featurecollection

if not os.path.isdir(results_dir_training):
    os.makedirs(results_dir_training) 
if not os.path.isdir(os.path.join(results_dir_training, data_dir_S21C_repeat)):
    os.makedirs(os.path.join(results_dir_training, data_dir_S21C_repeat)) 

# read csvs
cities = pd.read_csv(os.path.join(results_dir_cities, cities_csv))
dataset = pd.read_csv(os.path.join(results_dir_training, 'S21C_dataset_clean.csv'))

# find out which entries are still todo
done = np.unique(dataset['im_idx'].values)
all = np.arange(1801)
todo = all[~np.isin(all, done)]

# init gee session
#ee.Authenticate()
ee.Initialize()

cities = pd.read_csv(os.path.join(results_dir_cities, cities_csv))
cities = cities.iloc[todo[143:]]
cities['city_idx'] = cities.index

# dict of properties to add to images
properties = {'city_idx': 'city_idx', 'city': 'city', 'city_ascii': 'city_ascii', \
           'country':'country', 'geonameid': 'geonameid', 'lng': 'Lng_alt', \
           'lat': 'Lat_alt'}
property_keys = list(properties.keys())

# create features
cities_fc = create_featurecollection(cities, coords = {'lng': 'Lng_alt', \
                                    'lat': 'Lat_alt'}, properties = properties)

# get image collection
S21C = ee.ImageCollection('COPERNICUS/S2').filterBounds(cities_fc) \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1) \
    .select(bands)
    
# filter image collection to get 2 random images per city
city_collection = cities_fc.map(filterOnCities(S21C, property_keys, size, n=n_im)) #featurecollection of imagecollection per city
city_list = city_collection.toList(city_collection.size().getInfo()) #List of imagecollection per city

n_cities = len(city_list.getInfo())
print('cityCollection size: ', n_cities)

# prepare csv file
fieldnames = ['im_idx', 'pair_idx', 'im_id', 'filename', 'system_idx', 'city',\
              'city_ascii', 'country', 'geonameid', 'lng', 'lat', 'date',\
              'time', 'error']
if not os.path.exists(os.path.join(results_dir_training, csv_file_S21C_repeat)):    
    with open(os.path.join(results_dir_training, csv_file_S21C_repeat), 'a') as file:
        writer = csv.DictWriter(file, fieldnames, delimiter = ",")
        writer.writeheader()

# save individual images
tot_time = 0
for im in range(n_cities): 
    
    start_city = time.time()
    imagePair = ee.ImageCollection(city_list.get(im))
    pairsize = imagePair.size().getInfo()
    if pairsize != n_im:
        continue
    images = imagePair.toList(pairsize)
    area = imagePair.geometry()    
    
    for i in range(n_im):
        start_im = time.time()
        imx = ee.Image(images.get(i))
        
        # get some metadata
        metadata = dict()
        metadata['im_idx'] = str(imx.get('city_idx').getInfo() )
        metadata['pair_idx'] = chr(ord('a')+i)
        metadata['im_id'] = metadata['im_idx']+'_'+ metadata['pair_idx']
        metadata['filename'] = metadata['im_id']+'.npy'
        metadata['system_idx'] = imx.get('system:index').getInfo()
        for prop in property_keys[1:]:
            metadata[prop] = imx.get(prop).getInfo()
        datestr = metadata['system_idx'].split('_')[0]
        metadata['date'] = datestr.split('T')[0]
        metadata['time'] = datestr.split('T')[1]
        metadata['error'] = 'False'

        B = ee.ImageCollection([imx])
        B = ee.Image(B.median())
        
        # covert to numpy array
        try: 
            np_image = eeToNumpy(B, area, bands, ntry=ntry)
            # save
            np.save(os.path.join(results_dir_training,data_dir_S21C_repeat, metadata['filename']), np_image)
        except:
            metadata['error'] = 'ERROR'
        
        # save some info in csv-file
        with open(os.path.join(results_dir_training, csv_file_S21C_repeat), 'a') as file:
            writer = csv.DictWriter(file, fieldnames, delimiter = ",")
            writer.writerow(metadata)   
        

        end_im= time.time() - start_im
        tot_time = tot_time + end_im
        avg_time = tot_time / ((im*2)+i+1)

        print("\r image {}/{} for city {}/{}. Avg time image (sec): {}".format(i+1,n_im,im+1,n_cities, avg_time), end='')


#%%
""" Check the repeated downloaded Sentinel-2 1C images and remove possible errors """
from data_download_functions import filter_errors
from plots import plot_image, plot_random_images
from random import sample
import math

# read csv
dataset = pd.read_csv(os.path.join(results_dir_training, csv_file_S21C_repeat), header=0)
   
# filter out the images that did not lead to a download
errors = dataset.loc[dataset.loc[:,'error'] == 'ERROR']
dataset = dataset.loc[dataset.loc[:,'error'] != 'ERROR']

# remove duplicates
dataset = dataset.loc[~dataset.duplicated(subset = 'im_id',keep='last')]

# find the entries with only one image
counts = dataset.groupby(['im_idx']).count()
onescounts = counts[counts.loc[:,'city'] < 2]
multicounts = counts[counts.loc[:,'city'] > 2]
onesdataset = dataset.loc[dataset['im_idx'].isin(onescounts.index)]
multidataset = dataset.loc[dataset['im_idx'].isin(multicounts.index)]
dataset = dataset.drop(onesdataset.index, axis=0)
dataset = dataset.drop(multidataset.index, axis=0)

# find corrupt images at threshold 0.5
corrupts = filter_errors(os.path.join(results_dir_training, data_dir_S21C_repeat), dataset['filename'].values, threshold=0.5, plot = False)

# find the entries with only one image
corrupts_df = pd.DataFrame(corrupts, columns=['filename'])
corrupt_filenames = corrupts_df.loc[:,'filename'].values
corrupts_df['im_idx'] = [image.split('_')[0] for image in corrupt_filenames]
corrupts_df['pair_idx'] = [image.split('_')[1].split('.')[0] for image in corrupt_filenames]
corrupts_df['im_id'] = [image.split('.')[0] for image in corrupt_filenames]
corrupts_count = corrupts_df.groupby(['im_idx']).count()
corrupts_onescounts = corrupts_count[corrupts_count.loc[:,'im_id'] != 2]
corrupts_onesdataset = corrupts_df.loc[corrupts_df['im_idx'].isin(corrupts_onescounts.index)]
corrupts_twodataset = corrupts_df.drop(corrupts_onesdataset.index, axis=0)

# visually checking the corrupt twos dataset: 
for i in range(math.ceil(len(corrupts_twodataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat), corrupts_twodataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_twodataset['im_id'].values[i*8:(i+1)*8]))
not_corrupts = ['513','597','699','921','1193','1252']
not_corr = [i+'_a.npy' for i in not_corrupts]
not_corr.extend([i+'_b.npy' for i in not_corrupts])

# visually checking the corrupt ones dataset:
for i in range(math.ceil(len(corrupts_onesdataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat), corrupts_onesdataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_onesdataset['im_id'].values[i*8:(i+1)*8]))
# none of the corrupts_onesdataset is truly corrupt.

# remove the not corrupt images from the corrupts dataset
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(not_corr)]
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(corrupts_onesdataset['filename'])]
# remove corrupt images from datast
dataset = dataset[~dataset['filename'].isin(corrupts_df['filename'])]

# take a random sample to check
rand_im = sample(list(dataset['filename'].values), 8)
fig,ax = plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat), rand_im, [3,2,1], axis=False, normalize=True, titles = None)

# look only at 1 image per pair to find duplicate pairs
dataset_a = dataset[dataset['pair_idx'] == 'a']
duplicated = dataset_a[dataset_a.duplicated(subset = ['lng', 'lat'])]['im_idx'].values
# there are still 2 duplicates, even though the geonameid is different. --> remove duplicates
dataset = dataset[~dataset['im_idx'].isin(duplicated)]
dataset.to_csv(os.path.join(results_dir_training, 'S21C_dataset_repeat_clean.csv'),index=False)


# repeat finding corrupts for the onesdataset, to check which we can try one last time to download
# find corrupt images in onesdataset at threshold 0.5
corrupts = filter_errors(os.path.join(results_dir_training, data_dir_S21C_repeat), onesdataset['filename'].values, threshold=0.5, plot = False)

# find the entries with only one image
corrupts_df = pd.DataFrame(corrupts, columns=['filename'])
corrupt_filenames = corrupts_df.loc[:,'filename'].values
corrupts_df['im_idx'] = [image.split('_')[0] for image in corrupt_filenames]
corrupts_df['pair_idx'] = [image.split('_')[1].split('.')[0] for image in corrupt_filenames]
corrupts_df['im_id'] = [image.split('.')[0] for image in corrupt_filenames]
corrupts_count = corrupts_df.groupby(['im_idx']).count()
corrupts_onescounts = corrupts_count[corrupts_count.loc[:,'im_id'] != 2]
corrupts_onesdataset = corrupts_df.loc[corrupts_df['im_idx'].isin(corrupts_onescounts.index)]


# visually checking the corrupt ones dataset:
for i in range(math.ceil(len(corrupts_onesdataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat), corrupts_onesdataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_onesdataset['im_id'].values[i*8:(i+1)*8]))
not_corrupts = ['537','1168','899']
not_corr = [i+'_a.npy' for i in not_corrupts]
not_corr.extend([i+'_b.npy' for i in not_corrupts])

# remove the not corrupt images from the corrupts dataset
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(not_corr)]
# remove corrupt images from onesdatast
onesdataset = onesdataset[~onesdataset['filename'].isin(corrupts_df['filename'])]

# images that were not corrupt and were in the onesdataset get one more chance
todo = onesdataset.im_idx.values
todo = pd.DataFrame(todo, columns=['im_id'])
todo.to_csv(os.path.join(results_dir_training, 'todo.csv'),index=False)





#%%
"""Second Repeat the download for the images that were not downloaded correctly """
from data_download_functions import filterOnCities, eeToNumpy, \
create_featurecollection

if not os.path.isdir(results_dir_training):
    os.makedirs(results_dir_training) 
if not os.path.isdir(os.path.join(results_dir_training, data_dir_S21C_repeat2)):
    os.makedirs(os.path.join(results_dir_training, data_dir_S21C_repeat2)) 

# read csvs
cities = pd.read_csv(os.path.join(results_dir_cities, cities_csv))
todo = pd.read_csv(os.path.join(results_dir_training, 'todo.csv'))
todo = todo.im_id.values
#dataset = pd.read_csv(os.path.join(results_dir_training, 'S21C_dataset_clean.csv'))

# init gee session
#ee.Authenticate()
ee.Initialize()

cities = pd.read_csv(os.path.join(results_dir_cities, cities_csv))
cities = cities.iloc[todo]
cities['city_idx'] = cities.index

# dict of properties to add to images
properties = {'city_idx': 'city_idx', 'city': 'city', 'city_ascii': 'city_ascii', \
           'country':'country', 'geonameid': 'geonameid', 'lng': 'Lng_alt', \
           'lat': 'Lat_alt'}
property_keys = list(properties.keys())

# create features
cities_fc = create_featurecollection(cities, coords = {'lng': 'Lng_alt', \
                                    'lat': 'Lat_alt'}, properties = properties)

# get image collection
S21C = ee.ImageCollection('COPERNICUS/S2').filterBounds(cities_fc) \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1) \
    .select(bands)
    
# filter image collection to get 2 random images per city
city_collection = cities_fc.map(filterOnCities(S21C, property_keys, size, n=n_im)) #featurecollection of imagecollection per city
city_list = city_collection.toList(city_collection.size().getInfo()) #List of imagecollection per city

n_cities = len(city_list.getInfo())
print('cityCollection size: ', n_cities)

# prepare csv file
fieldnames = ['im_idx', 'pair_idx', 'im_id', 'filename', 'system_idx', 'city',\
              'city_ascii', 'country', 'geonameid', 'lng', 'lat', 'date',\
              'time', 'error']
if not os.path.exists(os.path.join(results_dir_training, csv_file_S21C_repeat2)): 
    with open(os.path.join(results_dir_training, csv_file_S21C_repeat2), 'a') as file:
         writer = csv.DictWriter(file, fieldnames, delimiter = ",")
         writer.writeheader()

# save individual images
tot_time = 0
for im in range(n_cities): 
    
    start_city = time.time()
    imagePair = ee.ImageCollection(city_list.get(im))
    pairsize = imagePair.size().getInfo()
    if pairsize != n_im:
        continue
    images = imagePair.toList(pairsize)
    area = imagePair.geometry()    
    
    for i in range(n_im):
        start_im = time.time()
        imx = ee.Image(images.get(i))
        
        # get some metadata
        metadata = dict()
        metadata['im_idx'] = str(imx.get('city_idx').getInfo() )
        metadata['pair_idx'] = chr(ord('a')+i)
        metadata['im_id'] = metadata['im_idx']+'_'+ metadata['pair_idx']
        metadata['filename'] = metadata['im_id']+'.npy'
        metadata['system_idx'] = imx.get('system:index').getInfo()
        for prop in property_keys[1:]:
            metadata[prop] = imx.get(prop).getInfo()
        datestr = metadata['system_idx'].split('_')[0]
        metadata['date'] = datestr.split('T')[0]
        metadata['time'] = datestr.split('T')[1]
        metadata['error'] = 'False'

        B = ee.ImageCollection([imx])
        B = ee.Image(B.median())
        
        # covert to numpy array
        try: 
            np_image = eeToNumpy(B, area, bands, ntry=ntry)
            # save
            np.save(os.path.join(results_dir_training,data_dir_S21C_repeat2, metadata['filename']), np_image)
        except:
            metadata['error'] = 'ERROR'
        
        # save some info in csv-file
        with open(os.path.join(results_dir_training, csv_file_S21C_repeat2), 'a') as file:
            writer = csv.DictWriter(file, fieldnames, delimiter = ",")
            writer.writerow(metadata)   
        

        end_im= time.time() - start_im
        tot_time = tot_time + end_im
        avg_time = tot_time / ((im*2)+i+1)

        print("\r image {}/{} for city {}/{}. Avg time image (sec): {}".format(i+1,n_im,im+1,n_cities, avg_time), end='')



#%%
""" Check the second set repeated downloaded Sentinel-2 1C images and remove possible errors """
from data_download_functions import filter_errors
from plots import plot_image, plot_random_images
from random import sample
import math

# read csv
dataset = pd.read_csv(os.path.join(results_dir_training, csv_file_S21C_repeat2), header=0)
   
# filter out the images that did not lead to a download
errors = dataset.loc[dataset.loc[:,'error'] == 'ERROR']
dataset = dataset.loc[dataset.loc[:,'error'] != 'ERROR']

# remove duplicates
dataset = dataset.loc[~dataset.duplicated(subset = 'im_id',keep='last')]

# find the entries with only one image
counts = dataset.groupby(['im_idx']).count()
onescounts = counts[counts.loc[:,'city'] < 2]
multicounts = counts[counts.loc[:,'city'] > 2]
onesdataset = dataset.loc[dataset['im_idx'].isin(onescounts.index)]
multidataset = dataset.loc[dataset['im_idx'].isin(multicounts.index)]
dataset = dataset.drop(onesdataset.index, axis=0)
dataset = dataset.drop(multidataset.index, axis=0)

# find corrupt images at threshold 0.5
corrupts = filter_errors(os.path.join(results_dir_training, data_dir_S21C_repeat2), dataset['filename'].values, threshold=0.5, plot = False)

# find the entries with only one image
corrupts_df = pd.DataFrame(corrupts, columns=['filename'])
corrupt_filenames = corrupts_df.loc[:,'filename'].values
corrupts_df['im_idx'] = [image.split('_')[0] for image in corrupt_filenames]
corrupts_df['pair_idx'] = [image.split('_')[1].split('.')[0] for image in corrupt_filenames]
corrupts_df['im_id'] = [image.split('.')[0] for image in corrupt_filenames]
corrupts_count = corrupts_df.groupby(['im_idx']).count()
corrupts_onescounts = corrupts_count[corrupts_count.loc[:,'im_id'] != 2]
corrupts_onesdataset = corrupts_df.loc[corrupts_df['im_idx'].isin(corrupts_onescounts.index)]
corrupts_twodataset = corrupts_df.drop(corrupts_onesdataset.index, axis=0)

# visually checking the corrupt twos dataset: 
for i in range(math.ceil(len(corrupts_twodataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat2), corrupts_twodataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_twodataset['im_id'].values[i*8:(i+1)*8]))
not_corrupts = ['537','899']
not_corr = [i+'_a.npy' for i in not_corrupts]
not_corr.extend([i+'_b.npy' for i in not_corrupts])

# visually checking the corrupt ones dataset:
for i in range(math.ceil(len(corrupts_onesdataset)/8)):
    plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat2), corrupts_onesdataset['filename'].values[i*8:(i+1)*8], [8,4,3],titles = list(corrupts_onesdataset['im_id'].values[i*8:(i+1)*8]))
# none of the corrupts_onesdataset is truly corrupt.

# remove the not corrupt images from the corrupts dataset
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(not_corr)]
corrupts_df = corrupts_df[~corrupts_df['filename'].isin(corrupts_onesdataset['filename'])]
# remove corrupt images from datast
dataset = dataset[~dataset['filename'].isin(corrupts_df['filename'])]

# take a random sample to check
rand_im = sample(list(dataset['filename'].values), 8)
fig,ax = plot_image(os.path.join(results_dir_training, data_dir_S21C_repeat2), rand_im, [3,2,1], axis=False, normalize=True, titles = None)

# look only at 1 image per pair to find duplicate pairs
dataset_a = dataset[dataset['pair_idx'] == 'a']
duplicated = dataset_a[dataset_a.duplicated(subset = ['lng', 'lat'])]['im_idx'].values
# there are still 2 duplicates, even though the geonameid is different. --> remove duplicates
dataset = dataset[~dataset['im_idx'].isin(duplicated)]
dataset.to_csv(os.path.join(results_dir_training, 'S21C_dataset_repeat2_clean.csv'),index=False)


#%%
""" combine datasets """
import shutil
dataset = pd.read_csv(os.path.join(intermediate_dir_training,'S21C_dataset_clean.csv'), header=0)
dataset1 = pd.read_csv(os.path.join(intermediate_dir_training,'S21C_dataset_repeat_clean.csv'), header=0)
dataset2 = pd.read_csv(os.path.join(intermediate_dir_training,'S21C_dataset_repeat2_clean.csv'), header=0)

data = pd.concat([dataset, dataset1, dataset2])

# look only at 1 image per pair to find duplicate pairs
data_a = data[data['pair_idx'] == 'a']
duplicated = data_a[data_a.duplicated(subset = ['lng', 'lat'])]['im_idx'].values
data = data[~data['im_idx'].isin(duplicated)]
dataset1 = dataset1[~dataset1['im_idx'].isin(duplicated)]

# save csv
data.to_csv(os.path.join(results_dir_training, csv_file_S21C))

# move the files to the S21C datafolder
if not os.path.isdir(os.path.join(results_dir_training, data_dir_S21C)):
    os.makedirs(os.path.join(results_dir_training, data_dir_S21C)) 

for im in dataset[dataset['im_idx'] > 833]['filename']:
    src = os.path.join(intermediate_dir_training, data_dir_S21C, im)
    dst = os.path.join(results_dir_training, data_dir_S21C, im)
    shutil.copy(src, dst)

for im in dataset1['filename']:
    src = os.path.join(intermediate_dir_training, 'data_S21C_repeat', im)
    dst = os.path.join(results_dir_training, data_dir_S21C, im)
    shutil.copy(src, dst)
    
for im in dataset2['filename']:
    src = os.path.join(intermediate_dir_training, 'data_S21C_repeat2', im)
    dst = os.path.join(results_dir_training, data_dir_S21C, im)
    shutil.copy(src, dst)    
    


#%%
""" Preprocess OSCD labels """

from data_download_functions import tif_to_numpy
from utils import list_directories
import gdal

# directories dictionary was added at later point in project          
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
    'labels_dir_oscd' : 'labels_OSCD',
    'tb_dir': '/media/cordolo/elements/Intermediate/tensorboard',
    'model_dir': '/media/cordolo/elements/Intermediate/trained_models'}

labels_dir_oscd = directories['labels_dir_oscd']

if not os.path.isdir(intermediate_dir_test):
    os.makedirs(intermediate_dir_test)
if not os.path.isdir(os.path.join(intermediate_dir_test, labels_dir_oscd)):
    os.makedirs(os.path.join(intermediate_dir_test, labels_dir_oscd))

# get info about which images are used for training and which for test
oscd_train = pd.read_csv(os.path.join(directories['intermediate_dir_cd'], directories['csv_file_train_oscd']))

# specify directories 
dirs = list_directories(os.path.join(source_dir_test,'labels'))
#image_dirs = ['cm']


# read and save the ground truth   
for city in dirs: 
            
    # get path
    image_path = os.path.join(source_dir_test, 'labels',city, 'cm',city+'-cm.tif')

    # oepn and load to numpy array   
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    gt = ds.GetRasterBand(1).ReadAsArray()
   
    # reset
    ds = None
    
    # get correct index
    im_idx = oscd_train[oscd_train['city'] == city].iloc[0].loc['im_idx']
           
    # save
    np.save(os.path.join(intermediate_dir_test,labels_dir_oscd, str(im_idx)+'.npy'), gt)


