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
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import csv

# 'global' variables

size = 600 #image size in pixels
im = 0 # image for testing
n_im = 2 # number of images per city
ntry = 5 # number of times to try to get a sentinel band
bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
data_dir = '/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/Data'
source_dir_cities = '/media/cordolo/elements/source_data/cities'
results_dir_cities = '/media/cordolo/elements/results/cities'
intermediate_dir_training = '/media/cordolo/elements/Intermediate/training_S2'
source_dir_test = '/media/cordolo/elements/source_data/test_OSCD'
intermediate_dir_test = '/media/cordolo/elements/Intermediate/test_OSCD'
data_out = '/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/dataout/'
#csv_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/S21C_dataset.csv'
#data_out = '/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/data/'
#csv_file = '/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/S21C_dataset.csv'

#%% 
"""create and preprocess world-cities dataset from geonames"""

import datapackage
import geocoder
import random

# variables 
minPop = 200000 #min population per city
data_url = 'https://datahub.io/core/world-cities/datapackage.json'

# data package
package = datapackage.Package(data_url)

# load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
   
# combine with worldcity data from simple maps
# worldcities.csv downloaded at 06/01/2020 from https://simplemaps.com/data/world-cities
cities = pd.read_csv(os.path.join(source_dir_cities,'worldcities.csv'), header=0)

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

#if there are still na's drop these
cities_merge = cities_merge.dropna(subset = ['Lng_alt', 'Lat_alt'])

# save
cities_merge = cities_merge.sample(frac=1).reset_index(drop=True)
cities_merge.to_csv(os.path.join(results_dir_cities,'r_cities.csv'), index=False)

#%%
from data_download_functions import filterOnCities, eeToNumpy, \
create_featurecollection

data_dir = 'data_S21C'
csv_file = 'S21C_dataset.csv'
cities_csv = 'r_cities.csv'

if not os.path.isdir(intermediate_dir_training):
    os.makedirs(intermediate_dir_training) 
if not os.path.isdir(os.path.join(intermediate_dir_training, data_dir)):
    os.makedirs(os.path.join(intermediate_dir_training, data_dir)) 

# init gee session
#ee.Authenticate()
ee.Initialize()

cities = pd.read_csv(os.path.join(results_dir_cities, cities_csv))
cities = cities.iloc[862:1400]

# create features
cities_fc = create_featurecollection(cities, ('Lng_alt', 'Lat_alt'), 
                            properties = ['city_idx','city', 'population'], 
                            columns = ['city', 'population'])

# get image collection
S21C = ee.ImageCollection('COPERNICUS/S2').filterBounds(cities_fc) \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1) \
    .select(bands)
    
# filter image collection to get 2 random images per city
city_collection = cities_fc.map(filterOnCities(S21C, size, n=n_im)) #featurecollection of imagecollection per city
city_list = city_collection.toList(city_collection.size().getInfo()) #List of imagecollection per city
print('cityCollection size: ', city_collection.size().getInfo())

n_cities = len(city_list.getInfo())
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
        im_id = imx.get('city_idx').getInfo()+'_'+chr(ord('a')+i)
        system_idx = imx.get('system:index').getInfo()
        city = imx.get('city').getInfo()
        datestr = system_idx.split('_')[0]
        date = datestr.split('T')[0]
        time_im = datestr.split('T')[1]

        B = ee.ImageCollection([imx])
        B = ee.Image(B.median())
        
        # covert to numpy array
        try: 
            np_image = eeToNumpy(B, area, bands, ntry=5)
            # save
            np.save(os.path.join(intermediate_dir_training,data_dir, im_id+'.npy'), np_image)
        except:
            im_id = 'ERROR'
        
        # save some info in csv-file
        with open(os.path.join(intermediate_dir_training, csv_file), 'a') as file:
            writer = csv.writer(file, delimiter = ",")
            writer.writerow([im_id, system_idx, city, date, time_im])   
        

        end_im= time.time() - start_im
        tot_time = tot_time + end_im
        avg_time = tot_time / ((im*2)+i+1)

        print("\r image {}/{} for city {}/{}. Avg time image (sec): {}".format(i+1,n_im,im+1,n_cities, avg_time), end='')



#%%
""" Preprocess OSCD test dataset """

from data_download_functions import tif_to_numpy
from utils import list_directories

data_dir_OSCD = 'data_OSCD'
csv_file_OSCD = 'OSCD_dataset.csv'

if not os.path.isdir(intermediate_dir_test):
    os.makedirs(intermediate_dir_test)
if not os.path.isdir(os.path.join(intermediate_dir_test, data_dir_OSCD)):
    os.makedirs(os.path.join(intermediate_dir_test, data_dir_OSCD))

# get info about which images are used for training and which for test
train_cities = pd.read_csv(os.path.join(source_dir_test,'train.txt'),header =None).values
test_cities = pd.read_csv(os.path.join(source_dir_test,'test.txt'),header =None).values

# specify directories 
dirs = list_directories(source_dir_test)
image_dirs = ['imgs_1_rect','imgs_2_rect']
systemid_dirs = ['imgs_1', 'imgs_2']

# get the images and metadata
for i_city, city in enumerate(dirs):
    
    # check if training or test image
    if city in train_cities:
        traintest = 'train'
    elif city in test_cities:
        traintest = 'test'
        
    # get dates
    dates = (pd.read_csv(os.path.join(source_dir_test,city,'dates.txt'),header =None).values).squeeze()
    
    # read and save the images
    for i_dir, image_dir in enumerate(image_dirs):
        image_path = os.path.join(source_dir_test, city, image_dir)
        im_bands = os.listdir(image_path)
        n_bands = len(im_bands)
        
        # read images to numpy array
        im = tif_to_numpy(image_path, im_bands, n_bands)

        im_id = str(i_city)+'_'+chr(ord('a')+i_dir)
        
        # save
        np.save(os.path.join(intermediate_dir_test,data_dir_OSCD, im_id+'.npy'), im)

        # get systemid
        system_idx= os.listdir(os.path.join(source_dir_test, city, systemid_dirs[i_dir]))[0][:-8]
        
        # save some info in csv-file
        with open(os.path.join(intermediate_dir_test, csv_file_OSCD), 'a') as file:
            writer = csv.writer(file, delimiter = ",")
            writer.writerow([im_id, system_idx, city, dates[i_dir].split(' ')[1], traintest])  


        
        
#%%        
        
from plots import plot_random_imagepairs, plot_random_images

fig = plot_random_imagepairs(2, os.path.join(intermediate_dir_test, data_dir_OSCD), [4,3,2])
