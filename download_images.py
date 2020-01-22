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
data_out = '/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/dataout/'
#csv_file = '/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/S21C_dataset.csv'
#data_out = '/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/data/'
#csv_file = '/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/S21C_dataset.csv'

#%% 
"World-cities dataset from geonames"
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
import csv

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
cities = cities.iloc[332:1000]

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
i = 55
j = 1
testim = np.load(os.path.join(intermediate_dir_training,data_dir,str(i)+'_'+chr(ord('a')+j)+'.npy'))
np_image = testim

#testim = testim/np.max(testim)
for i in range(testim.shape[2]):
    testim[:,:,i] = testim[:,:,i]/np.max(testim[:,:,i])
plt.imshow(testim[:,:,[3,2,1]])
    
plt.imshow(np_image[:,:,0])
plt.imshow(np_image[:,:,1])
plt.imshow(np_image[:,:,2])
plt.imshow(np_image[:,:,3])
plt.imshow(np_image[:,:,4])
plt.imshow(np_image[:,:,5])
plt.imshow(np_image[:,:,6])
plt.imshow(np_image[:,:,7])
plt.imshow(np_image[:,:,8])
plt.imshow(np_image[:,:,9])
plt.imshow(np_image[:,:,10])
plt.imshow(np_image[:,:,11])
plt.imshow(np_image[:,:,12])



#%%
# =============================================================================
# def new_test(feature):
#     print(feature.getInfo())
#     
#     return feature
# 
# test_collection = cityCollection.map(new_test)
# =============================================================================
    

#%%
resolution = 10
#def new_test(area, bands, resolution=10):
def prepare(imx):
    # get some metadata
    date = ee.Number(imx.get('GENERATION_TIME'))
    city = imx.getString('city')
    geom = imx.geometry()
    
    # reform image
    B = ee.ImageCollection([imx]) 
    B = ee.Image(B.median())
    # add lat/lng and metadata
    B = B.addBands(ee.Image.pixelLonLat()) \
         .set({'city': city})\
         .set({'date': date})\
         .set({'geom': geom})

    return ee.Image(B)  


def createPair(feature):

    imagePair = ee.ImageCollection(feature)
    imagePair = imagePair.map(prepare)
    return(imagePair)

cityCollection = cityCollection.map(createPair)
cityList = cityCollection.toList(cityCollection.size().getInfo()) #List of imagecollection per city

    
imageList = []    
bands = ['B1','B2','B3','B4']
#n_bnds = len(bands)

for im in [0,2]: 
    start_city = time.time()
    imagePair = ee.ImageCollection(cityList.get(im))
    images = imagePair.toList(imagePair.size().getInfo())
    area = imagePair.geometry()  
   
    
    for i in range(n_im):
        start = time.time()
        B = ee.Image(images.get(i))

        
        np_image = eeToNumpy(B, area, bands)
        np.save("/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/data/"+str(im)+'_'+str(i)+'.npy', np_image)
        
        imageList.append(np_image)   
        print("1 image: ", time.time() - start)
    print("1 city: ", time.time() - start_city)



test_collection = imagePair.map(new_test)
    
test_collection = S21C.map(new_test(area, bands, 10))     
    
    
    

    
    
    
#first = cityCollection.first()
#print("first city: ", first.getString('city').getInfo())
#cityList = cityCollection.toList(cityCollection.size().getInfo()) #List of imagecollection per city
imagePair = ee.ImageCollection(cityList.get(im))
#print("city name: ", imagePair.getString('city').getInfo())
images = imagePair.toList(imagePair.size().getInfo())

im0 = ee.Image(images.get(0))
#print('Projection, crs, and crs_transform:', im0.select('B2').projection().getInfo())
#print('Scale in meters:', im0.projection().nominalScale().getInfo())

#bands = [band['id'] for band in imagePair.getInfo()['features'][0]['bands']]
#n_imgs = len(imagePair.getInfo()['features'])
n_bnds = len(bands)

imageList = []    
bands = ['B2','B3','B4']
n_bnds = len(bands)
    
for im in [2,3]:  
    imagePair = ee.ImageCollection(cityList.get(im))
    images = imagePair.toList(imagePair.size().getInfo())
    area = imagePair.geometry()  
    for i in range(n_im):
        imx = ee.Image(images.get(i))
        for idx, band in enumerate(bands):
            print(band)
            B = imx.select(band)
            B = ee.ImageCollection([B, B])
            B = ee.Image(B.median()).rename(['result'])
            lat, lon, data = LatLonImg(B, area)
            arr = toImage(lat,lon,data) 
            arr = arr/np.max(arr)
            if idx == 0:
                image = np.zeros([arr.shape[0], arr.shape[1], n_bnds])
            image[:,:,idx] = arr
            
        imageList.append(image)   
    
    len(imageList)
    imageList[0].shape
    plt.imshow(imageList[5][:,:,[2,1,0]])




# covert the lat, lon and array into an image
def toImage(lats,lngs,data):
 
    # get the unique coordinates
    unique_lats = np.unique(lats)
    unique_lngs = np.unique(lngs)
 
    # get number of columns and rows from coordinates
    ncols = len(unique_lngs)
    nrows = len(unique_lats)
    ndims = data.shape[0]
 
    # determine pixelsizes
    #ys = uniqueLats[1] - uniqueLats[0]
    #xs = uniqueLons[1] - uniqueLons[0]
 
    # create an array with dimensions of image
    arr = np.zeros([nrows, ncols, ndims], np.float32) #-9999
 
    # fill the array with values
    counter =0
    for d in range(ndims):
        for y in range(nrows):
            for x in range(ncols):
                if lats[counter] == unique_lats[y] and lngs[counter] == unique_lngs[x] and counter < len(lats)-1:
                    counter+=1
                    arr[len(unique_lats)-1-y,x,d] = data[d,counter] # we start from lower left corner
    return arr
    
    
    
    
    
    
    
    
def createImage(item):
    
    bands = ['B1','B2','B3','B4']

    for idx, band in enumerate(bands):
        print(idx)
        print(band)
        B = item.select(band)
        B = ee.ImageCollection([B, B])
        B = ee.Image(B.median()).rename(['result'])
        area = imagePair.geometry()
        lat, lon, data = LatLonImg(B, area)
        arr = toImage(lat,lon,data) 
        arr = arr/np.max(arr)
        if idx == 0:
            image = np.zeros([arr.shape[0], arr.shape[1], n_bnds])
        image[:,:,idx] = arr
    
    return image








im0 = im0.reproject('ESPG:4326')
im0rgb = im0.select('B4', 'B3', 'B2')
im1 = ee.Image(images.get(1))
#im0 = im0.addBands(ee.Image.pixelLonLat())
print('dimensions B2: ', im0.arrayDimensions().getInfo()['bands'][0]['dimensions'])
dimensions = im0.arrayDimensions().getInfo()['bands'][1]['dimensions']


ndvi = ee.Image(im0.normalizedDifference(['B8', 'B4'])).rename(["ndvi"])
test = ndvi.addBands(ee.Image.pixelLonLat())

im0_1 = im0.addBands(ee.Image.pixelLonLat())
#im0_1 = im0_1.addBands(ee.Image.pixelCoordinates({'espg':4326}))
im0_1 = im0_1.reduceRegion(reducer=ee.Reducer.toList(),\
                                        geometry=im0_1.geometry(),\
                                        maxPixels=1e13,\
                                        scale=10)
    
B2 = im0.select('B2').reduceRegion(ee.Reducer.toList(), im0.geometry(),scale=10)    
    
B2= np.array(ee.Array(B2.get("B2")).getInfo())
B3= np.array((ee.Array(im0_1.get("B3")).getInfo()))
B4= np.array((ee.Array(im0_1.get("B4")).getInfo()))
lats = np.array((ee.Array(im0_1.get("latitude")).getInfo()))
lons = np.array((ee.Array(im0_1.get("longitude")).getInfo()))

# get unique coordinates
unique_lats = np.unique(lats)
unique_lons = np.unique(lons)

# get number of columns and rows from coordinates
ncols = len(unique_lons)
nrows = len(unique_lats)

# determine pixelsizes
ys = unique_lats[1] - unique_lats[0]
xs = unique_lons[1] - unique_lons[0]

def Information(item):
    return item.toList()

def getImages(item):
    return createImage(item)
    

def computeMedian(item):
    return ee.ImageCollection(item).median()

def imagesPerCity(item):
    imagePair = ee.ImageCollection(item)
    imagePair = imagePair.map(getImages)
    #print("city name: ", imagePair.getString('city').getInfo())
    #images = imagePair.toList(imagePair.size().getInfo())
    #im0 = ee.Image(images.get(0))
    #print('Projection, crs, and crs_transform:', im0.select('B2').projection().getInfo())
    
    return(imagePair)
    

example = cityCollection.map(imagesPerCity)
print(example.getInfo())


tot_time = 0

for im in range(n_cities): 

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
        city = imx.get('city').getInfo()
        # add the system index (sentinel-2 identifier)
        system_idx = imx.get('system:index').getInfo()
        datestr = system_idx.split('_')[0]
        date = datestr.split('T')[0]
        time_im = datestr.split('T')[1]
        
        # simpify image metadata
        B = ee.ImageCollection([imx])
        B = ee.Image(B.median())
        
        # covert to numpy array
        try: 
            np_image = eeToNumpy(B, area, bands, ntry)
            # save
            np.save(os.path.join(intermediate_dir_training, data_dir, im_id+'.npy'), np_image)
        except Exception as e:
            print(e)
            im_id = 'ERROR'
        
        # save some info in csv-file
        with open(os.path.join(intermediate_dir_training, csv_file), 'a') as file:
            writer = csv.writer(file, delimiter = ",")
            writer.writerow([im_id, system_idx, city, date, time_im])   
        
        end_im= start_im - time.time()
        tot_time = tot_time + end_im
        avg_time = tot_time / ((im*2)+i+1)

        print("\r image {}/{} for city {}/{}. Avg time image (sec): {}".format(i+1,n_im,im+1,n_cities, avg_time))
