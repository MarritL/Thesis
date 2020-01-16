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
bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
data_dir = '/home/cordolo/Documents/Studie Marrit/2019-2020/Thesis/Data'
data_out = '/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/data/'
csv_file = '/media/cordolo/5EBB-8C9A/Studie Marrit/2019-2020/Thesis/S21C_dataset.csv'

#%% 
"Try other dataset"
import datapackage
import geocoder

# variables specific for this part
minPop = 200000 #min population per city
data_url = 'https://datahub.io/core/world-cities/datapackage.json'

# to load Data Package into storage
package = datapackage.Package(data_url)

# to load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
   
# import data world cities + coordinates 
# worldcities.csv downloaded at 06/01/2020 from https://simplemaps.com/data/world-cities
cities = pd.read_csv(os.path.join(data_dir,'worldcities.csv'), header=0)

# data cleaning
cities_cleaned = cities.drop_duplicates(subset = ['lat', 'lng'])
cities_cleaned = cities_cleaned.dropna(subset = ['lat', 'lng', 'population'])
cities_cleaned = cities_cleaned[cities_cleaned['population'] >= minPop]

cities_merge = cities_cleaned.merge(data, left_on = ['city','country'], right_on = ['name','country'])
cities_merge = cities_merge.drop('name', axis=1)

# add alternative lat, lng coordinates
cities_merge['Lng_alt'] = np.nan
cities_merge['Lat_alt'] = np.nan


for idx, row in cities_merge[1004:].iterrows():
    #time.sleep(1)
    print('\r {}'.format(idx), end='')
    place = geocoder.geonames(row.geonameid, method='details', key='marrit_l')
    if place == None:
        continue
    if place.feature_class != 'P':
        continue
    
    cities_merge.loc[idx,'Lat_alt'] = float(place.lat)
    cities_merge.loc[idx,'Lng_alt'] = float(place.lng)

# save
cities_merge.to_csv(os.path.join(data_dir,'cities_merge.csv'), index=False)


# delete cities that are not found in the OSM database
cities_cleaned = cities_cleaned.dropna(subset = ['Lng_alt', 'Lat_alt'])

# save
cities_cleaned.to_csv(os.path.join(data_dir,'cities_cleaned.csv'), index=False)






#%% 
"Preprocessing simplemaps world cities dataset"

# imports
from data_download_functions import placenameToCoordinates

# variables specific for this part
minPop = 200000 #min population per city

# import data world cities + coordinates 
# worldcities.csv downloaded at 06/01/2020 from https://simplemaps.com/data/world-cities
cities = pd.read_csv(os.path.join(data_dir,'worldcities.csv'), header=0)

# data cleaning
cities_cleaned = cities.drop_duplicates(subset = ['lat', 'lng'])
cities_cleaned = cities_cleaned.dropna(subset = ['lat', 'lng', 'population'])
cities_cleaned = cities_cleaned[cities_cleaned['population'] >= minPop]

cities_merge = cities_cleaned.merge(data, left_on = ['city','country'], right_on = ['name','country'])
cities_merge = cities_merge.drop('name', axis=1)
#cities_join = cities_cleaned.set_index('city').join(data.set_index('name'), rsuffix='_2')

#%% 
"Preprocessing world cities dataset"

# imports
from data_download_functions import placenameToCoordinates

# variables specific for this part
minPop = 200000 #min population per city

# import data world cities + coordinates 
# worldcities.csv downloaded at 06/01/2020 from https://simplemaps.com/data/world-cities
# worldcitiespop.csv downloaded at 13/01/2020 from https://www.kaggle.com/max-mind/world-cities-database
cities = pd.read_csv(os.path.join(data_dir,'maxmind_world-cities-database/worldcitiespop.csv'), header=0)
cities = pd.read_csv(os.path.join(data_dir,'worldcities.csv'), header=0)

# data cleaning
cities_cleaned = cities.drop_duplicates(subset = ['Longitude', 'Latitude'])
cities_cleaned = cities_cleaned.dropna(subset = ['Longitude', 'Latitude', 'Population'])
cities_cleaned = cities_cleaned[cities_cleaned['Population'] >= minPop]
# 1 city contained a typo in the name
cities_cleaned.loc[cities_cleaned['City'] == 'ulubaria','City'] = 'uluberia'
cities_cleaned.loc[cities_cleaned['Accentcity'] == 'ulubaria','AccentCity'] = 'Uluberia'

# add alternative lat, lng coordinates
cities_cleaned['Lng_alt'] = np.nan
cities_cleaned['Lat_alt'] = np.nan
for idx, row in cities_cleaned.iterrows():
    time.sleep(1)
    print('\r {}'.format(idx), end='')
    place = placenameToCoordinates(row.City)
    if place == None:
        continue
    cities_cleaned.loc[idx,'Lat_alt'] = place.latitude
    cities_cleaned.loc[idx,'Lng_alt'] = place.longitude

# delete cities that are not found in the OSM database
cities_cleaned = cities_cleaned.dropna(subset = ['Lng_alt', 'Lat_alt'])

# save
cities_cleaned.to_csv(os.path.join(data_dir,'cities_cleaned.csv'), index=False)


#%%
from data_download_functions import filterOnCities, eeToNumpy
import csv

# init gee session
#ee.Authenticate()
ee.Initialize()

#cities = pd.read_csv(os.path.join(data_dir, 'cities_cleaned.csv'))
#cities = cities[cities['Population'] >= minPop]
#cities_full = cities
#cities = cities_merge
#cities = cities.iloc[:10]

# create features
features = []
for i in range(len(cities)):
    #p = ee.Geometry.Point((cities.iloc[i].Longitude, cities.iloc[i].Latitude))
    p = ee.Geometry.Point((cities.iloc[i].Lng_alt, cities.iloc[i].Lat_alt))
    f = ee.Feature(p) \
          .set({'city': cities.iloc[i].city, 'population':cities.iloc[i].population})
    features.append(f)

# create feature collection
cities_fc = ee.FeatureCollection(features)


# get image collection
S21C = ee.ImageCollection('COPERNICUS/S2').filterBounds(cities_fc) \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1) \
    .select(bands)
    

# filter image collection to get 2 random images per city
cityCollection = cities_fc.map(filterOnCities(S21C, size, n=n_im)) #featurecollection of imagecollection per city
cityList = cityCollection.toList(cityCollection.size().getInfo()) #List of imagecollection per city

print('cityCollection size: ', cityCollection.size().getInfo())

#imageList = []
#bands = ["B1", "B2", "B3", "B4"]
for im in range(len(cityList.getInfo())): 
    
    start_city = time.time()
    imagePair = ee.ImageCollection(cityList.get(im))
    images = imagePair.toList(imagePair.size().getInfo())
    area = imagePair.geometry()    
    
    for i in range(n_im):
        start = time.time()
        imx = ee.Image(images.get(i))
        
        # get some metadata
        im_id = str(im)+'_'+chr(ord('a')+i)
        system_idx = imx.get('system:index').getInfo()
        city = imx.get('city').getInfo()
        datestr = system_idx.split('_')[0]
        date = datestr.split('T')[0]
        time_im = datestr.split('T')[1]

        B = ee.ImageCollection([imx])
        B = ee.Image(B.median())
        
        # covert to numpy array
        try: 
            np_image = eeToNumpy(B, area, bands)
            # save
            np.save(data_out+im_id+'.npy', np_image)
        except:
            im_id = 'ERROR'
        
        # save some info in csv-file
        with open(csv_file, 'a') as file:
            writer = csv.writer(file, delimiter = ",")
            writer.writerow([im_id, system_idx, city, date, time_im])   
        
        #imageList.append(np_image)
        print("1 image: ", time.time() - start)
    print("1 city: ", time.time() - start_city)

i = 1
testim = np.load(data_out+'9_'+chr(ord('a')+i)+'.npy')
for i in range(testim.shape[2]):
    testim[:,:,i] = testim[:,:,i]/np.max(testim[:,:,i])
plt.imshow(testim[:,:,[3,2,1]])
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



