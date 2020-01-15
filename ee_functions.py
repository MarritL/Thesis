#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:32:14 2020

@author: M. Leenstra
"""
import ee
import numpy as np


def clipImage(geom, label, size, resolution=10):
    """
    Clips an Image to specified size around a point

    Parameters
    ----------
    geom : ee.Geometry
        Central point of the output image
    label : string
        Name of central point (assumed to be a city)
    size : int
        size of output image in pixels
    resolution : int
        spatial resolution of input image. Default is 10 (Sentinel-2).

    Returns
    -------
    ee.Image
        Google earth engine image clipped around central point

    """
    def wrap(image):
        nonlocal geom
        nonlocal label
        nonlocal size
        
        #cast to ee objects
        geom = ee.Geometry(geom)
        image = ee.Image(image)
        
        # calculate aio
        aoi = geom.buffer((size/2*resolution)).bounds()
        
        #add label and clip to aoi
        image = image.set({'city': label}) \
                     .clip(aoi) #\
        #             .select('B.+')
    
        return image
    return wrap




def filterOnCities(image_col, size, n=2): 
    """
    Filters an ImageCollection based on a FeatureCollection of city-points
    Should be used as a mapping function with a map over the FeatureCollection 
    of ee.Geometry() Points.

    Parameters
    ----------
    image_col : ee.ImageCollection
        google earth engine image collection, should contain multiple images for
        each city
    size : int
        approximage output size of final image (in pixels)
    n : int, optional
        Number of images per city. The default is 2.

    Returns
    -------
    ee.FeatureCollection
        Each feature in the google earth engine feature collection represents 
        1 city and contains an ee.ImageCollection of n images

    """
    def wrap(city):
        nonlocal image_col
        nonlocal size
        nonlocal n
        
        # get city info
        geometry = city.geometry();
        label = city.getString('city')
        
        # filter imges
        images = image_col.filterBounds(geometry) \
                         .randomColumn('random', 77) \
                         .sort('random') \
                         .limit(n) \
                         .set({'city': label}) \
                         .set({'geometry': geometry}) \
                         .map(clipImage(geometry, label, size))                    
        return ee.ImageCollection(images) 
    return wrap

def eeToNumpy(img, area, bands=["B4","B3","B2"], resolution=10):
    """
    Converts ee.Image to numpy array for specified area and bands. Bands are 
    

    Parameters
    ----------
    img : ee.Image
        Google earth explorer image to be converted to numpy array
    area : ee.Geometry
        Polygon of area to convert to numpy array
    bands : list / array
        Bands to be converted to numpy array. Default: ["B4", "B3", "B2"] (RGB)
    resolution : int
        Spatial resolution of the output. Default 10 for Sentinel-2

    Returns
    -------
    np_img : Numpy array
        stacked numpy array with specified bands in 3th dimension

    """
    img = img.addBands(ee.Image.pixelLonLat())
 
    img = img.reduceRegion(reducer=ee.Reducer.toList(),\
                                        geometry=area,\
                                        maxPixels=1e13,\
                                        scale=resolution)
    
    
    # get the image dimensions
    lats = np.array((ee.Array(img.get("latitude")).getInfo()))
    lngs = np.array((ee.Array(img.get("longitude")).getInfo()))
    nrow = len(np.unique(lats))
    ncol = len(np.unique(lngs))

    # convert image band-by-band to numpy array
    for i, band in enumerate(bands):
        np_b = np.array((ee.Array(img.get(band)).getInfo())).reshape((nrow, ncol))
        # normalize
        np_b = np_b / np.max(np_b)
        if i == 0:
            np_img = np.zeros([np_b.shape[0], np_b.shape[1], len(bands)])
        np_img[:,:,i] = np_b

    return np_img



