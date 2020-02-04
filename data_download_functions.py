#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:32:14 2020

@author: M. Leenstra
"""
import ee
import os
import csv
import numpy as np
from plots import plot_detectedlines

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from osgeo import gdal



def placenameToCoordinates(placename):
    """
    Get the location information from OpenStreetMap (OSM)

    Parameters
    ----------
    placename : string
        city to get location information 

    Returns
    -------
    geopy.location.Location
        locaton object with adress, altitude, longitude and latitude

    """
    
    # inner function to catch exceptions
    def getLocation(placename):
        from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
    
        try:
            return nom.geocode(placename)
        except GeocoderTimedOut:
            return getLocation(placename)
        except GeocoderUnavailable:
            return getLocation(placename)
    
    from geopy.geocoders import Nominatim
    nom = Nominatim(user_agent="data_download")
    
    return getLocation(placename)


# =============================================================================
# def clipImage(geom, label, index, size, resolution=10):
#     """
#     Clips an Image to specified size around a point
# 
#     Parameters
#     ----------
#     geom : ee.Geometry
#         Central point of the output image
#     label : string
#         Name of central point (assumed to be a city)
#     index : int
#         identifier of city
#     size : int
#         size of output image in pixels
#     resolution : int
#         spatial resolution of input image. Default is 10 (Sentinel-2).
# 
#     Returns
#     -------
#     ee.Image
#         Google earth engine image clipped around central point
# 
#     """
#     def wrap(image):
#         nonlocal geom
#         nonlocal label
#         nonlocal index
#         nonlocal size
#         
#         #cast to ee objects
#         geom = ee.Geometry(geom)
#         image = ee.Image(image)
#         
#         # calculate aio
#         aoi = geom.buffer((size/2*resolution)).bounds()
#         
#         #add label and clip to aoi
#         image = image.set({'city': label}) \
#                      .set({'city_idx': index})\
#                      .clip(aoi) #\
#         #             .select('B.+')
#     
#         return image
#     return wrap
# =============================================================================

# =============================================================================
# def filterOnCities(image_col, size, n=2): 
#     """
#     Filters an ImageCollection based on a FeatureCollection of city-points
#     Should be used as a mapping function with a map over the FeatureCollection 
#     of ee.Geometry() Points.
# 
#     Parameters
#     ----------
#     image_col : ee.ImageCollection
#         google earth engine image collection, should contain multiple images for
#         each city
#     size : int
#         approximage output size of final image (in pixels)
#     n : int, optional
#         Number of images per city. The default is 2.
# 
#     Returns
#     -------
#     ee.FeatureCollection
#         Each feature in the google earth engine feature collection represents 
#         1 city and contains an ee.ImageCollection of n images
# 
#     """
#     def wrap(city):
#         nonlocal image_col
#         nonlocal size
#         nonlocal n
#         
#         # get city info
#         geometry = city.geometry();
#         label = city.getString('city')
#         idx = city.get('im_idx')
#         
#         # filter imges
#         images = image_col.filterBounds(geometry) \
#                          .randomColumn('random', 77) \
#                          .sort('random') \
#                          .limit(n) \
#                          .set({'city': label}) \
#                          .set({'geometry': geometry}) \
#                          .set({'city_idx': idx}) \
#                          .map(clipImage(geometry, label, idx, size))  
#          
#         #checksize = images.size().getInfo()    
#         #assert checksize == n
#         #assert images.size().getInfo() >2, "collection is 2"
#         return ee.ImageCollection(images) 
#     return wrap
# =============================================================================

def clipImage(geom, properties, size, resolution=10):
    """
    Clips an Image to specified size around a point

    Parameters
    ----------
    geom : ee.Geometry
        Central point of the output image
    label : string
        Name of central point (assumed to be a city)
    index : int
        identifier of city
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
        nonlocal properties
        nonlocal size
        
        # cast to ee objects
        geom = ee.Geometry(geom)
        image = ee.Image(image)
        
        # calculate aio
        aoi = geom.buffer((size/2*resolution)).bounds()
        
        #clip to aoi
        image = image.clip(aoi) 
        
        # add metadata
        for prop in properties:
            image = image.set({prop: properties[prop]})
    
        return image
    return wrap


def filterOnCities(image_col, property_keys, size, n=2): 
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
        nonlocal property_keys
        nonlocal size
        nonlocal n
        
        # get geometry
        geometry = city.geometry();
        
        # filter imges
        images = image_col.filterBounds(geometry) \
                         .randomColumn('random', 77) \
                         .sort('random') \
                         .limit(n) \
                         .set({'geometry': geometry}) 
         
        # get metadata
        properties = dict()
        for prop in property_keys:
            properties[prop] = city.get(prop)
        
        # clip images               
        images = images.map(clipImage(geometry, properties, size))  
        
        return ee.ImageCollection(images) 
    return wrap

def eeToNumpy(img, area, bands=["B4","B3","B2"], resolution=10, ntry=5):
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

        # call .get(band) until the right number of pixels is returned
        # (error in gee: sometimes the image is returned only partly)
        np_b = np.array(range(5), dtype=np.float)
        n_try = 0
        while np_b.shape[0] != nrow*ncol:
            np_b = np.array((ee.Array(img.get(band)).getInfo()))
            n_try += 1
            if n_try > ntry:
                raise Exception('band {} does not have the correct shape'.format(band))
            
        np_b = np_b.reshape(nrow, ncol)
        
        if i == 0:
            np_img = np.zeros([np_b.shape[0], np_b.shape[1], len(bands)])
        np_img[:,:,i] = np_b

    return np_img


def create_featurecollection(df, coords, properties):
    """
    create a ee.FeatureCollection of point features out of a dataframe. Note
    google earth engine should be initialized with account.

    Parameters
    ----------
    df : pd.dataframe
        Dataframe with information about the point features. Should contain at 
        least the column names specified in coords and columns.
    coords : dict
        dictionary with two keys, 'lng' and 'lat' specifying the column names 
        for the longitude and latitude in the dataframe.
    properties : dict
        dict of properties for the features. The values in the dict should
        correspond to columnnames in the df.

    Returns
    -------
    ee.FeatureCollection
        Feature collection with point features and properties as specified.

    """
    
    features = []
    for idx, row in df.iterrows():
        # create geometry
        p = ee.Geometry.Point((df.loc[idx,coords['lng']], df.loc[idx,coords['lat']]))
        # create feature with metadata
        f = ee.Feature(p) 
            
        for prop in properties:
            if properties[prop] == None:
                f = f.set({prop: str(idx)})
            elif type(df.loc[idx,properties[prop]]) == np.int64:
                f = f.set({prop: df.loc[idx,properties[prop]].astype('object')})
            else: 
                f = f.set({prop: df.loc[idx,properties[prop]]})
        #.set({properties[0]: str(idx), properties[1]:df.loc[idx,columns[0]], properties[2]:df.loc[idx,columns[1]]})

        features.append(f)

    # create feature collection
    return ee.FeatureCollection(features)

def filter_errors_folder(image_folder, threshold=0.5):
    """
    This function filters errenous images out of a folder containing images. 
    It is based on the notion that erroneous images show slightly of horizontal 
    bands over the image. Hough line detection is used to detect these lines. 
    The erroneous images are separated from the correct images by thresholding 
    the standard deviation over the line anlges.     

    Parameters
    ----------
    image_folder : string
        path to folder where images are located
    threshold : float, optional
        threshold on standard deviation of angles, under this threshold the 
        images are considered erroneous. The default is 0.5.

    Returns
    -------
    errors : list
        list with the erroneous image filenames (strings)
    """

    # list all images in the folder
    entries = os.listdir(image_folder)
    
    errors = filter_errors(image_folder, entries, threshold)
    
    return errors


def filter_errors(image_folder, image_list, threshold=0.5, plot = False):
    """
    This function filters errenous images out of an the image_list. It is based
    on the notion that erroneous images show slightly of horizontal bands over
    the image. Hough line detection is used to detect these lines. The erroneous
    images are separated from the correct images by thresholding the standard
    deviation over the line anlges. 

    Parameters
    ----------
    image_folder : string
        path to folder where images are located
    image_list : list
        list of image filenames (strings)
    threshold : float
        threshold on standard deviation of angles, under this threshold the 
        images are considered erroneous. The default is 0.5.
    plot : boolean, optional
        if true the detected lines are plotted. Only for 1 image.

    Returns
    -------
    errors : list
        list with the erroneous image filenames (strings)
    fig : matplotlib Figure, only if plot = True
        figure with the lines detected in the image
        
    Tested
    ------
    bad_images = ['40_a.npy', '163_a.npy', '163_b.npy', '269_a.npy', '313_b.npy', '122_b.npy', '277_a.npy', '328_b.npy', '311_a.npy', '184_b.npy', '398_b.npy', '158_a.npy', '356_b.npy']
    good_images = ['240_a.npy', '451_b.npy', '558_b.npy','617_a.npy', '61_b.npy','267_b.npy','490_b.npy','243_b.npy', '685_b.npy','696_a.npy']
    good_images = ['318_a.npy', '527_b.npy', '100_a.npy', '100_b.npy', '101_a.npy', '101_b.npy', '102_a.npy', '102_b.npy', '103_a.npy', '103_b.npy', '104_a.npy', '104_b.npy', '105_a.npy','105_b.npy', '107_a.npy']

    """    
    
    errors = []
    for image in image_list:
        im = np.load((os.path.join(image_folder,image)))[:,:,0]
        
        # edge filter using canny algorithm
        edges = canny(im, sigma=6)
        # detect lines using hough transform
        test_angles = np.linspace(np.pi/4, 7*np.pi/4 / 2, 360/4)
        h, theta, d = hough_line(edges, theta=test_angles)
        accum, angle, dist = hough_line_peaks(h, theta, d)

        if plot:
            assert len(image_list) == 1, "plot can only be used for 1 image"
            fig = plot_detectedlines(im, h, theta, d)

        # filter based on standard deviation of angles
        std = np.std(angle)
        if std < threshold:
            errors.extend([image])
    
    if plot:
        return errors, fig 
    else:
        return errors
    
def tif_to_numpy(tif_folder, band_list, n_bands, n_rows=None, n_cols=None):
    """
    read tif images to numpy array. Assumed that all bands are stored in 
    separate folder, described by band_list.
    
    Parameters
    ----------
    tif_folder : string
        path to folder where tif-bands are located
    band_list : list
        list of band filenames (strings)
    n_bands : int
        number of bands in the final array
    n_rows : int, optional
        optional specify the number of rows, if image is different size the 
        image will be resampled (using bilinear interpolation). Default is None,
        in which case the number of rows of the first band will be used.
    n_cols : int, optional
        optional specify the number of columns, if image is different size the 
        image will be resampled (using bilinear interpolation). Default is None,
        in which case the number of columns of the first band will be used.

    Returns
    -------
    image : numpy.ndarray
        tif image casted to numpy array

    """   
    
    for i, im_band in enumerate(band_list):
        ds = gdal.Open(os.path.join(tif_folder, im_band), gdal.GA_ReadOnly)
        
        # initialize numpy array
        if i == 0:
            imageidx = 0
            if n_cols == None:
                n_cols = ds.RasterXSize
            if n_rows == None:
                n_rows = ds.RasterYSize 
            image = np.zeros((n_rows, n_cols, n_bands))  
        
        # check if object has correct shape
        if not ds.RasterXSize == n_cols or not ds.RasterYSize == n_rows:
            print("resample... {}".format(im_band))
            ds = gdal.Warp("", ds, format='mem', width=n_cols, height=n_rows, resampleAlg=1)
        
        # read values in tiff image
        for b in range(ds.RasterCount):
            band = b+1
            srcband = ds.GetRasterBand(band)
            if srcband is None:
                continue
        
            # save values in numpy array
            image[:,:,imageidx] = srcband.ReadAsArray()
            imageidx += 1
        
        # reset
        ds = None
        
    return image
        



