#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:51:21 2020

@author: M. Leenstra
"""

import numpy as np
import os
from random import sample
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patches as patches
     
def plot_random_imagepairs(n_imagepairs, image_folder, bands, axis=False, normalize=True, titles = None):
    """ 
    plot n random image pairs

    Parameters
    ----------
    n_imagepairs : int
        number of images to plot
    image_folder : string
        path to folder where images are located
    bands : list 
        numbers of bands to display. Must be either 3 or 1
    axis : boolean, optional
        whether or not axis should be plotted. The default is False.
    normalize : boolean, optional
        whether or not the bands should be normalized. The defailt is True.
    titles: list, optional
        list with a title (string) for each image. The default is None.

    Returns
    -------
    fig : matplotlib Figure
        figure with the images plotted in subsplots
    ax : list
        list of axes in the figure
        
    Example
    -------
    fig, ax = plot_random_imagepairs(2, os.path.join(intermediate_dir_training, data_dir), [8,3,2])

    """
    
    # list all images in the folder
    entries = os.listdir(image_folder)

    # take a random sample
    rand_im = sample(entries, n_imagepairs)
    rand_imidx = [s.split('_')[0] for s in rand_im]
    rand_impairs = [func(s) for s in rand_imidx for func in (lambda s: s+'_a.npy',lambda s: s+'_b.npy')]
    
    fig, ax = plot_image(image_folder, rand_impairs, bands, axis, normalize)
    
    return fig, ax

def plot_random_images(n_images, image_folder, bands, axis=False, normalize=True, titles = None):
    """ 
    plot n random images

    Parameters
    ----------
    n_images : int
        number of image to plot
    image_folder : string
        path to folder where images are located
    bands : list 
        numbers of bands to display. Must be either 3 or 1
    axis : boolean, optional
        whether or not axis should be plotted. The default is False.
    normalize : boolean, optional
        whether or not the bands should be normalized. The defailt is True.
    titles: list, optional
        list with a title for each image. The default is None.

    Returns
    -------
    fig : matplotlib Figure
        figure with the images plotted in subsplots
    ax : list
        list of axes in the figure
        
    Example
    -------
    fig, ax = plot_random_images(10, os.path.join(intermediate_dir_training, data_dir), [8,3,2])

    """
    
    # list all images in the folder
    entries = os.listdir(image_folder)

    # take a random sample
    rand_im = sample(entries, n_images)
    
    fig, ax = plot_image(image_folder, rand_im, bands, axis, normalize)
    
    return fig, ax
    
def plot_image(image_folder, image_list, bands, axis = False, normalize=True, titles = None):
    """ 
    plot the defined images

    Parameters
    ----------
    image_folder : string
        path to folder where images are located
    image_list : list
        list of image filenames (strings)
    bands : list 
        numbers of bands to display. Must be either 3 or 1
    axis : boolean, optional
        whether or not axis should be plotted. The default is False.
    normalize : boolean, optional
        whether or not the bands should be normalized. The defailt is True.
    titles: list, optional
        list with a title for each image. The default is None.

    Returns
    -------
    fig : matplotlib Figure
        figure with the images plotted in subsplots
    ax : list
        list of axes in the figure
        
    Example
    --------
    fig, ax = plot_image(os.path.join(intermediate_dir_training, data_dir), ['240_a.npy','240_b.npy'], [8,3,2,])
        
    """
    
    n_images = len(image_list)
    assert n_images > 0, "empty image list"
    n_bands = len(bands)
    assert n_bands == 1 or n_bands == 3, "only images with 3 or 1 band allowed"
    assert titles == None or len(titles) >= n_images, "provide a title for each image"
    rows = int(np.ceil(n_images / 2))
    cols = 1 if n_images < 2 else 2
    
    # prepare
    fig, ax = plt.subplots(rows,cols, figsize = (10,10))
    
    for i in range(n_images):
        im = np.load((os.path.join(image_folder,image_list[i])))[:,:,bands]
        
        # normalize
        if normalize:
            im = normalize2plot(im)

        # remove single-dimensional entries
        if n_bands == 1:
            im = np.squeeze(im)

        # plot image  
        r = int(np.floor(i/2))
        c = i%2
        if rows == 1:
            if cols == 1:
                ax.imshow(im)
                if titles != None:
                    ax.set_title(titles[i])
                if not axis:
                    ax.axis('off')
            else:
                ax[c].imshow(im) 
                if titles != None:
                    ax[c].set_title(titles[i])
                if not axis:
                    ax[c].axis('off')
        else:
            ax[r,c].imshow(im)
            if titles != None:
                ax[r,c].set_title(titles[i])
            if not axis:
                ax[r, c].axis('off')
            
    return fig, ax

def plot_detectedlines(image, h, theta, d):
    """
    plots lines detected in images

    Parameters
    ----------
    image : numpy.ndarray. shape: (N,M) or (N,M,3)
        numpy array of image. Either greyscale 2D-array or 3-band image.
    h : numpy.ndarray. shape: (N,M)
        Hough space returned by the hough_line function.
    theta : numpy.ndarray. shpae: (N,)
        Angles returned by the hough_line function. Assumed to be continuous. (angles[-1] - angles[0] == PI).
    d : numpy.ndarray. shape: (N,)
        Distances returned by the hough_line function.

    Returns
    -------
    fig : matplotlib Figure
        figure with the detected lines plotted over the image
    ax : list
        list of axes in the figure

    """
    from skimage.transform import hough_line_peaks
    
    assert len(image.shape) == 2 or image.shape[2] == 3, "only greyscale or 3-band images allowed"
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    ax.imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax.plot(origin, (y0, y1), '-r')
    ax.set_xlim(origin)
    ax.set_ylim((image.shape[0], 0))
    ax.set_axis_off()
    ax.set_title('Detected lines')
    
    return fig, ax

def normalize2plot(image, percentile = 98):
    """
    normalizes images for plotting purpose

    Parameters
    ----------
    image : numpy.ndarray. shape: (N,M) or (N,M,D)
        numpy array of image. Either greyscale 2D-array or 3D array image.
    percentile : int, optional
        percentile to use for normalization (exclude outliers). The default is 98.

    Returns
    -------
    image : numpy.ndarray. shape: (N,M) or (N,M,D)
        normalized numpy array of image. 

    """
    
    assert len(image.shape) == 2 or len(image.shape) == 3, 'image should be 2D or 3D'
    
    if len(image.shape) == 3:
        for i in range(image.shape[2]):
            image[:,:,i] = np.divide(image[:,:,i]-image[:,:,i].min(),np.percentile(image[:,:,i],percentile)-image[:,:,i].min())
    else:
        image = np.divide(image-image.min(),np.percentile(image,percentile)-image.min())
        
    return image

def plot_sampledpatches(image_folder, image_list, patch1_start, patch2_start, patch_size, bands, plot_patches=False, axis=False, normalize = True, titles = None):
    """
    

    Parameters
    ----------
    image_folder : string
        path to folder where images are located
    image_list : list
        list of image filenames (strings)
    patch1_start : numpy.ndarray. Shape: (2,)
        Upper-left corner of patch 1, defined by row, column
    patch2_start : numpy.ndarray. Shape: (2,)
        Upper-left corner of patch 1, defined by row, column
    patch_size : int
        size of the extracted patches
    bands : list 
        numbers of bands to display. Length must be either 3 or 1
    plot_patches : boolean, optional
        whether to plot also the extracted patches in a subplot. The default is False.
    axis : boolean, optional
        whether or not axis should be plotted. The default is False.
    normalize : boolean, optional
        whether or not the bands should be normalized. The defailt is True.
    titles: list, optional
        list with a title for each image. The default is None.

    Returns
    -------
    fig : matplotlib Figure
        figure with the patches plotted over the images. Optional also the patches self are plotted.
    ax : list
        list of axes in the figure
        
    Example:
    fig, ax = plot_sampledpatches(image_folder, image_list, patch1_start, patch2_start, 96, [3,2,1],plot_patches=True, axis = True, titles = ['image 1', 'image 2', 'patch 1', 'patch 2'])

    """
    
    
    #prepare
    n_bands = len(bands)
    assert n_bands == 1 or n_bands == 3, "only images with 3 or 1 band allowed"
    
    # plot images
    fig, ax = plot_image(image_folder, image_list, bands, axis = axis, normalize=normalize, titles = titles)

    # create squars for the patches. Note: first columns than rows
    patch1a = patches.Rectangle((patch1_start[::-1]),patch_size,patch_size,linewidth=2,edgecolor='r',facecolor='none')
    patch2a = patches.Rectangle((patch2_start[::-1]),patch_size,patch_size,linewidth=2,linestyle =(0, (3, 3)), edgecolor='r',facecolor='none')
    patch1b = patches.Rectangle((patch1_start[::-1]),patch_size,patch_size,linewidth=2,linestyle =(0, (3, 3)),edgecolor='r',facecolor='none')
    patch2b = patches.Rectangle((patch2_start[::-1]),patch_size,patch_size,linewidth=2,edgecolor='r',facecolor='none')
        
    # Add the patches to the images
    ax[0].add_patch(patch1a)
    ax[0].add_patch(patch2a)
    ax[0].text(patch1_start[1]+patch_size/2-24, patch1_start[0]+patch_size/2+10, 'p1', fontsize=16, color='r')
    ax[1].add_patch(patch1b)
    ax[1].add_patch(patch2b)
    ax[1].text(patch2_start[1]+patch_size/2-24, patch2_start[0]+patch_size/2+10, 'p2', fontsize=16, color='r')
    
    # plot also the patches
    if plot_patches:
        n_rows = 2
        n_cols = 2
        patch_starts = [patch1_start, patch2_start]
        for i in range(len(fig.axes)):
            fig.axes[i].change_geometry(n_rows, n_cols, i+1)
    
        for i in range(len(image_list)):
            ax = np.append(ax, np.array([fig.add_subplot(n_rows, n_cols, i+3)]))

            im = np.load((os.path.join(image_folder,image_list[i])))[:,:,bands]
            
            # normalize
            if normalize:
                im = normalize2plot(im)
    
            # remove single-dimensional entries
            if n_bands == 1:
                im = np.squeeze(im)

            # plot patch. Note: subsetting in numpy again first rows than columns  
            ax[len(ax)-1].imshow(im[patch_starts[i][0]:patch_starts[i][0]+patch_size,patch_starts[i][1]:patch_starts[i][1]+patch_size, :])
            
        if not axis:
            for i in range(n_rows+n_cols):
                ax[i].axis('off')
        
        if titles != None:
            assert len(titles) == len(ax), "Provide a title for each subplot"
            for i in range(n_rows+n_cols):
                ax[i].set_title(titles[i])
                            
    return fig, ax
