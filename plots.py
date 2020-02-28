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
     
def plot_random_imagepairs(n_imagepairs, image_folder, bands, axis=False, normalize=True, titles = None, rows = None, cols = None):
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
    rows : int, optional
        number of rows in the figure
    cols: int, optional
        number of cols in the figure

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
    
    fig, ax = plot_image(image_folder=image_folder, image_list= rand_impairs,bands=bands, axis=axis, \
                         normalize=normalize, titles=titles, rows=rows, cols=cols)
                         
    return fig, ax

def plot_random_images(n_images, image_folder, bands, axis=False, normalize=True, titles = None, rows = None, cols = None):
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
    rows : int, optional
        number of rows in the figure
    cols: int, optional
        number of cols in the figure

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
    
    fig, ax = plot_image(image_folder=image_folder, image_list=rand_im, bands = bands, \
                         axis=axis, normalize=normalize, titles=titles, rows=rows,cols=cols)
    
    
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

def normalize2plot(image, percentile = 99):
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
            image[:,:,i] = np.divide(image[:,:,i]-image[:,:,i].min(),
                np.percentile(image[:,:,i],percentile)-image[:,:,i].min()+1E-8)
    else:
        image = np.divide(image-image.min(),
                          np.percentile(image,percentile)-image.min()+1E-8)
    
    image[image>1]=1
    image[image<0]=0
        
    return image



def plot_image(image_folder, image_list, bands, axis = False, normalize=True, titles = None, rows = None, cols = None, constrainted_layout=True):
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
    rows : int, optional
        number of rows in the figure
    cols: int, optional
        number of cols in the figure

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
    if rows == None or rows <= 0:
        rows = int(np.ceil(n_images / 2))
    if cols == None or cols <= 0:
        cols = 1 if n_images < 2 else 2
    
    # prepare
    fig = plt.figure(constrained_layout=constrainted_layout, figsize=(5,5))
    gs = fig.add_gridspec(rows, cols)
    
    #fig, ax = plt.subplots(rows,cols, figsize = (10,10))
    
    ax = []
    for i in range(n_images):
        im = np.load((os.path.join(image_folder,image_list[i])))[:,:,bands]
        
        # normalize
        if normalize:
            im = normalize2plot(im)

        # remove single-dimensional entries
        if n_bands == 1:
            im = np.squeeze(im)

        # plot image  
        r = int(np.floor(i/cols))
        c = i%cols
        ax.append(fig.add_subplot(gs[r, c]))
        
        ax[i].imshow(im)
        if titles != None:
            ax[i].set_title(titles[i])
        if not axis:
            ax[i].axis('off')
            
    return fig, ax

def plot_sampledpatches(image_folder, image_list, patch1_start, patch2_start, patch_size, bands, plot_patches=False, axis=False, normalize = True, titles = None, constrainted_layout=False):
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
    fig, ax = plot_image_old(image_folder, image_list, bands, axis = axis, normalize=normalize, titles = titles)

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

def plot_sampled_triplets(image_folder, image_list, patch1_start, patch2_start, patch3_start, patch_size, bands, plot_patches=False, axis=False, normalize = True, titles = None,constrained_layout=True):
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
    
    unique_images = np.unique(image_list)
    n_images = len(unique_images)
    assert n_images == 1 or n_images ==2, "only single images or image pairs allowed"

    if plot_patches:
        nrows = 2
        ncols = 6
    else:
        nrows = 1
        ncols = 6
    
    fig = plt.figure(constrained_layout=constrained_layout)
    gs = fig.add_gridspec(nrows, ncols)
    
    # create squars for the patches. Note: first columns than rows
    patch_locs = []
    patch_starts = [patch1_start, patch2_start, patch3_start]
    for start in patch_starts:
        patch_locs.append(patches.Rectangle((start[::-1]),patch_size,patch_size,linewidth=2,edgecolor='r',facecolor='none'))
        patch_locs.append(patches.Rectangle((start[::-1]),patch_size,patch_size,linewidth=1,linestyle =(0, (3, 3)),edgecolor='r',facecolor='none'))       
        
    ax = []
    images = []
    for i in range(n_images):
        im = np.load((os.path.join(image_folder,unique_images[i])))[:,:,bands]
        
        # normalize
        if normalize:
            im = normalize2plot(im)

        # remove single-dimensional entries
        if n_bands == 1:
            im = np.squeeze(im)

        # plot imag
        if n_images == 1:
            ax.append(fig.add_subplot(gs[0, :]))
        else:
            if i == 0:
                ax.append(fig.add_subplot(gs[0, 0:3]))
            else:
                ax.append(fig.add_subplot(gs[0, 3:]))
                
        ax[i].imshow(im)
        if titles != None:
            ax[i].set_title(titles[i])
        if not axis:
            ax[i].axis('off')
    
        if plot_patches:
            images.append(im)
    
    if plot_patches:
        ax.append(fig.add_subplot(gs[1,0:2]))
        ax.append(fig.add_subplot(gs[1,2:4]))
        ax.append(fig.add_subplot(gs[1,4:]))

    for i,file in enumerate(image_list):
        if n_images == 1:
            ax_pos = 0 
            ax[ax_pos].add_patch(patch_locs[i*2])
        else:
            ax_pos = ord(file.split('.')[0].split('_')[1])-97
            ax[ax_pos].add_patch(patch_locs[(i*2)])
            not_ax = 1 if ax_pos == 0 else 0
            ax[not_ax].add_patch(patch_locs[i+((i+1)*1)])
        ax[ax_pos].text(patch_locs[(i*2)].xy[0], patch_locs[(i*2)].xy[1]+patch_size/2+20, 'p'+str(i+1), fontsize=10, color='r')
        
        # plot patch. Note: subsetting in numpy, thus again first rows than columns      
        if plot_patches:
                ax[i+n_images].imshow(images[ax_pos][patch_starts[i][0]:patch_starts[i][0]+patch_size,patch_starts[i][1]:patch_starts[i][1]+patch_size, :])

    if not axis:
        for i in range(len(ax)):
            ax[i].axis('off')
        
    if titles != None:
        assert len(titles) == len(ax), "Provide a title for each subplot"
        for i in range(len(ax)):
            ax[i].set_title(titles[i])
 
def plot_imagepair_plus_gt(image1, image2, gt, bands = [3,2,1], axis=False):
    """
    plot two images plus the ground truth

    Parameters
    ----------
    image1 : np.ndarray of shape N,M,D
        unnormalized numpy array of image1, D must be >= 3
    image2 : np.nadarray of shape N,M,D
        unnormalzied numpy array of iamge2, D must be >= 3
    gt : np.ndarray of shape N,M
        ground truth of the image pair
    bands : lsit, optional
        which bands should be plotted. The default is [3,2,1].
    axis : boolean, optional
        whether or not to plot the axis. The default is False.

    Returns
    -------
    fig : matplotlib Figure
        figure
    axes : list
        list of axes in the figure

    """
    
    fig, axes = plt.subplots(ncols=3, figsize=(10, 5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3)
    
    ax[0].imshow(normalize2plot(image1[:,:,bands]))
    ax[0].set_title('Image T1')
    if not axis:
        ax[0].axis('off')
    
    ax[1].imshow(normalize2plot(image2[:,:,bands]))
    ax[1].set_title('Image T2')
    if not axis:
        ax[1].axis('off')
    
    ax[2].imshow(gt, cmap=plt.cm.gray)
    ax[2].set_title('Ground truth')
    if not axis:
        ax[2].axis('off')
    
    return fig, axes
    
def plot_changemap_plus_gt(changemap, gt, axis=True):
    """
    plot the calculated change map with the gt

    Parameters
    ----------
    changemap : numpy ndarray of shape (N,M)
        binary change map.
    gt : numpy ndarray of shape (N,M)
        ground truth
    axis : boolean, optional
        whether or not to plot the axis. The default is True.

    Returns
    -------
    fig : matplotlib Figure
        figure
    axes : list
        list of axes in the figure

    """
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2)
    
    ax[0].imshow(changemap, cmap=plt.cm.gray)
    ax[0].set_title('Change map')
    if not axis:
        ax[0].axis('off')
    
    ax[1].imshow(gt, cmap=plt.cm.gray)
    ax[1].set_title('Ground truth')
    if not axis:
        ax[1].axis('off')
        
    return fig, axes 

def plot_image_old(image_folder, image_list, bands, axis = False, normalize=True, titles = None):
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
            else:
                ax[r,c].set_title(image_list[i])
            if not axis:
                ax[r, c].axis('off')
            
    return fig, ax