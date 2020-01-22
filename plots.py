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
     
def plot_random_imagepairs(n_imagepairs, image_folder, bands, axis=False, normalize=True):
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

    Returns
    -------
    fig : matplotlib Figure
        figure with the images plotted in subsplots
        
    Example
    -------
    fig = plot_random_imagepairs(2, os.path.join(intermediate_dir_training, data_dir), [8,3,2])

    """
    
    # list all images in the folder
    entries = os.listdir(image_folder)

    # take a random sample
    rand_im = sample(entries, n_imagepairs)
    rand_imidx = [s.split('_')[0] for s in rand_im]
    rand_impairs = [func(s) for s in rand_imidx for func in (lambda s: s+'_a.npy',lambda s: s+'_b.npy')]
    
    fig = plot_image(image_folder, rand_impairs, bands, axis, normalize)
    
    return fig

def plot_random_images(n_images, image_folder, bands, axis=False, normalize=True):
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

    Returns
    -------
    fig : matplotlib Figure
        figure with the images plotted in subsplots
        
    Example
    -------
    fig = plot_random_images(10, os.path.join(intermediate_dir_training, data_dir), [8,3,2])

    """
    
    # list all images in the folder
    entries = os.listdir(image_folder)

    # take a random sample
    rand_im = sample(entries, n_images)
    
    fig = plot_image(image_folder, rand_im, bands, axis, normalize)
    
    return fig
    
def plot_image(image_folder, image_list, bands, axis = False, normalize=True):
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

    Returns
    -------
    fig : matplotlib Figure
        figure with the images plotted in subsplots
        
    Example
    --------
    fig = plot_image(os.path.join(intermediate_dir_training, data_dir), ['240_a.npy','240_b.npy'], [8,3,2,])
        
    """
    
    n_images = len(image_list)
    assert n_images > 0, "empty image list"
    n_bands = len(bands)
    assert n_bands == 1 or n_bands == 3, "only images with 3 or 1 band allowed"
    rows = int(np.ceil(n_images / 2))
    cols = 1 if n_images < 2 else 2
    
    # prepare
    fig, ax = plt.subplots(rows,cols)
    
    for i in range(n_images):
        im = np.load((os.path.join(image_folder,image_list[i])))[:,:,bands]
        
        # normalize
        if normalize:
            for j in range(im.shape[2]):
                im[:,:,j] = np.divide(im[:,:,j]-im[:,:,j].min(),np.percentile(im[:,:,j],98)-im[:,:,j].min())

        # remove single-dimensional entries
        if n_bands == 1:
            im = np.squeeze(im)

        # plot image  
        r = int(np.floor(i/2))
        c = i%2
        if rows == 1:
            if cols == 1:
                ax.imshow(im)
                ax.set_title(image_list[i])
                if not axis:
                    ax.axis('off')
            else:
                ax[c].imshow(im) 
                ax[c].set_title(image_list[i])
                if not axis:
                    ax[c].axis('off')
        else:
            ax[r,c].imshow(im)
            ax[r,c].set_title(image_list[i])
            if not axis:
                ax[r, c].axis('off')
            
    return fig

