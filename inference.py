#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:41:53 2020

@author: cordolo
"""
import os
import numpy as np
import torch
from torch.nn.functional import softmax

from plots import normalize2plot
from data_generator import channelsfirst
from train import get_downstream_network

def load_images(data_dir, filenames, channels=np.arange(13)):
    
    images = {}
    for filename in filenames:
        images[filename] = np.load(
            os.path.join(data_dir, filename))[:, :, channels]

        # replace potential nan values 
        images[filename][np.isnan(images[filename])] = 0
                       
    return images

def to_torch(images):
    
    for im in images:
        images[im] = torch.from_numpy(images[im]).unsqueeze(0)
        
    return images

def inference(directories, dataset_settings, model_settings, train_settings, 
              channels=np.arange(13), percentile=99, extract_features=None, 
              avg_pool=None, use_softmax=False):
    
    # get network
    n_branches = 2
    network, conv_classifier = get_downstream_network(model_settings, train_settings['gpu'], n_branches)
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
       
    for q, idx in enumerate(dataset_settings['indices_test']):
        # get filenames
        filenames = [str(idx)+'_a.npy', str(idx)+'_b.npy']
        # prepare images
        images = load_images(directories['data_path'], filenames, channels=channels)    
        for filename in filenames:
            images[filename] = normalize2plot(images[filename], percentile)      
        images = channelsfirst(images)
        images = to_torch(images)
                      
        out = inference_on_images(network=network, 
                                  images=images, 
                                  conv_classifier=conv_classifier, 
                                  n_branches=n_branches, 
                                  gpu=train_settings['gpu'], 
                                  extract_features=extract_features,
                                  avg_pool=avg_pool,
                                  use_softmax=use_softmax)
        
        
        #back to numpy
        prob_maps = out.squeeze().detach().cpu().numpy()
        
        save_networkname = model_settings['filename'].split('/')[-1]
        if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname)):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname))
        if not os.path.exists(os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps')):
            os.mkdir(os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps'))
        
        np.save(os.path.join(directories['results_dir_cd'], save_networkname,'probability_maps',str(idx)+'.png'), prob_maps)
    
        print('\r {}/{}'.format(q+1, len(dataset_settings['indices_test'])))
        
    print('all probability maps saved!')

def inference_on_images(network, images, conv_classifier, 
                        n_branches=2, gpu=None, extract_features=None,
                        avg_pool=None, use_softmax=False):
    
    network.eval() 
    
    inputs = list()
    for im in images: 
        if gpu != None:
            inputs.append(images[im].float().cuda())
        else:
            inputs.append(images[im].float())               
         
    # forward pass 
    with torch.no_grad():
        outputs = network(inputs, n_branches, extract_features=extract_features, 
                      avg_pool=avg_pool,conv_classifier=conv_classifier, 
                      use_softmax=use_softmax)

    probabilities = softmax(outputs, dim=1)

    return probabilities
      
   
