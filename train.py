#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:21:16 2020

@author: M. Leenstra
"""


from models.netbuilder import NetBuilder, create_loss_function, create_optimizer, init_weight
from data_generator import PairDataset, TripletDataset, TripletDatasetPreSaved
from data_generator import PartlyOverlapDataset, TripletAPNDataset, PairHardNegDataset
from data_generator import TripletAPNHardNegDataset, ShuffleBandDataset, TripletFromFileDataset
from data_generator import TripletAPNFinetuneDataset, PairDatasetOverlap, SupervisedDataset
from data_generator import SupervisedDatasetFromFile, TripletAPNFromFileDataset
from models.siameseNetAPNFinetune import siameseNetAPNFinetune
from models.siamese_net import make_conv_classifier
from utils import AverageMeter
import torch
from torch.utils.data import DataLoader
from torchvision import transforms 
import os
import time
import csv
import copy
import numpy as np


def train(directories, dataset_settings, network_settings, train_settings):

    # init tensorboard
    network_name = network_settings['network']
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    
    # init save-file
    fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
                  'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
                  'n_channels','patch_size','batch_norm','dataset','min_overlap',\
                  'max_overlap', 'best_acc','best_epoch', 'best_loss', 'weight', \
                  'global_avg_pool', 'basis', 'n_train_patches', 'patches_per_image', \
                  'n_val_patches', 'n_eval_patches', 'eval_acc', 'eval_loss', 'eval_prob',\
                  'tp_oscd_eval_triangle', 'tn_oscd_eval_triangle', 'fp_oscd_eval_triangle',\
                  'fn_oscd_eval_triangle', 'f1_oscd_eval_triangle']
    if not os.path.exists(os.path.join(directories['intermediate_dir'],directories['csv_models'])):
        with open(os.path.join(directories['intermediate_dir'],directories['csv_models']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()
            
    # init save-file for patch starts during training
    fieldnames_trainpatches = ['epoch', 'im_idx', 
                           'row_patch0', 'col_patch0',
                           'row_patch1', 'col_patch1',
                           'row_patch2', 'col_patch2']
    if not os.path.exists(directories['model_dir']):
        os.mkdir(directories['model_dir'])
    if not os.path.exists(os.path.join(directories['model_dir'],
                        'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime))):
        with open(os.path.join(directories['model_dir'],
                        'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime)), 'a') as file2:
            filewriter = csv.DictWriter(file2, fieldnames_trainpatches, delimiter = ",")
            filewriter.writeheader()
                
    # build network
    network_settings['im_size'] = (1,1)
    n_branches, pos_weight = determine_branches(network_settings, dataset_settings)
                
    network = NetBuilder.build_network(
        net=network_settings['network'],
        cfg=network_settings['cfg'],
        n_channels=len(dataset_settings['channels']), 
        n_classes=network_settings['n_classes'],
        patch_size=network_settings['patch_size'],
        im_size=network_settings['im_size'],
        batch_norm=network_settings['batch_norm'],
        n_branches=n_branches,
        weights=network_settings['weights_file'],
        gpu=train_settings['gpu'])  
    conv_classifier = True if network_settings['cfg']['classifier'][0] == 'C' else False
    
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
           
        # Datasets
    if dataset_settings['dataset_type'] == 'supervised_from_file':
         dataset_settings['df'] = dataset_settings['dataset_train_df']
    dataset_train = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_train'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)
    if dataset_settings['dataset_type'] == 'supervised_from_file':
         dataset_settings['df'] = dataset_settings['dataset_val_df']    
    dataset_val = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_val'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)

    # Data loaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=train_settings['batch_size'], 
        shuffle=True,
        num_workers = 10)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 5)
    
    if dataset_settings['dataset_type'] == 'supervised' or dataset_settings['dataset_type'] == 'supervised_from_file':
        pos_weight = dataset_train.pos_weight
        loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
     
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
        loss_func = loss_func.cuda()

    optim = create_optimizer(network_settings['optimizer'], network.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])
    # save history
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    
    epoch_iters =  max(len(dataset_train) // train_settings['batch_size'],1)
    val_epoch_iters = max(len(dataset_val) // train_settings['batch_size'],1)
    
    best_net_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = 99999
    
    train_functions = {'train_normal': train_epoch,
                       'train_apn': train_epoch_apn,
                       'train_unet': train_epoch_unet}
    val_functions = {'val_normal': validate_epoch,
                     'val_apn': validate_epoch_apn,
                     'val_unet': validate_epoch_unet}
    
    if network_settings['network'] == 'triplet_apn' or network_settings['network'] == 'triplet_apn_dilated':
        train_func = train_functions['train_apn']
        val_func = val_functions['val_apn']
    elif network_settings['network'] == 'triplet_unet':
        train_func = train_functions['train_unet']
        val_func = val_functions['val_unet']
    else:
        train_func = train_functions['train_normal']
        val_func = val_functions['val_normal']
    
    try:            
        for epoch in range(train_settings['start_epoch'], 
                           train_settings['start_epoch']+train_settings['num_epoch']):
            
            # early stopping
            if (epoch - best_epoch) > train_settings['early_stopping']:
                break
                       
            #training epoch
            train_func(
                network_settings,
                network=network, 
                n_branches=n_branches,
                dataloader=dataloader_train, 
                optimizer=optim, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=history, 
                epoch=epoch, 
                writer=None,
                epoch_iters=epoch_iters, 
                disp_iter=train_settings['disp_iter'],
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'],
                directories=directories, 
                network_name=network_name, 
                outputtime=outputtime,
                conv_classifier=conv_classifier,
                model_dir = directories['model_dir'])
            
            # validation epoch
            best_net_wts, best_acc, best_epoch, best_loss = val_func(
                network_settings,
                network=network, 
                n_branches=n_branches,
                dataloader=dataloader_val, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=history, 
                epoch=epoch, 
                writer=None,
                val_epoch_iters=val_epoch_iters,
                best_net_wts=best_net_wts,
                best_acc=best_acc,
                best_epoch=best_epoch,
                best_loss=best_loss,
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'],
                directories=directories, 
                network_name=network_name, 
                outputtime=outputtime,
                conv_classifier=conv_classifier,
                model_dir = directories['model_dir'])
    
    # on keyboard interupt continue the script: saves the best model until interrupt
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving progress...")
        
    if network_settings['cfg']['top'] != None: 
        top = str(list(network_settings['cfg']['top']))
    else:
        top = 'None'
    # write info to csv-file
    savedata = {'filename':os.path.join(directories['model_dir'],
                        'network-{}_date-{}'.format(network_name, outputtime)), 
               'networkname': network_name, 
               'cfg_branch': str(list(network_settings['cfg']['branch'])), 
               'cfg_top': top,
               'cfg_classifier': str(list(network_settings['cfg']['classifier'])),
               'optimizer': network_settings['optimizer'],
               'lr': network_settings['lr'], 
               'weight_decay': network_settings['weight_decay'],
               'loss': network_settings['loss'],
               'n_classes': network_settings['n_classes'],
               'n_channels' : len(dataset_settings['channels']),
               'patch_size': network_settings['patch_size'],
               'batch_norm': network_settings['batch_norm'],
               'dataset': dataset_settings['dataset_type'], 
               'min_overlap': dataset_settings['min_overlap'],
               'max_overlap': dataset_settings['max_overlap'],
               'best_acc': best_acc,
               'best_epoch': best_epoch, 
               'best_loss': best_loss,
               'weight': None,
               'global_avg_pool': 'False', 
               'basis': None,
               'n_train_patches': len(dataset_settings['indices_train']), 
               'patches_per_image': dataset_settings['patches_per_image'],
               'n_val_patches': len(dataset_settings['indices_val'])}
    with open(os.path.join(directories['intermediate_dir'], 
                           directories['csv_models']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writerow(savedata)  
    
    # save best model's weights
    torch.save(best_net_wts, savedata['filename'])
    torch.save(history, os.path.join(directories['model_dir'],
                        'history_network-{}_date-{}'.format(network_name, outputtime)))
        
    print('Training Done!')

    

def evaluate(model_settings, directories, dataset_settings, network_settings, train_settings):
    
    # build network
    network_settings['im_size'] = (1,1)
    model_settings['network'] = model_settings['networkname']
    n_branches, pos_weight = determine_branches(model_settings, dataset_settings)
    net = get_network(model_settings, train_settings['gpu'])
    classifier = model_settings['cfg_classifier'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    conv_classifier = True if classifier[0] == 'C' else False
    
    loss_func, acc_func, one_hot = create_loss_function(model_settings['loss'])
    
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        net.cuda()
        loss_func = loss_func.cuda()
    
    # get dataset
    dataset_eval = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_eval'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)
                        
    # Data loaders
    dataloader = DataLoader(
        dataset_eval, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 1)
    
    val_epoch_iters = max(len(dataset_eval) // train_settings['batch_size'],1)
    best_net_wts = copy.deepcopy(net.state_dict())
    
    # validation epoch
    best_acc, best_loss, best_prob = validate_epoch(
        network_settings,
        network=net, 
        n_branches=n_branches,
        dataloader=dataloader, 
        loss_func=loss_func,
        acc_func=acc_func,
        history=None, 
        epoch=0, 
        writer=None,
        val_epoch_iters=val_epoch_iters,
        best_net_wts=best_net_wts,
        best_acc=0.0,
        best_epoch=0,
        best_loss=99999,
        gpu = train_settings['gpu'],
        im_size = network_settings['im_size'],
        extract_features=None,
        avg_pool=network_settings['avg_pool'],
        directories=directories, 
        network_name=model_settings['networkname'], 
        outputtime=model_settings['filename'].split('-')[-1],
        inference=True,
        conv_classifier=conv_classifier,
        model_dir=directories['model_dir'])

    return best_acc, best_loss, best_prob

def evaluate_apn(model_settings, directories, dataset_settings, network_settings, train_settings):
    
    # build network
    network_settings['im_size'] = (1,1)
    model_settings['network'] = model_settings['networkname']
    n_branches, pos_weight = determine_branches(model_settings, dataset_settings)
    net = get_network(model_settings, train_settings['gpu'])
    classifier = model_settings['cfg_classifier'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    conv_classifier = True if classifier[0] == 'C' else False
    
    loss_func, acc_func, one_hot = create_loss_function(model_settings['loss'])
    
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        net.cuda()
        loss_func = loss_func.cuda()
    
    # get dataset
    dataset_eval = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_eval'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)
                        
    # Data loaders
    dataloader = DataLoader(
        dataset_eval, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 1)
    
    val_epoch_iters = max(len(dataset_eval) // train_settings['batch_size'],1)
    best_net_wts = copy.deepcopy(net.state_dict())
    
    # validation epoch
    best_acc, best_loss = validate_epoch_apn(
        network_settings,
        network=net, 
        n_branches=n_branches,
        dataloader=dataloader, 
        loss_func=loss_func,
        acc_func=acc_func,
        history=None, 
        epoch=0, 
        writer=None,
        val_epoch_iters=val_epoch_iters,
        best_net_wts=best_net_wts,
        best_acc=0.0,
        best_epoch=0,
        best_loss=99999,
        gpu = train_settings['gpu'],
        im_size = network_settings['im_size'],
        extract_features=None,
        avg_pool=network_settings['avg_pool'],
        directories=directories, 
        network_name=model_settings['networkname'], 
        outputtime=model_settings['filename'].split('-')[-1],
        inference=True,
        conv_classifier=conv_classifier,
        model_dir=directories['model_dir'])

    return best_acc, best_loss

    
def use_features_downstream(model_settings, directories, dataset_settings, network_settings, train_settings, finetune=True):

    # init 
    if network_settings['concat']:
        network_name = 'CD_cat_'+model_settings['networkname']
    else:
        network_name = 'CD_'+model_settings['networkname']
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
 
    # init save-file
    fieldnames = ['filename','pretask_filename','networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
                  'layers_branches', 'layers_joint', 'cfg_classifier_cd',\
                  'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
                  'n_channels','patch_size','batch_norm','dataset','min_overlap',\
                  'max_overlap', 'best_avg_acc','best_epoch', 'best_loss', 'weight', \
                  'global_avg_pool', 'n_train_patches', 'patches_per_image', \
                  'n_val_patches', 'kthfold', 'n_eval_patches', 'eval_acc', 'eval_loss', 'eval_prob',\
                  'tp_oscd_eval_triangle', 'tn_oscd_eval_triangle', 'fp_oscd_eval_triangle',\
                  'fn_oscd_eval_triangle', 'f1_oscd_eval_triangle']
    if finetune:
        if not os.path.exists(os.path.join(directories['intermediate_dir'],directories['csv_models_finetune'])):
            with open(os.path.join(directories['intermediate_dir'],directories['csv_models_finetune']), 'a') as file:
                filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
                filewriter.writeheader()
        model_dir = directories['model_dir_finetune']
    else:
        if not os.path.exists(os.path.join(directories['intermediate_dir'],directories['csv_models_downstream'])):
            with open(os.path.join(directories['intermediate_dir'],directories['csv_models_downstream']), 'a') as file:
                filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
                filewriter.writeheader()
        model_dir = directories['model_dir_downstream']
           
    # get network
    network_settings['im_size'] = (1,1)
    model_settings['network'] = model_settings['networkname']
    n_branches = 2
    pos_weight = 1
    network = get_network(model_settings, train_settings['gpu'])  
    # freeze layers
    if not finetune:
        for param in network.parameters():
            param.requires_grad = False
        
    # add classifier to specified layer
    network = NetBuilder.build_cd_network(
        network=network, 
        network_name=network_name,
        network_settings=network_settings)
    conv_classifier = True if network_settings['cfg']['classifier_cd'][0] == 'C' else False     
    # change network name for saving if necessary
    if finetune:
        if network_settings['concat']:
            network_name = 'FINETUNE_cat_'+model_settings['networkname']
        else:
            network_name = 'FINETUNE_'+model_settings['networkname']
 
    
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
        
    # Datasets
    if dataset_settings['dataset_type'] == 'supervised_from_file':
         dataset_settings['df'] = dataset_settings['dataset_train_df']
    dataset_train = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_train'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)
    if dataset_settings['dataset_type'] == 'supervised_from_file':
         dataset_settings['df'] = dataset_settings['dataset_val_df']    
    dataset_val = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_val'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)

    # Data loaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=train_settings['batch_size'], 
        shuffle=True,
        num_workers = 1)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 1)
    
    if dataset_settings['dataset_type'] == 'supervised' or dataset_settings['dataset_type'] == 'supervised_from_file':
        pos_weight = dataset_train.pos_weight
        loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
        
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
        loss_func = loss_func.cuda()

    optim = create_optimizer(network_settings['optimizer'], network.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])
     
    # save history
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    
    epoch_iters =  max(len(dataset_train) // train_settings['batch_size'],1)
    val_epoch_iters = max(len(dataset_val) // train_settings['batch_size'],1)
    
    best_net_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = 99999
    
    
    try:            
        for epoch in range(train_settings['start_epoch'], 
                           train_settings['start_epoch']+train_settings['num_epoch']):
            
            # early stopping
            if (epoch - best_epoch) > train_settings['early_stopping']:
                break
            if ((epoch+1) % 5) == 0:
                best_acc = 0.0
                best_epoch = 0
                best_loss = 99999
                       
            #training epoch
            train_epoch(
                network_settings,
                network=network, 
                n_branches=n_branches,
                dataloader=dataloader_train, 
                optimizer=optim, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=history, 
                epoch=epoch, 
                writer=None,
                epoch_iters=epoch_iters, 
                disp_iter=train_settings['disp_iter'],
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'],
                directories=directories, 
                network_name=network_name, 
                outputtime=outputtime,
                conv_classifier=conv_classifier,
                model_dir=model_dir)
            
            # validation epoch
            best_net_wts, best_acc, best_epoch, best_loss = validate_epoch(
                network_settings,
                network=network, 
                n_branches=n_branches,
                dataloader=dataloader_val, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=history, 
                epoch=epoch, 
                writer=None,
                val_epoch_iters=val_epoch_iters,
                best_net_wts=best_net_wts,
                best_acc=best_acc,
                best_epoch=best_epoch,
                best_loss=best_loss,
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'],
                directories=directories, 
                network_name=network_name, 
                outputtime=outputtime,
                conv_classifier=conv_classifier,
                model_dir=model_dir)
            
            
    
    # on keyboard interupt continue the script: saves the best model until interrupt
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving progress...")
        
    
    # TODO: write info to csv-file
    savedata = {'filename': os.path.join(model_dir,
                        'network-{}_date-{}'.format(network_name, outputtime)), 
                'pretask_filename': model_settings['filename'],
                'networkname': network_name,
                'cfg_branch': model_settings['cfg_branch'],
                'cfg_top': model_settings['cfg_top'], 
                'cfg_classifier': model_settings['cfg_classifier'],
                'layers_branches': network_settings['layers_branches'], 
                'layers_joint': network_settings['layers_joint'], 
                'cfg_classifier_cd': str(list(network_settings['cfg']['classifier_cd'])),
                'optimizer': network_settings['optimizer'],
                'lr': network_settings['lr'], 
                'weight_decay': network_settings['weight_decay'],
                'loss': network_settings['loss'],                
                'n_classes': network_settings['n_classes'],
                'n_channels' : len(dataset_settings['channels']),
                'patch_size': network_settings['patch_size'],
                'batch_norm': network_settings['batch_norm'],
                'dataset': dataset_settings['dataset_type'], 
                'min_overlap': 1,
                'max_overlap': 1,
                'best_avg_acc': best_acc,
                'best_epoch': best_epoch, 
                'best_loss': best_loss,
                'weight': None,
                'global_avg_pool': 'False', 
                'basis': None,
                'n_train_patches': len(dataset_settings['indices_train']), 
                'patches_per_image': dataset_settings['patches_per_image'],
                'n_val_patches': len(dataset_settings['indices_val']),
                'kthfold': dataset_settings['kthfold']}

    if finetune:
        with open(os.path.join(directories['intermediate_dir'],directories['csv_models_finetune']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
            filewriter.writerow(savedata)  
    else:
        with open(os.path.join(directories['intermediate_dir'],directories['csv_models_downstream']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
            filewriter.writerow(savedata)  
    
    # save best model's weights
    torch.save(best_net_wts, savedata['filename'])
    torch.save(history, os.path.join(model_dir,
                        'history_network-{}_date-{}'.format(network_name, outputtime)))
        
    print('Done!')
    #writer.close()
    
    
    
def train_epoch(network_settings, network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, model_dir,
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False, conv_classifier=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_loss_all = AverageMeter()
    ave_acc = AverageMeter()
    
    network.train() 
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    epoch_start = time.time()   
    for i, batch_data in enumerate(iterator): 
        
        for item in batch_data: 
            batch_data[item] = torch.squeeze(batch_data[item])

        data_time.update(time.time() - tic)
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # get the inputs
        if n_branches == 1:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda()]                        
            else:
                inputs = [batch_data['patch0'].float()] 
        elif n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 

        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']

    
        # to gpu
        if gpu != None:
            labels = labels.cuda()
            
        # forward pass 
        outputs = network(inputs, n_branches, extract_features=extract_features, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=False)

        if network_settings['loss'] == 'bce_sigmoid+l1reg':  
            loss = loss_func(outputs, labels, network.parameters())
        else:
            loss = loss_func(outputs, labels)
        acc = acc_func(outputs, labels, im_size)

        # Backward
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_acc.update(acc.item())
        
        # save the starting points in csvfile for reproducability
        starts = batch_data['patch_starts'].detach().numpy()
        im_idx = batch_data['im_idx'].detach().numpy()
    
        with open(os.path.join(model_dir,
                'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime)), 'a') as file2:
            filewriter = csv.DictWriter(file2, fieldnames_trainpatches, delimiter = ",", extrasaction='ignore')
            for j in range(len(im_idx)): 
                row_patch0 = starts[j][0][0]
                col_patch0 = starts[j][0][1]
                if n_branches == 2: 
                    row_patch1 = starts[j][1][0]
                    col_patch1 = starts[j][1][1]
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0,
                                         'row_patch1':row_patch1, 'col_patch1':col_patch1})  
                elif n_branches == 3:
                    row_patch2 = starts[j][2][0]
                    col_patch2 = starts[j][2][1]
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0,
                                         'row_patch1':row_patch1, 'col_patch1':col_patch1,
                                         'row_patch2':row_patch2, 'col_patch2':col_patch2})  
                else:
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0})  

        

        # calculate accuracy, and display
        if (i+1) % disp_iter == 0:
            print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                  'Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch, i+1, epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_loss.average(), ave_acc.average()))
            ave_loss_all.update(ave_loss.average())
            ave_loss = AverageMeter()
            
    if (i+1) < disp_iter:
        ave_loss_all_print = ave_loss.average()
    else:
        ave_loss_all_print = ave_loss_all.average()
    print('Train epoch: [{}], Time: {:.2f} ' 
          'Train_Loss: {:.4f}, Train_Accuracy: {:0.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all_print, ave_acc.average()))
    
    if writer != None:
        writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
        writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
    
    if history != None:
        history['train']['epoch'].append(epoch)
        history['train']['loss'].append(ave_loss_all.average())
        history['train']['acc'].append(ave_acc.average())


    
def validate_epoch(network_settings, network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, model_dir,
             extract_features=None, avg_pool=False, inference=False, conv_classifier=False):    

    time_meter = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    
    use_softmax = False
    if inference: 
        probability = AverageMeter()
        sigmoid = torch.nn.Sigmoid()
        
    network.eval() 
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    #for i in range(epoch_iters):
    for i, batch_data in enumerate(iterator): 
        
        for item in batch_data: 
            batch_data[item] = torch.squeeze(batch_data[item])      
        
        # get the inputs
        if n_branches == 1:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda()]                        
            else:
                inputs = [batch_data['patch0'].float()] 
        elif n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 

        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']
    
        # to gpu
        if gpu != None:
            labels = labels.cuda()
            
        # forward pass 
        with torch.no_grad():
            outputs = network(inputs, n_branches, extract_features=extract_features, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=use_softmax)

        if network_settings['loss'] == 'bce_sigmoid+l1reg':  
            loss = loss_func(outputs, labels, network.parameters())
        else:
            loss = loss_func(outputs, labels)
        acc = acc_func(outputs, labels, im_size)    
        
        if inference:
            prob = sigmoid(outputs)
            correct = labels*prob
            prob_correct = torch.sum(correct)/len(outputs)       
            probability.update(prob_correct.item())

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_acc.update(acc.item())

        # measure elapsed time
        time_meter.update(time.time() - tic)
        tic = time.time()


    print('Val epoch: [{}], Time: {:.2f}, ' 
          'Val_Loss: {:.4f}, Val_Accuracy: {:0.4f}'
          .format(epoch, time_meter.value(),
                  ave_loss.average(), ave_acc.average()))
    
    if inference:
        best_acc = ave_acc.average()
        best_loss = ave_loss.average()
        best_prob = probability.average()
        return best_acc, best_loss, best_prob
    else:
        if epoch+1 % 5 == 0:
            torch.save(copy.deepcopy(network.state_dict()), os.path.join(model_dir,
                            'network-{}_date-{}_epoch-{}'.format(
                                network_name, outputtime, epoch)))
        
        if writer != None:
            writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
            writer.add_scalar('Val/Acc', ave_acc.average(), epoch)
    
        if ave_loss.average() < best_loss:
            best_acc = ave_acc.average()
            best_net_wts = copy.deepcopy(network.state_dict())
            best_epoch = epoch
            best_loss = ave_loss.average()
            torch.save(best_net_wts, os.path.join(model_dir,
                            'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
        
        if history != None:
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(ave_loss.average())
            history['val']['acc'].append(ave_acc.average())
        
        return(best_net_wts, best_acc, best_epoch, best_loss)

def train_epoch_apn(network_settings, network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, model_dir,
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False, conv_classifier=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_lossL1 = AverageMeter()
    ave_lossTriplet = AverageMeter()
    ave_loss_all = AverageMeter()
    ave_acc = AverageMeter()
    
    network.train() 
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    epoch_start = time.time()
    #for i in range(epoch_iters):
    for i, batch_data in enumerate(iterator): 
        
        for item in batch_data: 
            batch_data[item] = torch.squeeze(batch_data[item])

        data_time.update(time.time() - tic)
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # get the inputs
        if n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 

        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']
        #labels = batch_data['gt']
    
        # to gpu
        if gpu != None:
            labels = labels.cuda()
            
        # forward pass
        outputs = network(inputs, n_branches, extract_features=extract_features, 
                          avg_pool=avg_pool,conv_classifier=conv_classifier, 
                          use_softmax=False)
            
        #outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
        if torch.any(torch.isnan(outputs[0])) or torch.any(torch.isnan(outputs[1])):
            print("outputs is nan")
            import ipdb
            ipdb.set_trace()

        #loss = loss_func(outputs, labels)
        loss, loss1, loss2 = loss_func(outputs, labels)
        acc = acc_func(outputs, labels, im_size)

        # Backward
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_lossL1.update(loss1.data.item())
        ave_lossTriplet.update(loss2.data.item())
        ave_acc.update(acc.item())
        
        # save the starting points in csvfile for reproducability
        starts = batch_data['patch_starts'].detach().numpy()
        im_idx = batch_data['im_idx'].detach().numpy()
    
        with open(os.path.join(model_dir,
                'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime)), 'a') as file2:
            filewriter = csv.DictWriter(file2, fieldnames_trainpatches, delimiter = ",", extrasaction='ignore')
            for j in range(len(im_idx)): 
                row_patch0 = starts[j][0][0]
                col_patch0 = starts[j][0][1]
                if n_branches == 2: 
                    row_patch1 = starts[j][1][0]
                    col_patch1 = starts[j][1][1]
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0,
                                         'row_patch1':row_patch1, 'col_patch1':col_patch1})  
                if n_branches == 3:
                    row_patch1 = starts[j][1][0]
                    col_patch1 = starts[j][1][1]
                    row_patch2 = starts[j][2][0]
                    col_patch2 = starts[j][2][1]
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0,
                                         'row_patch1':row_patch1, 'col_patch1':col_patch1,
                                         'row_patch2':row_patch2, 'col_patch2':col_patch2})  
                else:
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0})  


        # calculate accuracy, and display
        if (i+1) % disp_iter == 0:
            print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                  'Loss: {:.4f}, L1: {:.4f}, LTriplet: {:.4f}'
                  .format(epoch, i+1, epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_loss.average(), 
                          ave_lossL1.average(), ave_lossTriplet.average()))
            ave_loss_all.update(ave_loss.average())
            ave_loss = AverageMeter()
            ave_lossL1 = AverageMeter()
            ave_lossTriplet = AverageMeter()
            
        
    if (i+1) < disp_iter:
        ave_loss_all_print = ave_loss.average()
    else:
        ave_loss_all_print = ave_loss_all.average()
    print('Train epoch: [{}], Time: {:.2f} ' 
          'Train_Loss: {:.4f}, Train_Accuracy: {:0.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all_print, ave_acc.average()))

     
    if writer != None:
        writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
        writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
    
    if history != None:
        history['train']['epoch'].append(epoch)
        history['train']['loss'].append(ave_loss_all.average())
        history['train']['acc'].append(ave_acc.average())


    
def validate_epoch_apn(network_settings, network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, model_dir,
             extract_features=None, avg_pool=False, inference=False, conv_classifier=False):    

    ave_loss = AverageMeter()
    ave_lossL1 = AverageMeter()
    ave_lossTriplet = AverageMeter()
    ave_acc = AverageMeter()
    time_meter = AverageMeter()
    
    network.eval()
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    for i, batch_data in enumerate(iterator):  

        for item in batch_data: 
            batch_data[item] = torch.squeeze(batch_data[item])
        
        # get the inputs
        if n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 
        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']

        # to GPU
        if gpu != None:
            labels = labels.cuda()

      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
            loss, loss1, loss2  = loss_func(outputs, labels)
            acc = acc_func(outputs, labels, im_size)
        
        loss = loss.mean()
        acc = acc.mean()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_lossL1.update(loss1.data.item())
        ave_lossTriplet.update(loss2.data.item())
        ave_acc.update(acc.item())

        # measure elapsed time
        time_meter.update(time.time() - tic)
        tic = time.time()

    print('Val epoch: [{}], Time: {:.2f}, '
          'Val_Loss: {:.4f}, L1: {:.4f}, LTriplet: {:.4f}'
                  .format(epoch, time_meter.value(),
                  ave_loss.average(), ave_lossL1.average(), 
                  ave_lossTriplet.average()))
    
    if inference:
        best_acc = ave_acc.average()
        best_loss = ave_loss.average()
        return best_acc, best_loss
    else: 
        if writer != None:
            writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
            writer.add_scalar('Val/Acc', ave_acc.average(), epoch)
        
        if ave_loss.average() < best_loss:
            best_acc = ave_acc.average()
            best_net_wts = copy.deepcopy(network.state_dict())
            best_epoch = epoch
            best_loss = ave_loss.average()
            torch.save(best_net_wts, os.path.join(model_dir,
                            'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
        
        if history != None:
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(ave_loss.average())
            history['val']['acc'].append(ave_acc.average())
        
        return(best_net_wts, best_acc, best_epoch, best_loss)

def train_epoch_unet(network_settings, network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, model_dir,
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False, conv_classifier=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_lossL1 = AverageMeter()
    ave_lossTriplet = AverageMeter()
    ave_lossBCE = AverageMeter()
    ave_loss_all = AverageMeter()
    ave_acc = AverageMeter()
    ave_acc_all = AverageMeter()
    
    network.train() 
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    epoch_start = time.time()
    for i, batch_data in enumerate(iterator): 
        
        for item in batch_data: 
            batch_data[item] = torch.squeeze(batch_data[item])

        data_time.update(time.time() - tic)
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # get the inputs
        if n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 

        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']
        labels_bottle = batch_data['label_bottle']

    
        # to gpu
        if gpu != None:
            labels = labels.cuda()
            labels_bottle = labels_bottle.cuda()
            
        # forward pass
        outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
        loss, loss1, loss2, loss3 = loss_func(outputs, labels, labels_bottle)
        acc = acc_func(outputs[0], labels_bottle, im_size)

        # Backward
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_loss_all.update(loss.data.item())
        ave_lossL1.update(loss1.data.item())
        ave_lossTriplet.update(loss2.data.item())
        ave_lossBCE.update(loss3.data.item())
        ave_acc.update(acc.item())
        ave_acc_all.update(acc.item())
        
        # save the starting points in csvfile for reproducability
        starts = batch_data['patch_starts'].detach().numpy()
        im_idx = batch_data['im_idx'].detach().numpy()
    
        with open(os.path.join(model_dir,
                'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime)), 'a') as file2:
            filewriter = csv.DictWriter(file2, fieldnames_trainpatches, delimiter = ",", extrasaction='ignore')
            for j in range(len(im_idx)): 
                row_patch0 = starts[j][0][0]
                col_patch0 = starts[j][0][1]
                if n_branches == 2: 
                    row_patch1 = starts[j][1][0]
                    col_patch1 = starts[j][1][1]
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0,
                                         'row_patch1':row_patch1, 'col_patch1':col_patch1})  
                if n_branches == 3:
                    row_patch2 = starts[j][2][0]
                    col_patch2 = starts[j][2][1]
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0,
                                         'row_patch1':row_patch1, 'col_patch1':col_patch1,
                                         'row_patch2':row_patch2, 'col_patch2':col_patch2})  
                else:
                    filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],
                                         'row_patch0':row_patch0, 'col_patch0':col_patch0})  


        # calculate accuracy, and display
        if (i+1) % disp_iter == 0:
            print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                  'Loss: {:.4f}, L1: {:.4f}, LTriplet: {:.4f}, LBCE: {:.4f}, Acc_bce: {:.4f}'
                  .format(epoch, i+1, epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_loss.average(), 
                          ave_lossL1.average(), ave_lossTriplet.average(),
                          ave_lossBCE.average(),
                          ave_acc.average()))
            ave_loss = AverageMeter()
            ave_lossL1 = AverageMeter()
            ave_lossTriplet = AverageMeter()
            ave_lossBCE = AverageMeter()
            ave_acc = AverageMeter()
            
        
    if (i+1) < disp_iter:
        ave_loss_all_print = ave_loss.average()
    else:
        ave_loss_all_print = ave_loss_all.average()
    print('Train epoch: [{}], Time: {:.2f} ' 
          'Train_Loss: {:.4f}, Train_Accuracy: {:0.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all_print, ave_acc.average()))

     
    if writer != None:
        writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
        writer.add_scalar('Train/Acc', ave_acc_all.average(), epoch)
    
    if history != None:
        history['train']['epoch'].append(epoch)
        history['train']['loss'].append(ave_loss_all.average())
        history['train']['acc'].append(ave_acc.average())


    
def validate_epoch_unet(network_settings, network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, model_dir,
             extract_features=None, avg_pool=False, evaluate=False, conv_classifier=False):    

    ave_loss = AverageMeter()
    ave_lossL1 = AverageMeter()
    ave_lossTriplet = AverageMeter()
    ave_lossBCE = AverageMeter()
    ave_acc = AverageMeter()
    time_meter = AverageMeter()
    
    if evaluate:
        probability = AverageMeter()
        sigmoid = torch.nn.Sigmoid()


    network.eval()
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    for i, batch_data in enumerate(iterator):  

        for item in batch_data: 
            batch_data[item] = torch.squeeze(batch_data[item])
        
        # get the inputs
        if n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 
        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']
        labels_bottle = batch_data['label_bottle']

        # to GPU
        if gpu != None:
            labels = labels.cuda()
            labels_bottle = labels_bottle.cuda()
      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
            if evaluate:
                prob = sigmoid(outputs)
                correct = labels*prob
                prob_correct = torch.sum(correct)/len(outputs)       
                probability.update(prob_correct.item())
            loss, loss1, loss2, loss3  = loss_func(outputs, labels, labels_bottle)
            acc = acc_func(outputs[0], labels_bottle, im_size)
        
        loss = loss.mean()
        acc = acc.mean()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_lossL1.update(loss1.data.item())
        ave_lossTriplet.update(loss2.data.item())
        ave_lossBCE.update(loss3.data.item())
        ave_acc.update(acc.item())

        # measure elapsed time
        time_meter.update(time.time() - tic)
        tic = time.time()

    print('Val epoch: [{}], Time: {:.2f}, '
          'Val_Loss: {:.4f}, L1: {:.4f}, LTriplet: {:.4f}, LBCE: {:.4f}, Acc: {:.4f}'
                  .format(epoch, time_meter.value(),
                  ave_loss.average(), ave_lossL1.average(), 
                  ave_lossTriplet.average(), ave_lossBCE.average(),
                  ave_acc.average()))

    if evaluate:
        best_acc = ave_acc.average()
        best_loss = ave_loss.average()
        best_prob = probability.average()
        return best_acc, best_loss, best_prob
    else: 
        if writer != None:
            writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
            writer.add_scalar('Val/Acc', ave_acc.average(), epoch)
        
        if ave_loss.average() < best_loss:
            best_acc = ave_acc.average()
            best_net_wts = copy.deepcopy(network.state_dict())
            best_epoch = epoch
            best_loss = ave_loss.average()
            torch.save(best_net_wts, os.path.join(model_dir,
                            'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
        
        if history != None:
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(ave_loss.average())
            history['val']['acc'].append(ave_acc.average())
        
        return(best_net_wts, best_acc, best_epoch, best_loss)

def get_dataset(data_path, indices, channels, one_hot, dataset_settings, network_settings, directories):
        # Datasets
    if dataset_settings['dataset_type'] == 'pair':
        dataset = PairDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels)       
    elif dataset_settings['dataset_type'] == 'triplet':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = TripletDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'],
            in_memory = in_memory)     
    elif dataset_settings['dataset_type'] == 'triplet_saved':
        print("construct from presaved dataset")
        dataset = TripletDatasetPreSaved(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot)           
    elif dataset_settings['dataset_type'] == 'overlap':
        dataset = PartlyOverlapDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot)           
    elif dataset_settings['dataset_type'] == 'triplet_apn':
        second_label = True if network_settings['loss'] == 'l1+triplet+bce' else False
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = TripletAPNDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot,
            second_label=second_label,
            in_memory=in_memory)#,
    elif dataset_settings['dataset_type'] == 'overlap_regression':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = PairDatasetOverlap(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=False,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'],
            in_memory=in_memory)          
    elif dataset_settings['dataset_type'] == 'pair_hard_neg':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = PairHardNegDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot,
            in_memory = in_memory)
    elif dataset_settings['dataset_type'] == 'triplet_apn_hard_neg':
        second_label = True if network_settings['loss'] == 'l1+triplet+bce' else False
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = TripletAPNHardNegDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot,
            second_label=second_label,
            in_memory=in_memory)
    elif dataset_settings['dataset_type'] == 'shuffle_band':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = ShuffleBandDataset(
            data_dir=data_path,
            indices=indices,
            channels=channels, 
            one_hot=one_hot,
            in_memory=in_memory)
    elif dataset_settings['dataset_type'] == 'triplet_from_file':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = TripletFromFileDataset(
            data_dir = data_path,
            indices = indices,
            patch_starts_df = dataset_settings['dataset_eval_df'],
            channels=channels,
            one_hot=one_hot,
            in_memory=in_memory)
    elif dataset_settings['dataset_type'] == 'triplet_apn_from_file':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = TripletAPNFromFileDataset(
            data_dir = data_path,
            indices = indices,
            patch_starts_df = dataset_settings['dataset_eval_df'],
            channels=channels,
            one_hot=one_hot,
            in_memory=in_memory)
    elif dataset_settings['dataset_type'] == 'supervised':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = SupervisedDataset(
            data_dir=data_path,
            indices=indices, 
            labels_dir=directories['labels_path'], 
            channels=channels,
            one_hot=one_hot,
            in_memory=in_memory)
    elif dataset_settings['dataset_type'] == 'supervised_from_file':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = SupervisedDatasetFromFile(
            data_dir=data_path,
            indices=indices, 
            patch_starts_df = dataset_settings['df'],
            labels_dir=directories['labels_path'], 
            channels=channels,
            one_hot=one_hot,
            in_memory=in_memory)
    else:
        raise Exception('dataset_type undefined! \n \
                        Choose one of: "pair", "triplet", "triplet_saved",\
                        "overlap", "triplet_apn", "triplet_oscd", "overlap_regression", \
                        "pair_hard_neg", "triplet_apn_hard_neg", "triplet_from_file",\
                        "supervised", "supervised_from_file"')
    return dataset

def determine_branches(network_settings, dataset_settings):
    pos_weight = 1
    if network_settings['network'] == 'siamese':
        print('n_branches = 2')
        n_branches = 2
    elif network_settings['network'] == 'siamese_concat':
        print('n_branches = 2')
        n_branches = 2
    elif network_settings['network'] == 'triplet':
        print('n_branches = 3')
        n_branches = 3
        network_settings['network'] = 'siamese'  
    elif network_settings['network'] == 'single':
        print('n_branches = 1')
        n_branches = 1
        network_settings['network'] = 'siamese'    
    elif network_settings['network'] == 'hypercolumn':
        print('n_branches = 2')
        n_branches = 2
        mean_overlap = np.mean([dataset_settings['min_overlap'], 
                                dataset_settings['max_overlap']])
        pos_weight = (1-mean_overlap)/mean_overlap
    elif network_settings['network'] == 'siamese_unet_diff':
        print('n_branches = 2')
        n_branches = 2
    elif network_settings['network'] == 'triplet_apn':
        print('n_branches = 3')
        n_branches = 3 
    elif network_settings['network'] == 'siamese_unet':
        print('n_branches = 2')
        n_branches = 2
    elif network_settings['network'] == 'triplet_unet':
        print('n_branches = 3')
        n_branches = 3
    elif network_settings['network'] == 'siamese_dilated':
        print('n_branches = 2')
        n_branches = 2
    elif network_settings['network'] == 'triplet_apn_dilated':
        print('n_branches = 3')
        n_branches = 3
    elif network_settings['network'] == 'logistic_regression':
        print('n_branches = 2')
        n_branches = 2
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet", "hypercolumn", \
                        "siamese_unet_diff", "triplet_apn", "siamese_unet",\
                        "siamese_concat", "logistic_regression"')
    return n_branches, pos_weight

def evaluate_features(model_settings, directories, dataset_settings, network_settings, train_settings):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    # get_network
    n_branches, pos_weight = determine_branches(network_settings, dataset_settings)
    net = get_network(model_settings, train_settings['gpu']) 
    net.eval()
    # get dataset
    dataset_eval = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_eval'], 
        channels=dataset_settings['channels'], 
        one_hot=False, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings,
        directories=directories)
         
    idxs = list()
    for im in dataset_eval.images.keys():
        idxs.append(im.split('_')[0])
    
    idxs = np.unique(idxs)

    for idx in idxs:
        im_a = np.moveaxis(dataset_eval.images[idx+'_a.npy'],-1,0)
        im_b = np.moveaxis(dataset_eval.images[idx+'_b.npy'],-1,0)
        rmse_raw = np.sqrt(np.mean(np.square(im_a-im_b), axis=(1,2)))

        im_a = torch.as_tensor(np.expand_dims(im_a, axis=0))
        im_b = torch.as_tensor(np.expand_dims(im_b, axis=0))
        data = [im_a.float(), im_b.float()]
       
        # get features
        with torch.no_grad():
            features = net(
                data, 
                n_branches=n_branches, 
                extract_features=network_settings['extract_features'])       
        
        features = features.squeeze().detach().numpy()
        rmse = np.sqrt(np.mean(np.square(features), axis=(1,2)))
        fig = plt.figure(figsize=(100,100))
        n_rows = int(np.ceil(np.sqrt(features.shape[0])))+1
        n_cols = int(np.ceil(np.sqrt(features.shape[0])))
        gs = GridSpec(n_rows, n_cols)
        for i, a_map in enumerate(features):
                  
            ax = fig.add_subplot(gs[i+n_cols])
            ax.imshow(a_map)
            ax.axis('off')
        plt.show()

def get_network(model_settings, gpu):
    
    # cfgs are saved as strings, cast back to list
    branch = model_settings['cfg_branch'].split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    branch = branch.replace(" ", "")
    branch = branch.split(",")
    top = model_settings['cfg_top'].split("[" )[1]
    top = top.split("]" )[0]
    top = top.replace("'", "")
    top = top.replace(" ", "")
    top = top.split(",")
    classifier = model_settings['cfg_classifier'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    # save in model_settigns
    model_settings['cfg'] = {'branch': np.array(branch, dtype='object'), 
                             'top': np.array(top,dtype='object'),
                             'classifier': np.array(classifier, dtype='object')}
    
    # batch_norm saved as string cast back to bool
    if model_settings['batch_norm'] == 'False' : 
        model_settings['batch_norm'] = False
    elif model_settings['batch_norm'] == 'True' : 
        model_settings['batch_norm'] = True
          
    # build network
    model_settings['network'] = model_settings['networkname']
   
    extra_settings = {'min_overlap': 1,
                      'max_overlap': 1}
    n_branches, pos_weight = determine_branches(model_settings, extra_settings)
        
    net = NetBuilder.build_network(
        net=model_settings['network'],
        cfg=model_settings['cfg'],
        n_channels=int(model_settings['n_channels']), 
        n_classes=int(model_settings['n_classes']),
        patch_size=int(model_settings['patch_size']),
        im_size=(96,96),
        batch_norm=model_settings['batch_norm'],
        n_branches=n_branches,
        weights=model_settings['filename'],
        gpu=gpu)  
        
    return net

def get_downstream_network(model_settings, gpu, n_branches):
    
    # batch_norm saved as string cast back to bool
    if model_settings['batch_norm'] == 'False' : 
        model_settings['batch_norm'] = False
    elif model_settings['batch_norm'] == 'True' : 
        model_settings['batch_norm'] = True
    
    # cfgs are saved as strings, cast back to list
    branch = model_settings['cfg_branch'].split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    branch = branch.replace(" ", "")
    branch = branch.split(",")
    if model_settings['batch_norm']:
        layers_branch = int(model_settings['layers_branches'] / 3)
    else: 
        layers_branch = int(model_settings['layers_branches'] / 2)
    branch = branch[:layers_branch]   
    classifier = model_settings['cfg_classifier_cd'].split("[" )[1]
    classifier = classifier.split("]" )[0]
    classifier = classifier.replace("'", "")
    classifier = classifier.replace(" ", "")
    classifier = classifier.split(",")
    top = model_settings['cfg_top'].split("[" )[1]
    top = top.split("]" )[0]
    top = top.replace("'", "")
    top = top.replace(" ", "")
    top = top.split(",")
    if model_settings['batch_norm']:
        layers_joint= int(model_settings['layers_joint'] / 3)
    else: 
        layers_joint = int(model_settings['layers_joint'] / 2)
    if layers_branch == 0:
        model_settings['cfg'] = {'branch': None, 
                             'top': None,
                             'classifier': np.array(classifier, dtype='object')}
    elif layers_joint == 0:
        model_settings['cfg'] = {'branch': np.array(branch, dtype='object'), 
                             'top': None,
                             'classifier': np.array(classifier, dtype='object')}
    else:
        top = top[:layers_joint]       
        model_settings['cfg'] = {'branch': np.array(branch, dtype='object'), 
                             'top': np.array(top,dtype='object'),
                             'classifier': np.array(classifier, dtype='object')}
   
    # build network
    model_settings['network'] = model_settings['networkname']
   
    n_branches = 2
        
    net = NetBuilder.build_network(
        net=model_settings['network'],
        cfg=model_settings['cfg'],
        n_channels=int(model_settings['n_channels']), 
        n_classes=int(model_settings['n_classes']),
        patch_size=int(model_settings['patch_size']),
        im_size=(96,96),
        batch_norm=model_settings['batch_norm'],
        n_branches=n_branches,
        weights=model_settings['filename'],
        gpu=gpu)  
        
    conv_classifier = True if model_settings['cfg']['classifier'][0] == 'C' else False
    return net, conv_classifier