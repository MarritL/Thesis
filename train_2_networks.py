#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:21:16 2020

@author: M. Leenstra
"""

#from tensorboardX import SummaryWriter
from models.netbuilder import NetBuilder, create_loss_function, create_optimizer
from data_generator import PairDataset, TripletDataset, TripletDatasetPreSaved
from data_generator import PartlyOverlapDataset, TripletAPNDataset, PairHardNegDataset
from data_generator import TripletAPNHardNegDataset, ShuffleBandDataset, TripletFromFileDataset
from data_generator import TripletAPNFinetuneDataset, PairDatasetOverlap, SupervisedDataset
from OSCDDataset import OSCDDataset, RandomFlip, RandomRot
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
#torch.manual_seed(211)

def train(directories, dataset_settings, network_settings, train_settings):

    # init 
    network_name = network_settings['network']
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    if network_settings['cfg2'] != None:
        outputtime2 = 'N2_{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    #tb_dir = os.path.join(directories['tb_dir'], 'network-{}_date-{}'
    #                      .format(network_name, outputtime))
    #writer = SummaryWriter(logdir=tb_dir)
    #print("tensorboard --logdir {}".format(tb_dir))
    
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
    
    if network_settings['cfg2'] != None:
        network2 = NetBuilder.build_network(
            net=network_settings['network'],
            cfg=network_settings['cfg2'],
            n_channels=len(dataset_settings['channels']), 
            n_classes=network_settings['n_classes'],
            patch_size=network_settings['patch_size'],
            im_size=network_settings['im_size'],
            batch_norm=network_settings['batch_norm'],
            n_branches=n_branches,
            weights=network_settings['weights_file'],
            gpu=train_settings['gpu'])     
        conv_classifier2 = True if network_settings['cfg']['classifier'][0] == 'C' else False
    
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
    
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
        loss_func = loss_func.cuda()
        
        if network_settings['cfg2'] != None:
            network2.cuda() 

    optim = create_optimizer(network_settings['optimizer'], network.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])
    
    if network_settings['cfg2'] != None:
        optim2 = create_optimizer(network_settings['optimizer'], network2.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])
    
    # Datasets
    dataset_train = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_train'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings)
    dataset_val = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_val'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings)

    # Data loaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=train_settings['batch_size'], 
        shuffle=True,
        num_workers = 16)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 8)
    
    if dataset_settings['dataset_type'] == 'supervised':
        pos_weight = dataset_train.pos_weight
        loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
     
    # save history
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    if network_settings['cfg2'] != None:
        history2 = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    
    epoch_iters =  max(len(dataset_train) // train_settings['batch_size'],1)
    val_epoch_iters = max(len(dataset_val) // train_settings['batch_size'],1)
    
    best_net_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = 99999
    if network_settings['cfg2'] != None:
        best_net_wts2 = copy.deepcopy(network2.state_dict())
        best_acc2 = 0.0
        best_epoch2 = 0
        best_loss2 = 99999
    
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
            #if (epoch - best_epoch) > train_settings['early_stopping']:
            #    break
                                   
            if network_settings['cfg2'] != None:
                #training epoch
                train_func(
                    network_settings,
                    network=network, 
                    network2=network2,
                    n_branches=n_branches,
                    dataloader=dataloader_train, 
                    optimizer=optim, 
                    optimizer2=optim2, 
                    loss_func=loss_func,
                    acc_func=acc_func,
                    history=history, 
                    history2=history2, 
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
                    outputtime2=outputtime2,
                    conv_classifier=conv_classifier,
                    conv_classifier2=conv_classifier2)
                
                # validation epoch
                best_net_wts, best_acc, best_epoch, best_loss, best_net_wts2, best_acc2, best_epoch2, best_loss2 = val_func(
                    network_settings,
                    network=network,
                    network2=network2, 
                    n_branches=n_branches,
                    dataloader=dataloader_val, 
                    loss_func=loss_func,
                    acc_func=acc_func,
                    history=history,
                    history2=history2, 
                    epoch=epoch, 
                    writer=None,
                    val_epoch_iters=val_epoch_iters,
                    best_net_wts=best_net_wts,
                    best_acc=best_acc,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    best_net_wts2=best_net_wts2,
                    best_acc2=best_acc2,
                    best_epoch2=best_epoch2,
                    best_loss2=best_loss2,
                    gpu = train_settings['gpu'],
                    im_size = network_settings['im_size'],
                    extract_features=network_settings['extract_features'],
                    avg_pool=network_settings['avg_pool'],
                    directories=directories, 
                    network_name=network_name, 
                    outputtime=outputtime,
                    outputtime2=outputtime2,
                    conv_classifier=conv_classifier,
                    conv_classifier2=conv_classifier2)
            else:
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
                    conv_classifier=conv_classifier)
                
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
                    conv_classifier=conv_classifier)
    
    # on keyboard interupt continue the script: saves the best model until interrupt
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving progress...")
        
    
    # TODO: write info to csv-file
    savedata = {'filename':os.path.join(directories['model_dir'],
                        'network-{}_date-{}'.format(network_name, outputtime)), 
               'networkname': network_name, 
               'cfg_branch': str(list(network_settings['cfg']['branch'])), 
               'cfg_top': str(list(network_settings['cfg']['top'])),
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
    
    if network_settings['cfg2'] != None:
        savedata2 = {'filename':os.path.join(directories['model_dir'],
                    'network-{}_date-{}'.format(network_name, outputtime2)), 
                   'networkname': network_name, 
                   'cfg_branch': str(list(network_settings['cfg2']['branch'])), 
                   'cfg_top': str(list(network_settings['cfg2']['top'])),
                   'cfg_classifier': str(list(network_settings['cfg2']['classifier'])),
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
                   'best_acc': best_acc2,
                   'best_epoch': best_epoch2, 
                   'best_loss': best_loss2,
                   'weight': None,
                   'global_avg_pool': 'False', 
                   'basis': None,
                   'n_train_patches': len(dataset_settings['indices_train']), 
                   'patches_per_image': dataset_settings['patches_per_image'],
                   'n_val_patches': len(dataset_settings['indices_val'])}
        with open(os.path.join(directories['intermediate_dir'], 
                               directories['csv_models']), 'a') as file:
                filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
                filewriter.writerow(savedata2)  
        
        # save best model's weights
        torch.save(best_net_wts2, savedata2['filename'])
        torch.save(history2, os.path.join(directories['model_dir'],
                            'history_network-{}_date-{}'.format(network_name, outputtime2)))
            
    print('Training Done!')
    #writer.close()
    

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
        network_settings=network_settings)
                        
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
        conv_classifier=conv_classifier)

    return best_acc, best_loss, best_prob

def finetune(model_settings, directories, dataset_settings, network_settings, train_settings):

    # init 
    network_name = model_settings['networkname']+'_finetune'
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    
    # init save-file
    fieldnames = ['filename','pretask_filename','networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
                  'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
                  'n_channels','patch_size','batch_norm','dataset','min_overlap',\
                  'max_overlap', 'best_acc','best_epoch', 'best_loss', 'weight', \
                  'global_avg_pool', 'n_train_patches', 'patches_per_image', \
                  'n_val_patches', 'kthfold', 'n_eval_patches', 'eval_acc', 'eval_loss', 'eval_prob',\
                  'tp_oscd_eval_triangle', 'tn_oscd_eval_triangle', 'fp_oscd_eval_triangle',\
                  'fn_oscd_eval_triangle', 'f1_oscd_eval_triangle']
    if not os.path.exists(os.path.join(directories['intermediate_dir'],directories['csv_models_finetune'])):
        with open(os.path.join(directories['intermediate_dir'],directories['csv_models_finetune']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()
           
    # get network
    network_settings['im_size'] = (1,1)
    n_branches, pos_weight = determine_branches(network_settings, dataset_settings)
    network = get_network(model_settings, train_settings['gpu'])  
    # freeze layers
    for param in network.parameters():
        param.requires_grad = False
    
    # add new classifier
    in_channels = network.joint[0].out_channels
    cfg = np.array(['C', network_settings['n_classes']])
    classifier = make_conv_classifier(cfg[1:], in_channels, batch_norm=True)
    network.classifier = classifier
    print(network)
    conv_classifier = True 
    
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
    
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        network.cuda()
        loss_func = loss_func.cuda()

    optim = create_optimizer(network_settings['optimizer'], network.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])
    
    # Datasets
    dataset_train = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_train'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings)
    dataset_val = get_dataset(
        data_path=directories['data_path'], 
        indices=dataset_settings['indices_val'], 
        channels=dataset_settings['channels'], 
        one_hot=one_hot, 
        dataset_settings=dataset_settings, 
        network_settings=network_settings)

    # Data loaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=train_settings['batch_size'], 
        shuffle=True,
        num_workers = 8)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 4)
    
    if dataset_settings['dataset_type'] == 'supervised':
        pos_weight = dataset_train.pos_weight
        loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
     
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
                conv_classifier=conv_classifier)
            
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
                conv_classifier=conv_classifier)
    
    # on keyboard interupt continue the script: saves the best model until interrupt
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving progress...")
        
    
    # TODO: write info to csv-file
    savedata = {'filename':os.path.join(directories['model_dir'],
                        'network-{}_date-{}'.format(network_name, outputtime)), 
               'pretask_filename': model_settings['filename'],
               'networkname': network_name, 
               'cfg_branch': str(list(network_settings['cfg']['branch'])), 
               'cfg_top': str(list(network_settings['cfg']['top'])),
               'cfg_classifier': str(list(cfg)),
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
               'n_val_patches': len(dataset_settings['indices_val']),
               'kthfold': dataset_settings['kthfold']}
    with open(os.path.join(directories['intermediate_dir'], 
                           directories['csv_models_finetune']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",", extrasaction='ignore')
            filewriter.writerow(savedata)  
    
    # save best model's weights
    torch.save(best_net_wts, savedata['filename'])
    torch.save(history, os.path.join(directories['model_dir'],
                        'history_network-{}_date-{}'.format(network_name, outputtime)))
        
    print('Finetune Done!')
    #writer.close()
    
def train_epoch(network_settings, network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, 
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False, conv_classifier=False,
                network2=None, optimizer2=None, history2=None, outputtime2=None,
                conv_classifier2=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_loss_all = AverageMeter()
    ave_acc = AverageMeter()
    
    network.train() 
    if network2 != None:
        batch_time2 = AverageMeter()
        ave_loss2 = AverageMeter()
        ave_loss_all2 = AverageMeter()
        ave_acc2 = AverageMeter()
        network2.train()
    
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
        if optimizer2 != None:
            optimizer2.zero_grad()
        
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
        #print(starts)
        im_idx = batch_data['im_idx'].detach().numpy()
    
        with open(os.path.join(directories['model_dir'],
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
    

#       # NETWORK 2
        # forward pass 
        if network2 != None:
            outputs2 = network2(inputs, n_branches, extract_features=extract_features, 
                              avg_pool=avg_pool,conv_classifier=conv_classifier2, 
                              use_softmax=False)
    
            if network_settings['loss'] == 'bce_sigmoid+l1reg':  
                loss2 = loss_func(outputs2, labels, network2.parameters())
            else:
                loss2 = loss_func(outputs2, labels)
            acc2 = acc_func(outputs2, labels, im_size)
    
            # Backward
            loss2.backward()
            optimizer2.step()
    
            # measure elapsed time
            batch_time2.update(time.time() - tic)
            tic = time.time()
    
            # update average loss and acc
            ave_loss2.update(loss2.data.item())
            ave_acc2.update(acc2.item())
                   
    
            # calculate accuracy, and display
            if (i+1) % disp_iter == 0:
                print('Epoch N2: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                      'Loss: {:.4f}, Acc: {:.4f}'
                      .format(epoch, i+1, epoch_iters,
                              batch_time2.average(), data_time.average(),
                              ave_loss2.average(), ave_acc2.average()))
                ave_loss_all2.update(ave_loss2.average())
                ave_loss2 = AverageMeter()


    if network2 !=None:
        if (i+1) < disp_iter:
            ave_loss_all_print = ave_loss.average()
            ave_loss_all_print2 = ave_loss2.average()
        else:
            ave_loss_all_print = ave_loss_all.average()
            ave_loss_all_print2 = ave_loss_all2.average()
        print('Train epoch: [{}], Time: {:.2f} ' 
          'Train_Loss: {:.4f}, Train_Accuracy: {:0.4f}, '
          'Train_Loss_N2: {:.4f}, Train_Accuracy_N2: {:0.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all_print, ave_acc.average(),
                  ave_loss_all_print2, ave_acc2.average()))
    else:
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
    
    if history2 != None:
        history2['train']['epoch'].append(epoch)
        history2['train']['loss'].append(ave_loss_all2.average())
        history2['train']['acc'].append(ave_acc2.average())


    
def validate_epoch(network_settings, network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, 
             extract_features=None, avg_pool=False, inference=False, conv_classifier=False,
             network2=None, history2=None, best_net_wts2=None, best_acc2=None, 
             best_epoch2=None, best_loss2=None, outputtime2=None, conv_classifier2=None):    

    time_meter = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    
    use_softmax = False
    if inference: 
        probability = AverageMeter()
        sigmoid = torch.nn.Sigmoid()
        #use_softmax = True
        
    network.eval() 
    if network2 != None:
        time_meter2 = AverageMeter()
        ave_loss2 = AverageMeter()
        ave_acc2 = AverageMeter()
        probability2 = AverageMeter()
        network2.eval()
    
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
        
#        ## NETWORK 2
        if network2 != None:
            # forward pass 
            with torch.no_grad():
                outputs2 = network2(inputs, n_branches, extract_features=extract_features, 
                              avg_pool=avg_pool,conv_classifier=conv_classifier2, 
                              use_softmax=use_softmax)
    
            if network_settings['loss'] == 'bce_sigmoid+l1reg':  
                loss2 = loss_func(outputs2, labels, network2.parameters())
            else:
                loss2 = loss_func(outputs2, labels)
            acc2 = acc_func(outputs2, labels, im_size)    
            
            if inference:
                prob2 = sigmoid(outputs)
                correct2 = labels*prob2
                prob_correct2 = torch.sum(correct2)/len(outputs2)       
                probability2.update(prob_correct2.item())
    
            # update average loss and acc
            ave_loss2.update(loss2.data.item())
            ave_acc2.update(acc2.item())
    
            # measure elapsed time
            time_meter2.update(time.time() - tic)
            tic = time.time()

    
    if network2 != None:
        print('Val epoch: [{}], Time: {:.2f}, ' 
          'Val_Loss: {:.4f}, Val_Accuracy: {:0.4f}, '
          'Val_Loss_N2: {:.4f}, Val_Accuracy_N2: {:0.4f}'
          .format(epoch, time_meter.value(),
                  ave_loss.average(), ave_acc.average(),
                  ave_loss2.average(), ave_acc2.average()))
        if inference:
            best_acc = ave_acc.average()
            best_loss = ave_loss.average()
            best_prob = probability.average()
            best_acc2 = ave_acc2.average()
            best_loss2 = ave_loss2.average()
            best_prob2 = probability2.average()
            return best_acc, best_loss, best_prob, best_acc2, best_loss2, best_prob2
        else:
            if writer != None:
                writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
                writer.add_scalar('Val/Acc', ave_acc.average(), epoch)
        
            if ave_loss.average() < best_loss:
                best_acc = ave_acc.average()
                best_net_wts = copy.deepcopy(network.state_dict())
                best_epoch = epoch
                best_loss = ave_loss.average()
                torch.save(best_net_wts, os.path.join(directories['model_dir'],
                                'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                    network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
            if ave_loss2.average() < best_loss2:   
                best_acc2 = ave_acc2.average()
                best_net_wts2 = copy.deepcopy(network2.state_dict())
                best_epoch2 = epoch
                best_loss2 = ave_loss2.average()
                torch.save(best_net_wts2, os.path.join(directories['model_dir'],
                                'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                    network_name, outputtime2, epoch, ave_loss2.average(), ave_acc2.average())))
            
            if history != None:
                history['val']['epoch'].append(epoch)
                history['val']['loss'].append(ave_loss.average())
                history['val']['acc'].append(ave_acc.average())
            if history2 != None:
                history2['val']['epoch'].append(epoch)
                history2['val']['loss'].append(ave_loss2.average())
                history2['val']['acc'].append(ave_acc2.average())
            
            return(best_net_wts, best_acc, best_epoch, best_loss,
                   best_net_wts2, best_acc2, best_epoch2, best_loss2)
        
    else:
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
            if writer != None:
                writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
                writer.add_scalar('Val/Acc', ave_acc.average(), epoch)
        
            if ave_loss.average() < best_loss:
                best_acc = ave_acc.average()
                best_net_wts = copy.deepcopy(network.state_dict())
                best_epoch = epoch
                best_loss = ave_loss.average()
                torch.save(best_net_wts, os.path.join(directories['model_dir'],
                                'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                    network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
            
            if history != None:
                history['val']['epoch'].append(epoch)
                history['val']['loss'].append(ave_loss.average())
                history['val']['acc'].append(ave_acc.average())
            
            return(best_net_wts, best_acc, best_epoch, best_loss)

def train_epoch_apn(network_settings, network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, 
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False):
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
        outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
        if torch.any(torch.isnan(outputs[0])) or torch.any(torch.isnan(outputs[1])):
            print("outputs is nan")
            import ipdb
            ipdb.set_trace()
# =============================================================================
#         # =====================================================================
#         anchor = outputs[0]
#         positive = outputs[1]
#         negative = outputs[2]
# 
#         anchor_mean = anchor.mean((2,3))  
#         positive_mean = positive.mean((2,3))
#         negative_mean = negative.mean((2,3))            
#         
#         out = [output.mean((2,3)) for output in outputs]
#             
#         # =====================================================================    
# =============================================================================
        #loss = loss_func(outputs, labels)
        loss, loss1, loss2 = loss_func(outputs, labels)
        #loss = loss_func(outputs[0], outputs[1], outputs[2])
        #print("Combined: {0:.3f}, L1: {1:.3f}, triplet: {2:.3f}, min: {3:.3f}-{4:.3f}, max: {5:.3f}-{6:.3f}, mean: {7:.3f}-{8:.3f}".format(loss, loss1, loss2, min0, min1, max0, max1, mean0, mean1))
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
    
        with open(os.path.join(directories['model_dir'],
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
# =============================================================================
#             print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
#                   'Loss: {:.4f}, Acc: {:.4f}'
#                   .format(epoch, i+1, epoch_iters,
#                           batch_time.average(), data_time.average(),
#                           ave_loss.average(), ave_acc.average()))
#             ave_loss_all.update(ave_loss.average())
#             ave_loss = AverageMeter()
# =============================================================================

            #print("Combined: {0:.3f}, L1: {1:.3f}, triplet: {2:.3f}".format(loss, loss1, loss2))
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
             best_loss, gpu, im_size, directories, network_name, outputtime, 
             extract_features=None, avg_pool=False, evaluate=False):    

    ave_loss = AverageMeter()
    ave_lossL1 = AverageMeter()
    ave_lossTriplet = AverageMeter()
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

        #labels = batch_data['gt']
        # to GPU
        if gpu != None:
            labels = labels.cuda()

      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
            if evaluate:
                prob = sigmoid(outputs)
                correct = labels*prob
                prob_correct = torch.sum(correct)/len(outputs)       
                probability.update(prob_correct.item())
            #loss = loss_func(outputs, labels)
            loss, loss1, loss2  = loss_func(outputs, labels)
            #loss = loss_func(outputs[0], outputs[1], outputs[2])
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
            torch.save(best_net_wts, os.path.join(directories['model_dir'],
                            'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
        
        if history != None:
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(ave_loss.average())
            history['val']['acc'].append(ave_acc.average())
        
        return(best_net_wts, best_acc, best_epoch, best_loss)

def train_epoch_unet(network_settings, network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, 
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False):
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
        labels_bottle = batch_data['label_bottle']
        #labels = batch_data['gt']
    
        # to gpu
        if gpu != None:
            labels = labels.cuda()
            labels_bottle = labels_bottle.cuda()
            
        # forward pass
        outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
# =============================================================================
#         # =====================================================================
#         anchor = outputs[0]
#         positive = outputs[1]
#         negative = outputs[2]
# 
#         anchor_mean = anchor.mean((2,3))  
#         positive_mean = positive.mean((2,3))
#         negative_mean = negative.mean((2,3))            
#         
#         out = [output.mean((2,3)) for output in outputs]
#             
#         # =====================================================================    
# =============================================================================
        #loss = loss_func(outputs, labels)
        loss, loss1, loss2, loss3 = loss_func(outputs, labels, labels_bottle)
        #loss = loss_func(outputs[0], outputs[1], outputs[2])
        #print("Combined: {0:.3f}, L1: {1:.3f}, triplet: {2:.3f}, min: {3:.3f}-{4:.3f}, max: {5:.3f}-{6:.3f}, mean: {7:.3f}-{8:.3f}".format(loss, loss1, loss2, min0, min1, max0, max1, mean0, mean1))
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
    
        with open(os.path.join(directories['model_dir'],
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
# =============================================================================
#             print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
#                   'Loss: {:.4f}, Acc: {:.4f}'
#                   .format(epoch, i+1, epoch_iters,
#                           batch_time.average(), data_time.average(),
#                           ave_loss.average(), ave_acc.average()))
#             ave_loss_all.update(ave_loss.average())
#             ave_loss = AverageMeter()
# =============================================================================

            #print("Combined: {0:.3f}, L1: {1:.3f}, triplet: {2:.3f}".format(loss, loss1, loss2))
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
             best_loss, gpu, im_size, directories, network_name, outputtime, 
             extract_features=None, avg_pool=False, evaluate=False):    

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

        #labels = batch_data['gt']
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
            #loss = loss_func(outputs, labels)
            loss, loss1, loss2, loss3  = loss_func(outputs, labels, labels_bottle)
            #loss = loss_func(outputs[0], outputs[1], outputs[2])
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
            torch.save(best_net_wts, os.path.join(directories['model_dir'],
                            'network-{}_date-{}_epoch-{}_loss-{}_acc-{}'.format(
                                network_name, outputtime, epoch, ave_loss.average(), ave_acc.average())))
        
        if history != None:
            history['val']['epoch'].append(epoch)
            history['val']['loss'].append(ave_loss.average())
            history['val']['acc'].append(ave_acc.average())
        
        return(best_net_wts, best_acc, best_epoch, best_loss)

def get_dataset(data_path, indices, channels, one_hot, dataset_settings, network_settings):
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
            #batch_size = 5)           # TODO: hardcoded
    elif dataset_settings['dataset_type'] == 'triplet_oscd':
        dataset = OSCDDataset(
            data_dir=data_path,
            indices=indices,
            patch_size=network_settings['patch_size'], 
            stride = dataset_settings['stride'], 
            transform = transforms.Compose([RandomFlip(), RandomRot()]), 
            one_hot=one_hot, 
            alt_label=True, 
            mode='train')
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
    elif dataset_settings['dataset_type'] == 'supervised':
        in_memory = True if len(np.unique(indices)) < 21 else False
        dataset = SupervisedDataset(
            data_dir=data_path,
            indices=indices, 
            labels_dir='/media/cordolo/marrit/Intermediate/CD_OSCD/labels_OSCD', 
            channels=channels,
            one_hot=one_hot,
            in_memory=in_memory)
    else:
        raise Exception('dataset_type undefined! \n \
                        Choose one of: "pair", "triplet", "triplet_saved",\
                        "overlap", "triplet_apn", "triplet_oscd", "overlap_regression", \
                        "pair_hard_neg", "triplet_apn_hard_neg", "triplet_from_file",\
                        "supervised"')
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
        network_settings=network_settings)
         
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