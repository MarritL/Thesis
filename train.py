#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:21:16 2020

@author: M. Leenstra
"""

from tensorboardX import SummaryWriter
from models.netbuilder import NetBuilder, create_loss_function, create_optimizer
from data_generator import PairDataset, TripletDataset, TripletDatasetPreSaved
from data_generator import PartlyOverlapDataset, TripletAPNDataset, PairHardNegDataset
from data_generator import TripletAPNHardNegDataset
from data_generator import TripletAPNFinetuneDataset, PairDatasetOverlap
from OSCDDataset import OSCDDataset, RandomFlip, RandomRot
from models.siameseNetAPNFinetune import siameseNetAPNFinetune
from utils import AverageMeter
import torch
from torch.utils.data import DataLoader
from torchvision import transforms 
import os
import time
import csv
import copy
import numpy as np
from extract_features import get_network

def train(directories, dataset_settings, network_settings, train_settings):

    # init tensorboard
    network_name = network_settings['network']
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    tb_dir = os.path.join(directories['tb_dir'], 'network-{}_date-{}'
                          .format(network_name, outputtime))
    writer = SummaryWriter(logdir=tb_dir)
    print("tensorboard --logdir {}".format(tb_dir))
    
    # init save-file
    fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
                  'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
                  'n_channels','patch_size','batch_norm','dataset','min_overlap',\
                  'max_overlap', 'best_acc','best_epoch', 'best_loss', 'weight', \
                  'global_avg_pool', 'basis', 'n_train_patches', 'patches_per_image', \
                  'n_val_patches']
    if not os.path.exists(os.path.join(directories['intermediate_dir'],directories['csv_models'])):
        with open(os.path.join(directories['intermediate_dir'],directories['csv_models']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()
            
    # init save-file for patch starts during training
    fieldnames_trainpatches = ['epoch', 'im_idx', 'patch_starts']
    if not os.path.exists(os.path.join(directories['model_dir'],
                        'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime))):
        with open(os.path.join(directories['model_dir'],
                        'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime)), 'a') as file2:
            filewriter = csv.DictWriter(file2, fieldnames_trainpatches, delimiter = ",")
            filewriter.writeheader()
                
    # build network
    pos_weight = 1
    network_settings['im_size'] = (1,1)
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
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet", "hypercolumn", \
                        "siamese_unet_diff", "triplet_apn", "siamese_unet",\
                        "siamese_concat"')
                
    net = NetBuilder.build_network(
        net=network_settings['network'],
        cfg=network_settings['cfg'],
        n_channels=len(dataset_settings['channels']), 
        n_classes=network_settings['n_classes'],
        patch_size=network_settings['patch_size'],
        im_size=network_settings['im_size'],
        batch_norm=network_settings['batch_norm'],
        n_branches=n_branches,
        weights=network_settings['weights_file'])  
    #net = net.float()
       
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)
    
    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        net.cuda()
        loss_func = loss_func.cuda()

    optim = create_optimizer(network_settings['optimizer'], net.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])

    # Datasets
    if dataset_settings['dataset_type'] == 'pair':
        dataset_train = PairDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'])      
        dataset_val = PairDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'])       
    elif dataset_settings['dataset_type'] == 'triplet':
        in_memory = True if len(np.unique(dataset_settings['indices_train'])) < 21 else False
        dataset_train = TripletDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'],
            in_memory = in_memory)     
        dataset_val = TripletDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'],
            in_memory = in_memory)
    elif dataset_settings['dataset_type'] == 'triplet_saved':
        print("construct from presaved dataset")
        dataset_train = TripletDatasetPreSaved(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot)           
        dataset_val = TripletDatasetPreSaved(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot)  
    elif dataset_settings['dataset_type'] == 'overlap':
        dataset_train = PartlyOverlapDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot)           
        dataset_val = PartlyOverlapDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot) 
    elif dataset_settings['dataset_type'] == 'triplet_apn':
        second_label = True if network_settings['loss'] == 'l1+triplet+bce' else False
        in_memory = True if len(np.unique(dataset_settings['indices_train'])) < 21 else False
        dataset_train = TripletAPNDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            second_label=second_label,
            in_memory=in_memory)#,
            #batch_size = 5)           # TODO: hardcoded
        dataset_val = TripletAPNDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            second_label=second_label,
            in_memory=in_memory)#,
            #batch_size = 5)  # TODO: hardcoded
    elif dataset_settings['dataset_type'] == 'triplet_oscd':
        dataset_train = OSCDDataset(
            data_dir=directories['data_path'], 
            indices = dataset_settings['indices_train'], 
            patch_size=network_settings['patch_size'], 
            stride = dataset_settings['stride'], 
            transform = transforms.Compose([RandomFlip(), RandomRot()]), 
            one_hot=one_hot, 
            alt_label=True, 
            mode='train')
        dataset_val = OSCDDataset(
            data_dir=directories['data_path'], 
            indices = dataset_settings['indices_val'], 
            patch_size=network_settings['patch_size'], 
            stride = dataset_settings['stride'], 
            transform = transforms.Compose([RandomFlip(), RandomRot()]), 
            one_hot=one_hot, 
            alt_label=True, 
            mode='train')
    elif dataset_settings['dataset_type'] == 'overlap_regression':
        in_memory = True if len(np.unique(dataset_settings['indices_train'])) < 21 else False
        dataset_train = PairDatasetOverlap(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=False,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'],
            in_memory=in_memory)          
        dataset_val = PairDatasetOverlap(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=False,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'],
            in_memory=in_memory)  
    elif dataset_settings['dataset_type'] == 'pair_hard_neg':
        in_memory = True if len(np.unique(dataset_settings['indices_train'])) < 21 else False
        dataset_train = PairHardNegDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            in_memory = in_memory)
        dataset_val = PairHardNegDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            in_memory = in_memory)
    elif dataset_settings['dataset_type'] == 'triplet_apn_hard_neg':
        second_label = True if network_settings['loss'] == 'l1+triplet+bce' else False
        in_memory = True if len(np.unique(dataset_settings['indices_train'])) < 21 else False
        dataset_train = TripletAPNHardNegDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            second_label=second_label,
            in_memory=in_memory)
        dataset_val = TripletAPNHardNegDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            second_label=second_label,
            in_memory=in_memory)
    else:
        raise Exception('dataset_type undefined! \n \
                        Choose one of: "pair", "triplet", "triplet_saved",\
                        "overlap", "triplet_apn", "triplet_oscd", "overlap_regression" \
                        , "pair_hard_neg", "triplet_apn_hard_neg"')
    
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
         
    # save history
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    
    epoch_iters =  max(len(dataset_train) // train_settings['batch_size'],1)
    val_epoch_iters = max(len(dataset_val) // train_settings['batch_size'],1)
    
    best_net_wts = copy.deepcopy(net.state_dict())
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
            
            #training epoch
            train_func(
                network=net, 
                n_branches=n_branches,
                dataloader=dataloader_train, 
                optimizer=optim, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=history, 
                epoch=epoch, 
                writer=writer,
                epoch_iters=epoch_iters, 
                disp_iter=train_settings['disp_iter'],
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'],
                directories=directories, 
                network_name=network_name, 
                outputtime=outputtime)
            
            # validation epoch
            best_net_wts, best_acc, best_epoch, best_loss = val_func(
                network=net, 
                n_branches=n_branches,
                dataloader=dataloader_val, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=history, 
                epoch=epoch, 
                writer=writer,
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
                outputtime=outputtime)
    
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
        
    print('Training Done!')
    writer.close()

def validate(model_settings, eval_settings):
    # build network
    model_settings['network'] = model_settings['networkname']
    if model_settings['networkname'] == 'siamese':
        n_branches = 2
        pos_weight = 1
        model_settings['im_size'] = (1,1)
    elif model_settings['networkname'] == 'triplet':
        print('n_branches = 3')
        n_branches = 3
        pos_weight=1
        model_settings['network'] = 'siamese'
        model_settings['im_size'] = (1,1)        
    elif model_settings['networkname'] == 'hypercolumn':
        print('n_branches = 2')
        n_branches = 2
        mean_overlap = np.mean([model_settings['min_overlap'], 
                                model_settings['max_overlap']])
        pos_weight = (1-mean_overlap)/mean_overlap
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet", "hypercolumn", \
                        "siamese_unet_diff"')
        
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

    
    net = NetBuilder.build_network(
        net=model_settings['network'],
        cfg=model_settings['cfg'],
        n_channels=model_settings['n_channels'], 
        n_classes=model_settings['n_classes'],
        patch_size=model_settings['patch_size'],
        im_size=model_settings['im_size'],
        batch_norm=model_settings['batch_norm'],
        n_branches=n_branches,
        weights=model_settings['filename'])  
       
    loss_func, acc_func, one_hot = create_loss_function(model_settings['loss'], pos_weight=pos_weight)
    
    # load net to GPU     
    if eval_settings['gpu'] != None:
        torch.cuda.set_device(eval_settings['gpu'])
        net.cuda()
        loss_func = loss_func.cuda()
                     
    # Datasets
    if model_settings['dataset'] == 'pair':
        dataset_val = PairDataset(
            data_dir=eval_settings['data_path'], 
            indices=eval_settings['indices_eval'], 
            channels=np.arange(13))       
    elif model_settings['dataset'] == 'triplet' or model_settings['dataset'] == 'triplet_saved': 
        dataset_val = TripletDataset(
            data_dir=eval_settings['data_path'], 
            indices=eval_settings['indices_eval'], 
            channels=np.arange(13), 
            one_hot=one_hot,
            min_overlap = model_settings['min_overlap'],
            max_overlap = model_settings['max_overlap'])  
    else:
        raise Exception('dataset_type undefined! \n \
                        Choose one of: "pair", "triplet", "triplet_saved"')
    
    # Data loaders
    dataloader = DataLoader(
        dataset_val, 
        batch_size=eval_settings['batch_size'], 
        shuffle=False,
        num_workers = 1)
    
    val_epoch_iters = max(len(dataset_val) // eval_settings['batch_size'],1)
    best_net_wts = copy.deepcopy(net.state_dict())
    
    # validation epoch
    best_net_wts, best_acc, best_epoch, best_loss = validate_epoch(
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
        gpu = eval_settings['gpu'],
        im_size = (1,1),
        extract_features=None)

    return best_acc, best_loss

def finetune(model_settings, directories, dataset_settings, network_settings, train_settings):
    # init tensorboard
    network_name = model_settings['networkname']+'_finetune'
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    tb_dir = os.path.join(directories['tb_dir'], 'network-{}_date-{}'
                          .format(network_name, outputtime))
    writer = SummaryWriter(logdir=tb_dir)
    print("tensorboard --logdir {}".format(tb_dir))
    
    # init save-file
    fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', 'cfg_classifier', \
                  'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
                  'n_channels','patch_size','batch_norm','dataset','min_overlap',\
                  'max_overlap', 'best_acc','best_epoch', 'best_loss']
    
    if not os.path.isdir(directories['results_dir_cd']):
        os.makedirs(directories['results_dir_cd'])     
    if not os.path.exists(os.path.join(directories['results_dir_cd'], 
                                       directories['csv_models_results'])):
        with open(os.path.join(directories['results_dir_cd'], 
                               directories['csv_models_results']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writeheader()
   
    # find size last layer branch
    branch = model_settings['cfg_branch'].split("[" )[1]
    branch = branch.split("]" )[0]
    branch = branch.replace("'", "")
    branch = branch.replace(" ", "")
    branch = branch.split(",")
    branch = np.array(branch)
    last_in = int(branch[-1])
    
    # get network
    net = get_network(model_settings)  
    finetune_net = siameseNetAPNFinetune(
        net, 
        last_in=last_in,
        n_classes=network_settings['n_classes'])
    print(finetune_net)
      
    one_hot = False
    
    # datasets
    dataset_train = TripletAPNFinetuneDataset(
        data_dir = os.path.join(directories['intermediate_dir_cd'],directories['data_dir_oscd']), 
        data_dir_gt = os.path.join(directories['intermediate_dir_cd'],directories['labels_dir_oscd']), 
        indices = dataset_settings['indices_train'], 
        channels = dataset_settings['channels'],
        one_hot = one_hot,
        batch_size = train_settings['batch_size'])
    dataset_val = TripletAPNFinetuneDataset(
        data_dir = os.path.join(directories['intermediate_dir_cd'],directories['data_dir_oscd']), 
        data_dir_gt = os.path.join(directories['intermediate_dir_cd'],directories['labels_dir_oscd']),  
        indices = dataset_settings['indices_train'], 
        channels = dataset_settings['channels'],
        one_hot = one_hot,
        batch_size = train_settings['batch_size'])
    
    # dataloaders
    dataloader_train = DataLoader(
        dataset = dataset_train, 
        batch_size =1, 
        shuffle = True)
    dataloader_val = DataLoader(
        dataset = dataset_val, 
        batch_size = 1, 
        shuffle = False)
    
    iterator = iter(dataloader_train)
    for i, batch_data in enumerate(iterator): 
        print("\r get dataset weights... {}/{}".format(i+1,len(dataset_train), end=''))
        
    n_branches = 2
    pos_weight = dataset_train.weight / (len(dataset_train) * \
                                         train_settings['batch_size'] * \
                                         network_settings['patch_size']**2)
    
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'], pos_weight=pos_weight)

    # load net to GPU     
    if train_settings['gpu'] != None:
        torch.cuda.set_device(train_settings['gpu'])
        net.cuda()
        loss_func = loss_func.cuda()
    
    optim = create_optimizer(network_settings['optimizer'], net.parameters(), 
                             network_settings['lr'], 
                             weight_decay=network_settings['weight_decay'])
        
    epoch_iters =  max(len(dataset_train) // train_settings['batch_size'],1)
    val_epoch_iters = max(len(dataset_val) // train_settings['batch_size'],1)
    
    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = 99999
    
    try:
        for epoch in range(train_settings['start_epoch'], 
                           train_settings['start_epoch']+train_settings['num_epoch']):
            
            #training epoch
            train_epoch(
                network=finetune_net, 
                n_branches=n_branches,
                dataloader=dataloader_train, 
                optimizer=optim, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=None, 
                epoch=epoch, 
                writer=writer,
                epoch_iters=epoch_iters, 
                disp_iter=train_settings['disp_iter'],
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'],
                directories=directories, 
                network_name=network_name, 
                outputtime=outputtime)
            
            # validation epoch
            best_net_wts, best_acc, best_epoch, best_loss = validate_epoch(
                network=finetune_net, 
                n_branches=n_branches,
                dataloader=dataloader_val, 
                loss_func=loss_func,
                acc_func=acc_func,
                history=None, 
                epoch=epoch, 
                writer=writer,
                val_epoch_iters=val_epoch_iters,
                best_net_wts=best_net_wts,
                best_acc=best_acc,
                best_epoch=best_epoch,
                best_loss=best_loss,
                gpu = train_settings['gpu'],
                im_size = network_settings['im_size'],
                extract_features=network_settings['extract_features'],
                avg_pool=network_settings['avg_pool'])
    
    # on keyboard interupt continue the script: saves the best model until interrupt
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Saving progress...")
         
     
    # TODO: write info to csv-file
    savedata = {'filename':os.path.join(directories['model_dir'],
                        'network-{}_date-{}'.format(network_name, outputtime)), 
               'networkname': network_name, 
               'cfg_branch': model_settings['cfg']['branch'], 
               'cfg_top': None,
               'cfg_classifier': 'supervised',
               'optimizer': network_settings['optimizer'],
               'lr': network_settings['lr'], 
               'weight_decay': network_settings['weight_decay'],
               'loss': network_settings['loss'],
               'n_classes': network_settings['n_classes'],
               'n_channels' : len(dataset_settings['channels']),
               'patch_size': network_settings['patch_size'],
               'batch_norm': model_settings['batch_norm'],
               'dataset': dataset_settings['dataset_type'], 
               'min_overlap': None,
               'max_overlap': None,
               'best_acc': best_acc,
               'best_epoch': best_epoch, 
               'best_loss': best_loss,
               'weight': None,
               'global_avg_pool': 'False', 
               'basis': None,
               'n_train_patches': len(dataset_settings['indices_train']), 
               'patches_per_image': 5,
               'n_val_patches': len(dataset_settings['indices_val'])}
    with open(os.path.join(directories['intermediate_dir'], 
                           directories['csv_models']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writerow(savedata)  
    
    # save best model's weights
    torch.save(best_net_wts, savedata['filename'])
        
    print('Training Done!')
    writer.close()
    
    
def train_epoch(network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, directories, network_name, outputtime, 
                fieldnames_trainpatches=['epoch', 'im_idx', 'patch_starts'],
                extract_features=None, avg_pool=False):
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
# =============================================================================
#         if torch.any(torch.isnan(outputs[0])) or torch.any(torch.isnan(outputs[1])):
#             print("outputs is nan")
#             import ipdb
#             ipdb.set_trace()
# # =============================================================================
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
        loss = loss_func(outputs, labels)
        #loss, loss1, loss2 = loss_func(outputs, labels)
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
        #ave_lossL1.update(loss1.data.item())
        #ave_lossTriplet.update(loss2.data.item())
        ave_acc.update(acc.item())
        
        # save the starting points in csvfile for reproducability
        starts = batch_data['patch_starts'].detach().numpy()
        im_idx = batch_data['im_idx'].detach().numpy()
    
        with open(os.path.join(directories['model_dir'],
                'network-{}_date-{}_trainpatches.csv'.format(network_name, outputtime)), 'a') as file2:
            filewriter = csv.DictWriter(file2, fieldnames_trainpatches, delimiter = ",", extrasaction='ignore')
            for j in range(len(im_idx)):
                filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],'patch_starts':starts[j]})  

        

        # calculate accuracy, and display
        if (i+1) % disp_iter == 0:
            print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                  'Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch, i+1, epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_loss.average(), ave_acc.average()))
            ave_loss_all.update(ave_loss.average())
            ave_loss = AverageMeter()

            #print("Combined: {0:.3f}, L1: {1:.3f}, triplet: {2:.3f}".format(loss, loss1, loss2))
# =============================================================================
#             print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
#                   'Loss: {:.4f}, Acc: {:.4f}, L1: {:.4f}, LTriplet: {:.4f}'
#                   .format(epoch, i+1, epoch_iters,
#                           batch_time.average(), data_time.average(),
#                           ave_loss.average(), ave_acc.average(), 
#                           ave_lossL1.average(), ave_lossTriplet.average()))
#             ave_loss_all.update(ave_loss.average())
#             ave_loss = AverageMeter()
#             ave_lossL1 = AverageMeter()
#             ave_lossTriplet = AverageMeter()
# =============================================================================
            
        
    print('Train epoch: [{}], Time: {:.2f} ' 
          'Train_Loss: {:.4f}, Train_Accuracy: {:0.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all.average(), ave_acc.average()))

     
    if writer != None:
        writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
        writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
    
    if history != None:
        history['train']['epoch'].append(epoch)
        history['train']['loss'].append(ave_loss_all.average())
        history['train']['acc'].append(ave_acc.average())


    
def validate_epoch(network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, 
             extract_features=None, avg_pool=False):    

    ave_loss = AverageMeter()
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

        #labels = batch_data['gt']
        # to GPU
        if gpu != None:
            labels = labels.cuda()
      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
            loss = loss_func(outputs, labels)
            #loss, loss1, loss2  = loss_func(outputs, labels)
            #loss = loss_func(outputs[0], outputs[1], outputs[2])
            acc = acc_func(outputs, labels, im_size)
        
        loss = loss.mean()
        acc = acc.mean()

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

def train_epoch_apn(network, n_branches, dataloader, optimizer, loss_func, 
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
                filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],'patch_starts':starts[j]})  


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
            
        
    print('Train epoch: [{}], Time: {:.2f}, ' 
          'Train_Loss: {:.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all.average()))

     
    if writer != None:
        writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
        writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
    
    if history != None:
        history['train']['epoch'].append(epoch)
        history['train']['loss'].append(ave_loss_all.average())
        history['train']['acc'].append(ave_acc.average())


    
def validate_epoch_apn(network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, 
             extract_features=None, avg_pool=False):    

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

        #labels = batch_data['gt']
        # to GPU
        if gpu != None:
            labels = labels.cuda()

      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
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

def train_epoch_unet(network, n_branches, dataloader, optimizer, loss_func, 
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
                filewriter.writerow({'epoch': epoch, 'im_idx': im_idx[j],'patch_starts':starts[j]})  


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
            
        
    print('Train epoch: [{}], Time: {:.2f}, ' 
          'Train_Loss: {:.4f}, Train_Acc: {:.4f}'
          .format(epoch, time.time()-epoch_start,
                  ave_loss_all.average(), ave_acc_all.average()))

     
    if writer != None:
        writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
        writer.add_scalar('Train/Acc', ave_acc_all.average(), epoch)
    
    if history != None:
        history['train']['epoch'].append(epoch)
        history['train']['loss'].append(ave_loss_all.average())
        history['train']['acc'].append(ave_acc.average())


    
def validate_epoch_unet(network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, directories, network_name, outputtime, 
             extract_features=None, avg_pool=False):    

    ave_loss = AverageMeter()
    ave_lossL1 = AverageMeter()
    ave_lossTriplet = AverageMeter()
    ave_lossBCE = AverageMeter()
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
        labels_bottle = batch_data['label_bottle']

        #labels = batch_data['gt']
        # to GPU
        if gpu != None:
            labels = labels.cuda()
            labels_bottle = labels_bottle.cuda()
      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches, extract_features=extract_features, avg_pool=avg_pool)
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