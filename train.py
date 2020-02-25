#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:21:16 2020

@author: M. Leenstra
"""

from tensorboardX import SummaryWriter
from models.netbuilder import NetBuilder, create_loss_function, create_optimizer
from data_generator import PairDataset, TripletDataset
from utils import AverageMeter
import torch
from torch.utils.data import DataLoader
import os
import time
import csv
import copy

def train(directories, dataset_settings, network_settings, train_settings):
    
    # init tensorboard
    network_name = network_settings['network']
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    tb_dir = os.path.join(directories['tb_dir'], 'network-{}_date-{}'
                          .format(network_name, outputtime))
    writer = SummaryWriter(logdir=tb_dir)
    print("tensorboard --logdir {}".format(tb_dir))
    
    # init save-file
    fieldnames = ['filename', 'networkname', 'cfg_branch', 'cfg_top', \
                  'optimizer', 'lr', 'weight_decay', 'loss', 'n_classes', \
                  'n_channels','patch_size','batch_norm','dataset', 'best_acc','best_epoch']
# =============================================================================
#     with open(os.path.join(directories['intermediate_dir'],directories['csv_models']), 'a') as file:
#         filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
#         filewriter.writeheader()
# =============================================================================
            
    # build network
    if network_settings['network'] == 'siamese':
        n_branches = 2
    elif network_settings['network'] == 'triplet':
        print('n_branches = 3')
        n_branches = 3
        network_settings['network'] = 'siamese'
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet"')
        
        
    net = NetBuilder.build_network(
        net=network_settings['network'],
        cfg=network_settings['cfg'],
        n_channels=len(dataset_settings['channels']), 
        n_classes=network_settings['n_classes'],
        patch_size=network_settings['patch_size'],
        batch_norm=network_settings['batch_norm'],
        n_branches=n_branches)  
       
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'])
    
    ## TODO: load net into GPU (also the data): NOTE: should be done before 
    # constructing optimizer https://pytorch.org/docs/stable/optim.html    
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
        dataset_train = TripletDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot)           
        dataset_val = TripletDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'])  
    
    # Data loaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=train_settings['batch_size'], 
        shuffle=True,
        num_workers = 2)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 1)
         
    # save history?
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    
    epoch_iters =  max(len(dataset_train) // train_settings['batch_size'],1)
    val_epoch_iters = max(len(dataset_val) // train_settings['batch_size'],1)
    
    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(train_settings['start_epoch'], 
                       train_settings['start_epoch']+train_settings['num_epoch']):
        
        #training epoch
        train_epoch(
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
            gpu = train_settings['gpu'])
        
        # validation epoch
        best_net_wts, best_acc, best_epoch = validate(
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
            gpu = train_settings['gpu']) 

        # save progress
        #checkpoint(nets, history, cfg, epoch+1, DIR) 
    
    # TODO: write info to csv-file
    savedata = {'filename':os.path.join(directories['model_dir'],
                        'network-{}_date-{}'.format(network_name, outputtime)), 
               'networkname': network_name, 
               'cfg_branch': str(list(network_settings['cfg']['branch'])), 
               'cfg_top': str(list(network_settings['cfg']['top'])),
               'optimizer': network_settings['optimizer'],
               'lr': network_settings['lr'], 
               'weight_decay': network_settings['weight_decay'],
               'loss': network_settings['loss'],
               'n_classes': network_settings['n_classes'],
               'n_channels' : len(dataset_settings['channels']),
               'patch_size': network_settings['patch_size'],
               'batch_norm': network_settings['batch_norm'],
               'dataset': dataset_settings['dataset_type'], 
               'best_acc': best_acc,
               'best_epoch': best_epoch}
    with open(os.path.join(directories['intermediate_dir'], 
                           directories['csv_models']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writerow(savedata)  
    
    # TODO: save model (make better, best model etc.)
    torch.save(best_net_wts, savedata['filename'])
        
    print('Training Done!')
    writer.close()


    
def train_epoch(network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu):
    print("gpu ", gpu)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    
    network.train() 
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    epoch_start = time.time()
    #for i in range(epoch_iters):
    for i, batch_data in enumerate(iterator):
        
        # load a batch of data
        #batch_data = next(iterator)
        data_time.update(time.time() - tic)
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # get the inputs
        if n_branches == 2:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda()] 
                print("inputs to gpu")
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float()] 

        elif n_branches == 3:
            if gpu != None:
                inputs = [batch_data['patch0'].float().cuda(), 
                          batch_data['patch1'].float().cuda(),
                          batch_data['patch2'].float().cuda()] 
                print("inputs to gpu")
            else:
                inputs = [batch_data['patch0'].float(), 
                          batch_data['patch1'].float(),
                          batch_data['patch2'].float()] 
        labels = batch_data['label']
    
        # TODO: to gpu
        if gpu != None:
            print('labels to gpu')
            #for patch in inputs:
            #    patch.cuda()
            labels = labels.cuda()

        # forward pass
        outputs = network(inputs, n_branches)
        print("outputs ", outputs)
        print("lables: ", labels)
        loss = loss_func(outputs, labels)
        acc = acc_func(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_acc.update(acc.item())
        

        # calculate accuracy, and display
        if i % disp_iter == 0:
            print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                  'Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch, i, epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_loss.average(), ave_acc.average()))

        fractional_epoch = epoch + 1. * i / epoch_iters
        history['train']['epoch'].append(fractional_epoch)
        history['train']['loss'].append(loss.data.item())
        history['train']['acc'].append(acc.item())

    print('Train epoch: [{}], Time: {:.2f}' 
          .format(epoch, (time.time()-epoch_start)))
        
    writer.add_scalar('Train/Loss', ave_loss.average(), epoch)
    writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
 
    


    
def validate(network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             gpu):    

    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    time_meter = AverageMeter()

    network.eval()
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    #for i in range(val_epoch_iters):
    for i, batch_data in enumerate(iterator):
        
        # load a batch of data
        #batch_data = next(iterator)
        
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
        
        # TODO: to GPU
        if gpu != None:
            labels.cuda()
      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs, n_branches)
            loss = loss_func(outputs, labels)
            acc = acc_func(outputs, labels)
        
        loss = loss.mean()
        acc = acc.mean()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_acc.update(acc.item())

        # measure elapsed time
        time_meter.update(time.time() - tic)
        tic = time.time()

        # calculate accuracy, and display
        fractional_epoch = epoch + 1. * i / val_epoch_iters
        history['val']['epoch'].append(fractional_epoch)
        history['val']['loss'].append(loss.data.item())
        history['val']['acc'].append(acc.item())

    print('Val epoch: [{}], Time: {:.2f}, ' 
          'Val_Loss: {:.4f}, Val_Accuracy: {:0.4f}'
          .format(epoch, time_meter.value(),
                  ave_loss.average(), ave_acc.average()))
    
    writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
    writer.add_scalar('Val/Acc', ave_acc.average(), epoch)
    
    if ave_acc.average() > best_acc:
        best_acc = ave_acc.average()
        best_net_wts = copy.deepcopy(network.state_dict())
        best_epoch = epoch
    
    return(best_net_wts, best_acc, best_epoch)