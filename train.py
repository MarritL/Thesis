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

def train(directories, dataset_settings, network_settings, train_settings):
    
    # init tensorboard
    outputtime = '{}'.format(time.strftime("%d%m%Y_%H%M%S", time.localtime()))
    tb_dir = os.path.join(directories['tb_dir'], 'network-{}_date-{}'
                          .format(network_settings['network'], outputtime))
    writer = SummaryWriter(logdir=tb_dir)
    print("tensorboard --logdir {}".format(tb_dir))
    
    # build network
    net = NetBuilder.build_network(
        net=network_settings['network'],
        n_channels=len(dataset_settings['channels']), 
        n_classes=network_settings['n_classes'])    
    loss_func, acc_func, one_hot = create_loss_function(network_settings['loss'])
    # TODO: add regularization
    optim = create_optimizer(network_settings['optimizer'], net.parameters(), 
                             network_settings['lr'])

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
            one_hot=one_hot)  
    
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
     
    ## TODO: load net into GPU (also the data)
    
    # save history?
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
               'val':{'epoch': [], 'loss': [], 'acc': []}}
    
    epoch_iters =  len(dataset_train) // train_settings['batch_size']
    val_epoch_iters = len(dataset_val) // train_settings['batch_size']
    
    for epoch in range(train_settings['start_epoch'], 
                       train_settings['start_epoch']+train_settings['num_epoch']):
        
        #training epoch
        train_epoch(
            net, 
            dataloader_train, 
            optim, 
            loss_func,
            acc_func,
            history, 
            epoch, 
            writer,
            epoch_iters, 
            train_settings['disp_iter'])
        
        # validation epoch
        validate(
            net, 
            dataloader_val, 
            loss_func,
            acc_func,
            history, 
            epoch, 
            writer,
            val_epoch_iters) 

       
        # save progress
        #checkpoint(nets, history, cfg, epoch+1, DIR) 
    
    # TODO: write info to csv-file
    print('Training Done!')
    writer.close()

# =============================================================================
# #training epoch
# epoch = 0
# network = net
# #interator = iterator_train
# epoch = epoch+1
# dataloader = dataloader_train
# i = 0
# =============================================================================
    
def train_epoch(network, dataloader, optimizer, loss_func, acc_func, history, 
                epoch, writer, epoch_iters, disp_iter):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    
    network = network.train() 
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    for i in range(epoch_iters):
        
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        
        # set gradients to zero
        optimizer.zero_grad()
        
        # get the inputs
        inputs = [batch_data['patch0'].float(), batch_data['patch1'].float()] 
        labels = batch_data['label']
    
        # TODO: to gpu
        #batch_data = {'img_data': batch_images.cuda(), 'seg_label':batch_segms.cuda()}

        # forward pass
        outputs = network(inputs)
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

        
    writer.add_scalar('Train/Loss', ave_loss.average(), epoch)
    writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
 
    

# =============================================================================
# epoch = 0
# network = net
# dataloader = dataloader_train
# i = 0
# =============================================================================
    
def validate(network, dataloader, loss_func, acc_func, history, epoch, writer, 
             val_epoch_iters):    

    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    time_meter = AverageMeter()

    network.eval()
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    for i in range(val_epoch_iters):
        
        # load a batch of data
        batch_data = next(iterator)
        
        # get the inputs
        inputs = [batch_data['patch0'].float(), batch_data['patch1'].float()] 
        labels = batch_data['label']
        
        # TODO: to GPU
        #batch_data = {'img_data': batch_images.cuda(), 'seg_label':batch_segms.cuda()}
      
        with torch.no_grad():
            # forward pass
            outputs = network(inputs)
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

    print('Epoch: [{}], Time: {:.2f}, ' 
          'Val_Loss: {:.4f}, Val_Accuracy: {:0.4f}'
          .format(epoch, time_meter.average(),
                  ave_loss.average(), ave_acc.average()))
    
    writer.add_scalar('Val/Loss', ave_loss.average(), epoch)
    writer.add_scalar('Val/Acc', ave_acc.average(), epoch)