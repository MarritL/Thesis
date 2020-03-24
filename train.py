#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:21:16 2020

@author: M. Leenstra
"""

from tensorboardX import SummaryWriter
from models.netbuilder import NetBuilder, create_loss_function, create_optimizer
from data_generator import PairDataset, TripletDataset, TripletDatasetPreSaved
from data_generator import PartlyOverlapDataset, TripletAPNDataset
from utils import AverageMeter
import torch
from torch.utils.data import DataLoader
import os
import time
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt

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
                  'max_overlap', 'best_acc','best_epoch', 'best_loss']
# =============================================================================
#     with open(os.path.join(directories['intermediate_dir'],directories['csv_models']), 'a') as file:
#         filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
#         filewriter.writeheader()
# =============================================================================
            
    # build network
    pos_weight = 1
    network_settings['im_size'] = (1,1)
    if network_settings['network'] == 'siamese':
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
    else:
        raise Exception('Architecture undefined! \n \
                        Choose one of: "siamese", "triplet", "hypercolumn", \
                        "siamese_unet_diff", "triplet_apn"')
                
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
        dataset_train = TripletDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'])           
        dataset_val = TripletDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot,
            min_overlap = dataset_settings['min_overlap'],
            max_overlap = dataset_settings['max_overlap'])  
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
        dataset_train = TripletAPNDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_train'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot)           
        dataset_val = TripletAPNDataset(
            data_dir=directories['data_path'], 
            indices=dataset_settings['indices_val'], 
            channels=dataset_settings['channels'], 
            one_hot=one_hot)  
    else:
        raise Exception('dataset_type undefined! \n \
                        Choose one of: "pair", "triplet", "triplet_saved",\
                        "overlap", "triplet_apn"')
    
    # Data loaders
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 8)
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=train_settings['batch_size'], 
        shuffle=False,
        num_workers = 4)
         
# =============================================================================
#     # save history?
#     history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 
#                'val':{'epoch': [], 'loss': [], 'acc': []}}
# =============================================================================
    
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
                network=net, 
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
                extract_features=network_settings['extract_features'])
            
            # validation epoch
            best_net_wts, best_acc, best_epoch, best_loss = validate_epoch(
                network=net, 
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
                extract_features=network_settings['extract_features'])
    
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
               'best_loss': best_loss}
    with open(os.path.join(directories['intermediate_dir'], 
                           directories['csv_models']), 'a') as file:
            filewriter = csv.DictWriter(file, fieldnames, delimiter = ",")
            filewriter.writerow(savedata)  
    
    # save best model's weights
    torch.save(best_net_wts, savedata['filename'])
        
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

    
def train_epoch(network, n_branches, dataloader, optimizer, loss_func, 
                acc_func, history, epoch, writer, epoch_iters, disp_iter,
                gpu, im_size, extract_features=None):
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

        if torch.any(torch.isnan(inputs[0])) or torch.any(torch.isnan(inputs[1])) or torch.any(torch.isnan(inputs[2])):
            print("inputs = nan")
            import ipdb
            ipdb.set_trace()
        # forward pass
        outputs = network(inputs, n_branches, extract_features=extract_features)
        if torch.any(torch.isnan(outputs[0])) or torch.any(torch.isnan(outputs[1])):
            print("outputs is nan")
            import ipdb
            ipdb.set_trace()
        loss, loss1, loss2 = loss_func(outputs, labels)
        #loss = loss_func(outputs[0], outputs[1], outputs[2])
        if torch.isnan(loss) :
            print("loss is nan")
            import ipdb
            ipdb.set_trace()
# =============================================================================
#         min0 = np.min(outputs[0][0].detach().numpy())
#         min1 = np.min(outputs[1][0].detach().numpy())
#         max0 = np.max(outputs[0][0].detach().numpy())
#         max1 = np.max(outputs[1][0].detach().numpy())
#         mean0 = np.mean(outputs[0][0].detach().numpy())
#         mean1 = np.mean(outputs[1][0].detach().numpy())
# =============================================================================
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
        ave_lossTriplet.update(loss2.data.itme())
        ave_acc.update(acc.item())
        

        # calculate accuracy, and display
        if (i+1) % disp_iter == 0:
            #print("Combined: {0:.3f}, L1: {1:.3f}, triplet: {2:.3f}".format(loss, loss1, loss2))
            print('Epoch: [{}][{}/{}], Batch-time: {:.2f}, Data-time: {:.2f}, '
                  'Loss: {:.4f}, Acc: {:.4f}, L1: {:.4f}, LTriplet: {:.4f}'
                  .format(epoch, i+1, epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_loss.average(), ave_acc.average(), 
                          ave_lossL1.average(), ave_lossTriplet.average()))
            ave_loss_all.update(ave_loss.average())
            ave_loss = AverageMeter()
            ave_lossL1 = AverageMeter()
            ave_lossTriplet = AverageMeter()
            
            
# =============================================================================
#         if (i+1) > 105:
#             fig, (ax0, ax1) = plt.subplots(ncols=2)
#             im0 = ax0.imshow(outputs[0][0].detach())
#             im1 = ax1.imshow(outputs[1][0].detach())
#             fig.colorbar(im0, ax=ax0)
#             fig.colorbar(im1, ax=ax1)
# =============================================================================


# =============================================================================
#         fractional_epoch = epoch + 1. * i / epoch_iters
#         history['train']['epoch'].append(fractional_epoch)
#         history['train']['loss'].append(loss.data.item())
#         history['train']['acc'].append(acc.item())
# =============================================================================

    print('Train epoch: [{}], Time: {:.2f}' 
          .format(epoch, (time.time()-epoch_start)))
        
    writer.add_scalar('Train/Loss', ave_loss_all.average(), epoch)
    writer.add_scalar('Train/Acc', ave_acc.average(), epoch)
 
    


    
def validate_epoch(network, n_branches, dataloader, loss_func, acc_func, history, 
             epoch, writer, val_epoch_iters, best_net_wts, best_acc, best_epoch,
             best_loss, gpu, im_size, extract_features=None):    

    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    time_meter = AverageMeter()

    network.eval()
    
    iterator = iter(dataloader)

    # main loop
    tic = time.time()
    for i, batch_data in enumerate(iterator):  

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
            outputs = network(inputs, n_branches, extract_features)
            loss, loss1, loss2  = loss_func(outputs, labels)
            #loss = loss_func(outputs[0], outputs[1], outputs[2])
            acc = acc_func(outputs, labels, im_size)

# =============================================================================
# fig, ax = plt.subplots()
# im2 = ax.imshow(outputs[1][0].detach())
# fig.colorbar(im2, ax=ax)
# =============================================================================
        
        loss = loss.mean()
        acc = acc.mean()

        # update average loss and acc
        ave_loss.update(loss.data.item())
        ave_acc.update(acc.item())

        # measure elapsed time
        time_meter.update(time.time() - tic)
        tic = time.time()

# =============================================================================
#         # calculate accuracy, and display
#         fractional_epoch = epoch + 1. * i / val_epoch_iters
#         history['val']['epoch'].append(fractional_epoch)
#         history['val']['loss'].append(loss.data.item())
#         history['val']['acc'].append(acc.item())
# =============================================================================

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
    
    return(best_net_wts, best_acc, best_epoch, best_loss)