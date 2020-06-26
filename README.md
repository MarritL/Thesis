# Towards self-supervised learning for change detection
Code for MSc. Thesis project
Author: Marrit Leenstra
Date: 25/06/2020

## Abstract
Training Convolutional Neural Networks (CNNs) generally requires large-scale labelled datasets. Unfortunately, the availability of refer-
ence labels for remote sensing applications that require multitemporal information is limited. Recently, self-supervised learning has been
proposed as unsupervised feature learning strategy to train deep neural networks without labelled data. Within this paradigm, networks
learn though solving a pretext task with a supervision signal extracted from the data self. In this thesis two pretext tasks to learn
from unlabelled multitemporal data are proposed: (1) discriminate between overlapping and non-overlapping patches and (2) minimiz-
ing difference between overlapping patches in the feature space. The aim of these tasks is to learn features that can be used in change
detection applications, hence the features should reduce radiometric difference due to acquisition conditions and enhance radiometric
difference caused by real changes on the ground. The developed pretext tasks are effective in the first, but can still improve in the latter.
Change detection with the proposed methodology results in a best Average Accuracy of 81.90% and best F1-score of 45.36%, which is
better than learning from scratch.

## Run
Use the **main.py** script to go through the pipeline of the project. The first cell is used for the main configuration applicaple to all cells, parameters only relevant to one part of the project are set in relevant cells. Note that not all settings are compatible.

## Documentation
### Unlabelled training dataset
* 1520 Sentinel-2 1C image pairs. 
* Channels: B1 - B2 - B3 - B4 - B5 - B6 - B7 - B8 - B8A - B9 - B10 - B11 - B12

### Change detection dataset
* OSCD Benchmark dataset (Daudt et al., 2018), available via the DASE portal (http://dase.grss-ieee.org/) 
The dataset is originally divided into 14 training pairs with freely available ground reference maps and 10 test pairs for which reference labels are only
available through the DASE portal. Here 12 images are used as training set; the remaining 2 images for which ground reference maps are freely available will be added to the test set in order to be able to qualitatively evaluate the change detection results.

### Scripts
**main.py**: main pipeline of the project. \n
Settings for pretext task 1: 
* network_settings = {
    'network': 'siamese', 
    'optimizer': 'adam',
    'lr':1e-3,
    'weight_decay': 1e-4,
    'loss': 'bce_sigmoid',
    'n_classes': 2,
    'patch_size': 96,
    'im_size': (96,96),
    'batch_norm': True,
    'weights_file': '',
    'extract_features': None,
    'avg_pool': False} 
* cfg = {'branch': np.array([32,32,32], dtype='object'), 
       'top': np.array([32], dtype='object'),
       'classifier': np.array([128,64,network_settings['n_classes']])} 
* train_settings = {
    'start_epoch': 0,
    'num_epoch':300,
    'batch_size': 25,
    'disp_iter': 25,
    'gpu': 0,
    'early_stopping': 5}
* dataset_settings = {
    'dataset_type' : 'triplet_apn',
    'perc_train': 0.85,
    'channels': np.arange(13),
    'min_overlap': 1, 
    'max_overlap': 1,
    'stride': int(network_settings['patch_size']),
    'patches_per_image': 5} \n
    
Settings for pretext task 2: 
* network_settings = {
    'network': 'triplet_apn', 
    'optimizer': 'adam',
    'lr':1e-3,
    'weight_decay': 1e-4,
    'loss': 'l1+triplet',
    'n_classes': 2,
    'patch_size': 96,
    'im_size': (96,96),
    'batch_norm': True,
    'weights_file': '',
    'extract_features': None,
    'avg_pool': False} 
* cfg = {'branch': np.array([32,32,32], dtype='object'), 
       'top': np.array([32], dtype='object'),
       'classifier': np.array([128,64,network_settings['n_classes']])} 
* train_settings = {
    'start_epoch': 0,
    'num_epoch':300,
    'batch_size': 25,
    'disp_iter': 25,
    'gpu': 0,
    'early_stopping': 5}
* dataset_settings = {
    'dataset_type' : 'triplet_apn',
    'perc_train': 0.85,
    'channels': np.arange(13),
    'min_overlap': 1, 
    'max_overlap': 1,
    'stride': int(network_settings['patch_size']),
    'patches_per_image': 5} \n

## References
* R. C. Daudt, B. Le Saux, A. Boulch, and Y. Gousseau, “Urban change detection for multispectral earth observation using convolutional neural networks,” in IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2018, pp. 2115–2118. doi: 10.1109/IGARSS.2018.8518015.
