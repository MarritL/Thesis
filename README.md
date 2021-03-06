# Towards self-supervised learning for change detection
Code for MSc. Thesis project \
Author: Marrit Leenstra \
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
Use the **main.py** script to go through the pipeline of the project. \ 
The first cell is used for the main configuration applicaple to all cells, parameters only relevant to one part of the project are set in relevant cells. Note that not all settings are compatible.

## Documentation
### Unlabelled training dataset
* 1520 Sentinel-2 1C image pairs. 
* Channels: B1 - B2 - B3 - B4 - B5 - B6 - B7 - B8 - B8A - B9 - B10 - B11 - B12

### Change detection dataset
* OSCD Benchmark dataset (Daudt et al., 2018), available via the DASE portal (http://dase.grss-ieee.org/) \
The dataset is originally divided into 14 training pairs with freely available ground reference maps and 10 test pairs for which reference labels are only
available through the DASE portal. Here 12 images are used as training set; the remaining 2 images for which ground reference maps are freely available will be added to the test set in order to be able to qualitatively evaluate the change detection results.

### Scripts
**main.py**: main pipeline of the project. \
* For pretext task training use the cell: """ Train on S21C dataset""". \
Set hyperparemeters in first cell.
* For training of classifiers for change detection use the cell: """ Train supervised classifier on OSCD-dataset, fixed patches""". \
Note: first train the cross-validation folds, then full training set. Set hyperparemeters in first cell.
* For inference with CVA classifier use cell: """ Evaluate (preliminary) network on change detection task, using CVA classifier """ 
* For inference with convolutional classifiers use cell: """ Evalutate downstream model on OSCD dataset, using 1-layer conv. classifier or 2-layer conv. classifier """ \
\
Settings for pretext task 1: 
* network_settings = { \
&emsp; 'network': 'siamese', \
&emsp;'optimizer': 'adam', \
&emsp;'lr':1e-3, \
    &emsp;'weight_decay': 1e-4, \
    &emsp;'loss': 'bce_sigmoid', \
    &emsp;'n_classes': 2, \
    &emsp;'patch_size': 96, \
    &emsp;'im_size': (96,96), \
    &emsp;'batch_norm': True, \
    &emsp;'weights_file': '', \
    &emsp;'extract_features': None, \
    &emsp;'avg_pool': False} 
* cfg = {'branch': np.array([32,32,32], dtype='object'),  \
       &emsp;'top': np.array([32], dtype='object'), \
       &emsp;'classifier': np.array([128,64,network_settings['n_classes']])} 
* train_settings = { \
    &emsp;'start_epoch': 0, \
    &emsp;'num_epoch':300, \
    &emsp;'batch_size': 25, \
    &emsp;'disp_iter': 25, \
    &emsp;'gpu': 0, \
    &emsp;'early_stopping': 5}
* dataset_settings = { \
    &emsp;'dataset_type' : 'triplet_apn', \
    &emsp;'perc_train': 0.85, \
    &emsp;'channels': np.arange(13), \
    &emsp;'min_overlap': 1,  \
    &emsp;'max_overlap': 1, \
    &emsp;'stride': int(network_settings['patch_size']), \
    &emsp;'patches_per_image': 5} \
    
Settings for pretext task 2: 
* network_settings = { \
    &emsp;'network': 'triplet_apn',  \
    &emsp;'optimizer': 'adam', \
    &emsp;'lr':1e-3, \
    &emsp;'weight_decay': 1e-4, \
    &emsp;'loss': 'l1+triplet', \
    &emsp;'n_classes': 2, \
    &emsp;'patch_size': 96, \
    &emsp;'im_size': (96,96), \
    &emsp;'batch_norm': True, \
    &emsp;'weights_file': '', \
    &emsp;'extract_features': None, \
    &emsp;'avg_pool': False} 
* cfg = {'branch': np.array([32,32,32], dtype='object'),  \
       &emsp;'top': np.array([32], dtype='object'), \
       &emsp;'classifier': np.array([128,64,network_settings['n_classes']])} 
* train_settings = { \
    &emsp;'start_epoch': 0, \
    &emsp;'num_epoch':300, \
    &emsp;'batch_size': 25, \
    &emsp;'disp_iter': 25, \
    &emsp;'gpu': 0, \
    &emsp;'early_stopping': 5}
* dataset_settings = { \
    &emsp;'dataset_type' : 'triplet_apn', \
    &emsp;'perc_train': 0.85, \
    &emsp;'channels': np.arange(13), \
    &emsp;'min_overlap': 1,  \
    &emsp;'max_overlap': 1, \
    &emsp;'stride': int(network_settings['patch_size']), \
    &emsp;'patches_per_image': 5} \
    
**train.py**: functions used during training. \
**data_generator.py**: script that loads the datasets in batches. \
**models**: folder with all models\
**models/netbuilder.py**: script that loads the correct model, loss-function and accuracy-function \
**models/siamese_net.py**: network used for pretext task 1 \
**models/siamese_net_apn.py**: network used for pretext task 2 \
**models/x**: other models that are not used in final version of thesis \
**change_detection.py**: functions to apply thresholding technqiues to features extracted from the images, used for CVA \
**inference.py**: functions to create change maps.
**extract_features**: functions to extract features from specified layers of CNN. Also functions to caluculate distance map or difference map. \
**plots.py**: plotting functions used in througout the other scripts \
**data_download_functions.py**: functions to download Unlabelled training dataset  \
**download_images.py**: script used to download Unlabelled training dataset (result in S21C_dataset.csv). \  
**geometric_registration.py**: script for evaluation of geometric registration accuracy of unlabelled dataset and OSCD dataset. \
**experiment_radiometric_difference.py**: script for evaluation of radiometric registration accuracy of unlabelled dataset and OSCD dataset. \
**utils.py**: some supporting functions  \
**setup.py**: quick setup of settings to be used in other scripts (e.g. directories dictionary)

## References
* R. C. Daudt, B. Le Saux, A. Boulch, and Y. Gousseau, “Urban change detection for multispectral earth observation using convolutional neural networks,” in IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2018, pp. 2115–2118. doi: 10.1109/IGARSS.2018.8518015.
