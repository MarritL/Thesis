#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:59:30 2020

@author: M. Leenstra
"""

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
 
def compute_confusion_matrix(gt, changemap, classes=[1,2], class_names=['no change', 'change'], user_producer=False, normalize=True, axis = 1, plot=True, title=None):
    """
    Compute the confusion matrix 

    Parameters
    ----------
    gt : np.ndarray of shape (N,M)
        ground truth array
    changemap : np.ndarray of shape (N,M)
        boolean change map
    classes : list
        classes in the ground truth, integers. The default is [1,2]
    class_names : list
        descripion of classes, can be used for plotting. The default is ['no change', 'change']
    user_producer : boolean, optional
        if true, user, producer and total acc are calculated. The default is False.
    normalize : boolean, optional
        if true confusion matrix is normalized over axis. The default is True.
    axis : int, optional
        0 for division by column total and 1 for division by row total. The default is 1.
    plot : boolean, optional
        if true confusion matrix will be plotted. The default is True.
    title : string, optional
        title for plot. The default is None.

    Returns
    -------
    cm : numpy.ndarray of shape (n_classes, n_classes)
        confuion matrix 
    fig : matplotlib figure
        if plot is true, figure will be returned, otherwise None
    ax : matplotlib axis
        if plot is true, axis will be returned, otherwise None

    """

    # Compute confusion matrix
    pred = np.ones_like(gt)
    pred[changemap] = 2
    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=axis)[:, np.newaxis]
        
    if user_producer:
        if normalize:
            print("user and producer accuracy can only be calculated for un- \
                  normalized confusion matrices")
        else:
            cm = compute_user_producer_acc(cm)
            class_names.extend('accuracy')
    
    fig, ax = None, None
    if plot:
        fig, ax = plot_confusion_matrix(cm, class_names, normalize, title)  
    
    return cm, fig, ax

def compute_user_producer_acc(cm):
    """ computes user and producer accuracy for existing confusion matrix
    
    arguments
    ---------
    cm: numpy.ndarray of shape(N,N)
       confusion matrix with counts
    
    returns
    -------
    new_cm: numpy_ndarray of shape (N+1,N+1)
        confionsion matrix with an extra column and row for producer and 
        user accuracys. The last cell holds overall accuracy. 
    
    """    
    
    cm = cm.astype('float')
    row_sum = np.sum(cm,axis=1)
    for i, r in enumerate(row_sum):
        if r == 0:
            row_sum[i]= 1e-8
    
    col_sum = np.sum(cm,axis=0)
    for i, c in enumerate(col_sum):
        if c == 0:
            col_sum[i]= 1e-8
    
    diag = np.eye(cm.shape[0])*cm
    
    user = diag[np.nonzero(diag)]/col_sum
    producer = diag[np.nonzero(diag)]/row_sum
    accuracy = np.sum(diag)/np.sum(cm)
    
    new_cm = np.zeros((cm.shape[0]+1, cm.shape[1]+1), dtype='float')
    new_cm[:cm.shape[0],:cm.shape[1]] = cm
    new_cm[cm.shape[0],:-1] = user
    new_cm[:-1,cm.shape[0]] = producer
    new_cm[cm.shape[0], cm.shape[1]] = accuracy
    
    return new_cm

def plot_confusion_matrix(cm, class_names, normalize = True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix for images with one-hot encoded labels.
    
    
    arguments
    ---------
    cm: numpy.ndarray of shape (N,N)
        confusion matrix
    class_names: list
        class labels for confusion matrix
    normalize: boolean
        If true, values will be normalized. Default=True
    title: string
        title for plot. default = None
    cmap: matplotlib color map
        default = plt.cm.Blues
    
    returns
    -------
    fig, ax : matplotlib figure and axis
        plot of confusion matrix
    
    """
    if not title:
        title = 'Confusion matrix'

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

def compute_matthews_corrcoef(gt, changemap):
    """
    Compute the Matthews correlation coefficient (MCC)
    
    Parameters
    ----------
    gt : np.ndarray of shape (N,M)
        ground truth array
    changemap : np.ndarray of shape (N,M)
        boolean change map 

    returns
    -------
    mcc: float
        Matthews correlation coefficient (MCC).
            
    """
    
    pred = np.ones_like(gt)
    pred[changemap] = 2

    # Compute matthews correlation coef
    mcc = matthews_corrcoef(gt.flatten(), pred.flatten())
    
    return mcc  
 
def compute_mcc(conf_matrix):
    """ compute multi-class mcc using equation from source:
        https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef
        
    arguments
    ---------
        conf_matrix: numpy.ndarray
            confusion matrix. true in rows, predicted in columns
            
    return
    ------
        mcc: float
            matthews correlation coefficient
    
    """
    correct = np.trace(conf_matrix)
    tot = np.sum(conf_matrix)
    cl_true = np.sum(conf_matrix,axis = 1)
    cl_pred = np.sum(conf_matrix, axis = 0)

    mcc = (correct*tot-np.sum(cl_true*cl_pred))/(np.sqrt((tot**2-np.sum(cl_pred**2))*(tot**2-np.sum(cl_true**2))))
    return mcc

