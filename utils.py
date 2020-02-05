#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:49:59 2020

@author: M. Leenstra
"""
import os

def list_directories(directory):
    dirlist = []
    for f in os.listdir(directory): 
        if os.path.isdir(os.path.join(directory,f)):
            dirlist.append(f)
    return dirlist

class AverageMeter(object):
    """
    Computes and stores the average and current value
    source: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/utils.py
    """
    
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


