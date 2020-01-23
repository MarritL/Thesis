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

