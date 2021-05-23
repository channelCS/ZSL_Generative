#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:17:15 2021

@author: agupta
"""

import yaml
from collections import OrderedDict
from os import path as osp

def ordered_yaml():
    """
    Support OrderedDict for yaml.
    
    Returns:
        yaml Loader and Dumper.
        
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse(opt_path):
    """
    Parse option file.
    
    Args:
        opt_path (str): Option file path.
        root_path (str): Indicate Root path.
        is_train (str): Indicate whether in training or not. Default: True.
        
    Returns:
        (dict): Options.
    """
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    experiments_root = osp.join('./logs', opt['name'])
    opt['log'] = experiments_root
#        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

    return opt
