#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic PGD algorithm. Can be equipped with projection to any (convex) set.

"""

from SimplexProjections import *
import numpy as np

def PGD(obj_func, obj_func_grad, Proj, step_size, x0, num_iters, tol):
    '''
    Generic PGD algorithm.
    Proj can be anything. 
    For timing purposes only.
    '''
    
    x = x0
    err = obj_func(x)
    ii = 0
    
    while err > tol and ii <= num_iters:
        x_temp = x - step_size*obj_func_grad(x)
        x = Proj(x_temp)
        err = obj_func(x)
        ii += 1
    
    return err, ii
        
    

