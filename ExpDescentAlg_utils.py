#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools for Exponential Descent Algorithm, as discussed in 
' The Ordered Subsets Mirror Descent Optimization ... '
by Ben-Tal et al
and
' Mirror descent and nonlinear projected subgradient methods ...'
by Beck & Teboulle

NB: EMDA MUST be initialized with an interior point of the simplex.
"""
import numpy as np
from baseOptimizer import BaseOptimizer


def EMDA(obj_func, obj_func_grad, step_size, x0, num_iters, tol):
    '''
    Simple version of EDA, mainly for timing purposes. We recommend using the 
    class based implementation for most purposes.
    '''
    x = x0
    err = obj_func(x)
    ii = 0
    
    while err > tol and ii <= num_iters:
        grad = obj_func_grad(x)
        xtemp = x*np.exp(-step_size*grad)
        tempsum = np.sum(xtemp)
        x = xtemp/tempsum
        err = obj_func(x)
        ii += 1
    
    return err, ii

class ExpDescentAlg(BaseOptimizer):
    '''
    Class for EDA algorithm on simplex.
    '''
    def __init__(self, objfunc, objfuncGrad, step_size, x0):
        self._objfunc = objfunc
        self._objfuncGrad = objfuncGrad
        self._step_size = step_size
        self._x = x0
        self._k = 0 # iterate counter
        self._fVals = [self._objfunc(self._x)]
        self._iterates = [self._x]
        
    def prox_step(self, x, grad):
        '''
        Apply the MD prox operator
        '''
        xtemp = x*np.exp(-self._step_size*grad/np.sqrt(self._k+1))
        tempsum = np.sum(xtemp)
        xnew = xtemp/tempsum
        
        return xnew
    
    def step(self):
        '''
        Take a single step of MDA
        '''
        x_k = self._x
        grad_k = self._objfuncGrad(x_k)
        xnew = self.prox_step(x_k, grad_k)
        self._x = xnew
        self._iterates.append(xnew)
        self._fVals.append(self._objfunc(xnew))
        self._k +=1
        print(self._objfunc(xnew))
        
