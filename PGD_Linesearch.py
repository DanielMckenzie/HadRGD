#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple class for implementing Projected Gradient Descent.
"""

import numpy as np
from SimplexProjections import *

from baseOptimizer import BaseOptimizer

class PGDL_Feasible(BaseOptimizer):
    '''
    Class implementing PGD over the simplex with line search. 
    This class uses the Armijo rule along the feasible direction (see pg.
    274-275 of "Nonlinear Optimization" by Bertsekas).
    Upon initialization the user also needs to specify the projection alg. 
    '''
    
    def __init__(self, objfunc, objfuncGrad, projAlg, x0, s, beta, rho1):
        self._objfunc = objfunc
        self._objfuncGrad = objfuncGrad
        self._projAlg = projAlg
        self._s = s
        self._beta = beta
        self._rho1 = rho1
        self.x = x0
        
    def ArmijoLineSearch(self, xCurr, xBar):
        Armijo_Cond_Satisfied = False
        m = 0
        Armijo_RHS = -self._rho1*np.dot(self._objfuncGrad(xCurr), xBar-xCurr)
        max_iter = 25
        while not Armijo_Cond_Satisfied and m <= max_iter:
            alpha = self._beta**m
            xNew = xCurr + alpha*(xBar - xCurr)
            LHS_temp = self._objfunc(xCurr) - self._objfunc(xNew)
            if  LHS_temp >= alpha*Armijo_RHS:
                Armijo_Cond_Satisfied = True
            else:
                m+=1
        print('Number of PGDL iterations is '+ str(m))       
        return xNew
    
    def step(self):
        grad = self._objfuncGrad(self.x)
        xBar = self._projAlg(self.x - self._s*grad)
        xNew = self.ArmijoLineSearch(self.x, xBar)
        self.x = xNew
        return self.x, self._objfunc(xNew)
            
       
class PGDL_ProjArc(BaseOptimizer):
    '''
    Class implementing PGD over the simplex with line search. 
    This class uses the Armijo rule along the projection arc (see pg.
    274-275 of "Nonlinear Optimization" by Bertsekas).
    Upon initialization the user also needs to specify the projection alg. 
    '''
    
    def __init__(self, objfunc, objfuncGrad, projAlg, x0, sbar, beta, rho1):
        self._objfunc = objfunc
        self._objfuncGrad = objfuncGrad
        self._proj = projAlg
        self._sbar = sbar
        self._beta = beta
        self._rho1 = rho1
        self.x = x0
        
    def ArmijoLineSearch(self, xCurr, grad):
        Armijo_Cond_Satisfied = False
        m = 0
        s = self._sbar
        while not Armijo_Cond_Satisfied and m <= 1000:
            s = s*self._beta**m
            xNew = self._proj(self.x - s*grad)
            LHS_temp = self._objfunc(xCurr) - self._objfunc(xNew)
            RHS_temp = -self._rho1*np.dot(grad, xNew - self.x)
            if  LHS_temp >= RHS_temp:
                Armijo_Cond_Satisfied = True
            else:
                m+=1
        print('Step size is'+ str(s))
        return xNew
    
    def step(self):
        grad = self._objfuncGrad(self.x)
        xNew = self.ArmijoLineSearch(self.x, grad)
        self.x = xNew
        return self.x, self._objfunc(xNew)
        
        
    