#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reproduces the results of Figure 3 in "From the simplex to the 
sphere".

"""

import numpy as np
from SimplexProjections import *
import time as time
from PGD_Linesearch import PGDL_Feasible
from utils import *
from Riemannian_algs import HadRGD_BB, HadRGD_AW
import pickle as pkl
import matplotlib.pyplot as plt
import copy as copy
import copt

num_dims = 21
num_trials_per_dim = 10

# initialize params and data storing matrices
tol = 1e-8
max_iters = 400

time_PGDL = np.zeros((num_dims, num_trials_per_dim))
time_HadRGD_AW = np.zeros((num_dims, num_trials_per_dim))
time_HadRGD_BB = np.zeros((num_dims, num_trials_per_dim))
time_PFW = np.zeros((num_dims, num_trials_per_dim))

num_iters_PGDL = np.zeros((num_dims, num_trials_per_dim))
num_iters_HadRGD_AW = np.zeros((num_dims, num_trials_per_dim))
num_iters_HadRGD_BB = np.zeros((num_dims, num_trials_per_dim))
num_iters_PFW = np.zeros((num_dims, num_trials_per_dim))

err_PGDL = np.zeros((num_dims, num_trials_per_dim))
err_HadRGD_AW = np.zeros((num_dims, num_trials_per_dim))
err_HadRGD_BB = np.zeros((num_dims, num_trials_per_dim))
err_PFW = np.zeros((num_dims, num_trials_per_dim))

sizes = []

for i in range(num_dims):
    
    # Define parameters
    n = 2000 + 1000*i
    sizes.append(n)
    m = int(np.ceil(0.1*n))
    A = np.random.rand(m,n)
    x_true = SampleSimplex(n) 
    b = np.dot(A,x_true)
    Lipschitz_constant = PowerMethod(A)
    step_size =  10*(1.99/Lipschitz_constant) 
    Had_step_size = 10*np.sqrt(n)*np.sqrt(1.99/Lipschitz_constant)
    
    # Define objective function and gradient 
    def cost(x):
        '''
        Least squares loss function.
        '''
        temp = np.dot(A,x) - b
        return np.dot(temp,temp)/2

    def cost_grad(x):
        '''
        Gradient of least squares loss function.
        '''
        temp = np.dot(A,x) - b
        return np.dot(A.T,temp)
    
    # Hadamard parametrization variants of objective function and gradient
    
    def cost_H(z):
        '''
        Hadamard parametrized least squares cost. Mainly for use with autodiff.
        '''
        temp = np.dot(A,z*z) - b
        return np.dot(temp, temp)

    def cost_H_grad(z):
        '''
        Gradient of Hadamard parametrized least squares cost. For use in line search.
        '''
        temp = np.dot(A,z*z) - b
        return np.dot(A.T,temp)*z
    
    # Now perform num_trials_per_dim trials each
    for j in range(num_trials_per_dim):
        x0 = SampleSimplex(n) # Initialize in interior of simplex.
        z0 = np.sqrt(x0)  # initialization for Hadamard methods.

        # Initialize PGD with Line search
        beta = 0.75
        s = step_size
        sigma = 1e-4
        PGDL = PGDL_Feasible(cost, cost_grad, DuchiProject, x0, s, beta, sigma)
        
        # Run PGD with Line Search
        num_iters = 0
        err = 1e6
        start_time = time.time()
        while err > tol and num_iters < max_iters:
            _, err = PGDL.step()
            num_iters += 1
          
        time_PGDL[i, j] = time.time() - start_time
        err_PGDL[i, j] = err
        num_iters_PGDL[i, j] = num_iters
        print(err)
        
        # Initialize HadRGD_AW
        z = copy.copy(z0)
        default_step_size = Had_step_size
        AWOpt = HadRGD_AW(cost_H, cost_H_grad, z, default_step_size)

        # Run HadRGD_AW
        num_iters = 0
        err = 1e6
        start_time = time.time()
        
        while err > tol and num_iters < max_iters:
            err = AWOpt.step()
            num_iters += 1
        
        time_HadRGD_AW[i, j] = time.time() - start_time
        err_HadRGD_AW[i, j] = err
        num_iters_HadRGD_AW[i, j] = num_iters
        print(err)
     
        # Initialize HadRGD_BB
        tau= 3.0
        rho1 = 0.1
        delta = 0.5
        eta = 0.5
        z = copy.copy(z0)
        BBOpt = HadRGD_BB(cost_H, cost_H_grad, tau, rho1, delta, eta, z)
        
        # Run HadRGD_BB
        num_iters = 0
        err = 1e6
        start_time = time.time()
        
        while err > tol and num_iters < max_iters:
            err = BBOpt.step()
            num_iters += 1
        
        time_HadRGD_BB[i, j] = time.time() - start_time
        err_HadRGD_BB[i, j] = err
        num_iters_HadRGD_BB[i, j] = num_iters
        print(err)
        
        # Run Pairwise Frank-Wolfe
        init_idx = np.random.randint(n)
        x0 = np.zeros(n)
        x0[init_idx] = 1.0
        cb = copt.utils.Trace(cost)
        PFW_sol = copt.minimize_frank_wolfe(cost, x0, LinMinOracle, 
                                            x0_rep=init_idx,variant='pairwise',
                                            jac=cost_grad, #lipschitz=Lipschitz_constant,
                                            callback=cb, tol=tol, max_iter=10*max_iters,
                                            verbose=True)
        success_idx = FindFirstLessThan(cb.trace_fx, tol)
        print(cb.trace_fx[success_idx])
        time_PFW[i, j] = cb.trace_time[success_idx]
        err_PFW[i, j] = cb.trace_fx[success_idx]
        num_iters_PFW[i, j] = success_idx
        
        

## Uncomment the code below to save the results.
#myFile = open('Results/LeastSquaresBenchmarkResultsInterior_LineSearch_Oct_14.p', 'wb')
#results = {"time_PGDL":time_PGDL, 
#           "time_HadRGD_AW": time_HadRGD_AW,
#           "time_HadRGD_BB": time_HadRGD_BB,
#           "time_PFW": time_PFW,
#           "num_iters_PGDL": num_iters_PGDL,
#           "num_iters_HadRGD_AW": num_iters_HadRGD_AW,
#           "num_iters_HadRGD_BB": num_iters_HadRGD_BB,
#           "num_iters_PFW": num_iters_PFW,
#           "err_PGDL": err_PGDL,
#           "err_HadRGD_AW": err_HadRGD_AW,
#           "err_HadRGD_BB": err_HadRGD_BB,
#           "err_PFW": err_PFW,
#           "sizes": sizes,
#           }
#pkl.dump(results, myFile)
#myFile.close()
    