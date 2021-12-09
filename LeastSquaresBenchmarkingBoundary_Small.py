#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for benchmarking HadRGD (with no line search) against several commonly 
used methods: PGD, Frank-Wolfe (no line search), Mirror Descent.

This version uses x_true on boundary.
"""

import numpy as np
from SimplexProjections import *
import time as time
import copt
from PGD_Variants import PGD
from utils import *
from projection_simplex import projection_simplex_pivot
from Riemannian_algs import RGD
from ExpDescentAlg_utils import EMDA
import pickle as pkl
import matplotlib.pyplot as plt

num_dims = 10
num_trials_per_dim = 10

# initialize params and data storing matrices
tol = 1e-8
max_iters = 3000

time_PGD_simplex = np.zeros((num_dims, num_trials_per_dim))
time_EMDA = np.zeros((num_dims, num_trials_per_dim))
time_RGD_hadamard = np.zeros((num_dims, num_trials_per_dim))
time_PFW = np.zeros((num_dims, num_trials_per_dim))

num_iters_PGD_simplex = np.zeros((num_dims, num_trials_per_dim))
num_iters_EMDA = np.zeros((num_dims, num_trials_per_dim))
num_iters_RGD_hadamard = np.zeros((num_dims, num_trials_per_dim))
num_iters_PFW = np.zeros((num_dims, num_trials_per_dim))

err_PGD_simplex = np.zeros((num_dims, num_trials_per_dim))
err_EMDA = np.zeros((num_dims, num_trials_per_dim))
err_RGD_hadamard = np.zeros((num_dims, num_trials_per_dim))
err_PFW = np.zeros((num_dims, num_trials_per_dim))
sizes = []

for i in range(num_dims):
    
    # Define parameters
    n = 500 + 100*i
    sizes.append(n)
    m = int(np.ceil(0.1*n))
    A = np.random.rand(m,n)
    x_true = CondatProject(np.random.rand(n))
    b = np.dot(A,x_true)
    Lipschitz_constant = PowerMethod(A)
    step_size = 1.99/Lipschitz_constant # aggressive but convergence still guaranteed.
    print('step_size is ' + str(step_size))
    Had_step_size = n**0.25*np.sqrt(step_size) # Heuristic but we find this to work.

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
        x0 = SampleSimplex(n)
        z0 = np.sqrt(x0)  # initialization for Hadamard methods.

        # PGD on simplex using Duchi's Algorithm
        start_time = time.time()
        err, num_iters = PGD(cost, cost_grad, DuchiProject, step_size, x0,
                             max_iters, tol)
        time_PGD_simplex[i,j] = time.time() - start_time
        err_PGD_simplex[i,j] = err
        num_iters_PGD_simplex[i,j] = num_iters
        print(err)
        
        # RGD on sphere using Hadamard parametrization
        start_time = time.time()
        err, num_iters = RGD(cost_H, cost_H_grad, Had_step_size, z0, 
                             max_iters, tol)
        time_RGD_hadamard[i,j] = time.time() - start_time
        err_RGD_hadamard[i,j] = err
        num_iters_RGD_hadamard[i,j] = num_iters
        print(err)
        
        # Pairwise Frank-Wolfe 
        # As FW on simplex is essentially a coordinate descent alg. it gets 
        # more iterations.
        init_idx = np.random.randint(n)
        x0 = np.zeros(n)
        x0[init_idx] = 1.0
        cb = copt.utils.Trace(cost)
        sol = copt.minimize_frank_wolfe(cost, x0, LinMinOracle, x0_rep=init_idx,
                                        variant='pairwise', jac=cost_grad, 
                                        step="DR", lipschitz=Lipschitz_constant,
                                        callback=cb, verbose=True, max_iter=int(np.ceil(np.sqrt(n)*max_iters)))
        success_idx = FindFirstLessThan(cb.trace_fx, tol)
        print(success_idx)
        print(cb.trace_fx[success_idx])
        time_PFW[i, j] = cb.trace_time[success_idx]
        err_PFW[i, j] = cb.trace_fx[success_idx]
        num_iters_PFW[i, j] = success_idx
        
        
        # Entropic Mirror Descent Algorithm (EMDA)
        start_time = time.time()
        step_size_EMDA = 20/Lipschitz_constant
        err, num_iter = EDA(cost, cost_grad, step_size_EMDA, z0,
                            int(np.ceil(np.sqrt(n)*max_iters)), tol)
        time_EMDA[i,j] = time.time() - start_time
        err_EMDA[i,j] = err
        num_iters_EMDA[i, j] = num_iter
        print(num_iter)
        print(err)
        

## Uncomment the code below to save results.
#myFile = open('Results/LeastSquaresBenchmarkResultsBoundary_Oct_12.p', 'wb')
#results = {"time_PGD_simplex":time_PGD_simplex, 
#           "time_RGD_hadamard": time_RGD_hadamard,
#           "time_EMDA": time_EMDA,
#           "time_PFW":time_PFW,
#           "err_PGD_simplex": err_PGD_simplex,
#           "err_EMDA": err_EMDA,
#           "err_RGD_hadamard": err_RGD_hadamard,
#           "err_PFW": err_PFW,
#           "num_iters_PGD_simplex": num_iters_PGD_simplex,
#           "num_iters_EMDA": num_iters_EMDA,
#           "num_iters_RGD_hadamard": num_iters_RGD_hadamard,
#           "num_iters_PFW:": num_iters_PFW,
#           "sizes": sizes
#           }
#pkl.dump(results, myFile)
#myFile.close()
    