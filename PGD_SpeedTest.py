#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking speed of variants of PGD.
"""

import numpy as np
from SimplexProjections import *
import time as time
from PGD_Variants import PGD
from utils import *
from projection_simplex import projection_simplex_pivot
import pickle as pkl

num_dims = 10
num_trials_per_dim = 10

# initialize params and data storing matrices
tol = 1e-8
num_iters = 200

time_PGDC = np.zeros((num_dims, num_trials_per_dim))
time_PGDD = np.zeros((num_dims, num_trials_per_dim))
time_PGDS = np.zeros((num_dims, num_trials_per_dim))
time_PGDB = np.zeros((num_dims, num_trials_per_dim))
sizes = []

for i in range(num_dims):
    
    # Define parameters
    n = 1000 + 500*i
    sizes.append(n)
    m = int(np.ceil(0.1*n))
    A = np.random.rand(m,n)
    # x_true = SampleSimplex(n)
    x_true = CondatProject(np.random.rand(n))
    b = np.dot(A,x_true)
    Lipschitz_constant = PowerMethod(A)
    step_size = 1.9/Lipschitz_constant # aggressive but convergence still guaranteed.
    
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
    
    # Now perform num_trials_per_dim trials each
    for j in range(num_trials_per_dim):
        x0 = CondatProject(np.random.randn(n))
        # x0 = SampleSimplex(n)
        
        # Duchi's Algorithm
        start_time = time.time()
        err = PGD(cost, cost_grad, DuchiProject, step_size, x0, num_iters, tol)
        time_PGDD[i,j] = time.time() - start_time
        print(err)
        
        # Sorting-based Algorithm
        start_time = time.time()
        err = PGD(cost, cost_grad, SortProject, step_size, x0, num_iters, tol)
        time_PGDS[i,j] = time.time() - start_time
        print(err)
        
        # Condat's Algorithm
        start_time = time.time()
        err = PGD(cost, cost_grad, CondatProject, step_size, x0, num_iters, tol)
        time_PGDC[i,j] = time.time() - start_time
        print(err)
        
        # Pivot method of Blondel (essentially Duchi's algorithm)
        start_time = time.time()
        err = PGD(cost, cost_grad, projection_simplex_pivot, step_size, x0, num_iters, tol)
        time_PGDB[i,j] = time.time() - start_time
        

## Uncomment to plot results
#PGDC = np.mean(time_PGDC, axis=1)  
#PGDS = np.mean(time_PGDS, axis=1)
#PGDD = np.mean(time_PGDD, axis=1)
#PGDB = np.mean(time_PGDB, axis=1)
#
#plt.plot(sizes, PGDC, label="Condat")
#plt.plot(sizes, PGDS, label="Sorting")
#plt.plot(sizes, PGDD, label="Duchi")
#plt.plot(sizes, PGDB, label="Blondel")
#plt.legend()
#plt.show()

## Uncomment to save results        
#myFile = open('Results/PGD_Speed_Oct_13_Boundary.p', 'wb')
#results = {"time_PGDD": time_PGDD,
#           "time_PGDS": time_PGDS,
#           "time_PGDC": time_PGDC,
#           "time_PGDB": time_PGDB,
#           "sizes": sizes
#           }
#pkl.dump(results, myFile)
#myFile.close()
