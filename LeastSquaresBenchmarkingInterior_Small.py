#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for benchmarking HadRGD (with no line search) against several commonly 
used methods: PGD, Frank-Wolfe (no line search), Mirror Descent.

Using smaller problem sizes as several prior algorithms do not scale well.

Can toggle three solution types:
    1. x_true in interior of simplex.
    2. x_true on boundary, moderately sparse.
    3. x_true is a corner, so 1-sparse.
"""


import numpy as np
from SimplexProjections import *
import time as time
import copt
from PGD_Variants import PGD
from utils import *
from projection_simplex import projection_simplex_pivot
from Riemannian_algs import RGD
from ExpDescentAlg_utils import EDA
import pickle as pkl
import matplotlib.pyplot as plt

num_dims = 10
num_trials_per_dim = 10

# initialize params and data storing matrices
tol = 1e-8
max_iters = 500

time_PGD_simplex = np.zeros((num_dims, num_trials_per_dim))
time_EDA = np.zeros((num_dims, num_trials_per_dim))
time_RGD_hadamard = np.zeros((num_dims, num_trials_per_dim))
time_PFW = np.zeros((num_dims, num_trials_per_dim))

num_iters_PGD_simplex = np.zeros((num_dims, num_trials_per_dim))
num_iters_EDA = np.zeros((num_dims, num_trials_per_dim))
num_iters_RGD_hadamard = np.zeros((num_dims, num_trials_per_dim))
num_iters_PFW = np.zeros((num_dims, num_trials_per_dim))

err_PGD_simplex = np.zeros((num_dims, num_trials_per_dim))
err_EDA = np.zeros((num_dims, num_trials_per_dim))
err_RGD_hadamard = np.zeros((num_dims, num_trials_per_dim))
err_PFW = np.zeros((num_dims, num_trials_per_dim))
sizes = []

for i in range(num_dims):
    
    # Define parameters
    n = 500 + 100*i
    sizes.append(n)
    m = int(np.ceil(0.1*n))
    A = np.random.rand(m,n)
    x_true = SampleSimplex(n) # Important: True Solution is in interior of Simplex.
    # x_true = CondatProject(np.random.rand(n))
    b = np.dot(A,x_true)
    Lipschitz_constant = PowerMethod(A)
    step_size = 1.99/Lipschitz_constant # agressive but convergence still guaranteed.
    print('step_size is ' + str(step_size))
    Had_step_size = np.sqrt(n)*np.sqrt(step_size) # Heuristic but we find this to work?
    # Had_step_size = 10*step_size
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
        x0 = SampleSimplex(n) # Important: initialize in interior of simplex.
        z0 = np.sqrt(x0)  # initialization for Hadamard methods.
        #z0 = np.random.randn(n)
        #z0 = z0/np.linalg.norm(z0)
        #x0 = z0*z0
        
        # PGD on simplex using Duchi's Algorithm
        start_time = time.time()
        err, num_iters = PGD(cost, cost_grad, DuchiProject, step_size, x0, max_iters, tol)
        time_PGD_simplex[i,j] = time.time() - start_time
        err_PGD_simplex[i,j] = err
        num_iters_PGD_simplex[i,j] = num_iters
        print(err)
        
        # RGD on sphere using Hadamard parametrization
        start_time = time.time()
        err, num_iters = RGD(cost_H, cost_H_grad, Had_step_size, z0, max_iters, tol)
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
        
        
        # Exponential/Mirror descent
        start_time = time.time()
        step_size_EDA = 20/Lipschitz_constant
        err, num_iter = EDA(cost, cost_grad, step_size_EDA, z0, int(np.ceil(np.sqrt(n)*max_iters)), tol)
        time_EDA[i,j] = time.time() - start_time
        err_EDA[i,j] = err
        num_iters_EDA[i, j] = num_iter
        print(num_iter)
        print(err)
        
#PGD_simplex = np.mean(time_PGD_simplex, axis=1)  
#PGD_hadamard = np.mean(time_PGD_hadamard, axis=1)
#RGD_hadamard = np.mean(time_RGD_hadamard, axis=1)
#FW = np.mean(time_FW, axis=1)
# EDA = np.mean(time_EDA, axis=1)

#PGD_simplex_iters = np.mean(num_iters_PGD_simplex, axis=1)
#PGD_hadamard_iters = np.mean(num_iters_PGD_hadamard, axis=1)
#RGD_hadamard_iters = np.mean(num_iters_RGD_hadamard, axis=1)


#plt.plot(sizes, PGD_simplex, label="PGD on simplex")
#plt.plot(sizes, PGD_hadamard, label="PGD using Hadamard")
#plt.plot(sizes, RGD_hadamard, label="HadRGD")
#plt.plot(sizes, FW, label="Frank-Wolfe")
#plt.plot(sizes, EDA, label="Exponential Descent Algorithm")
#plt.legend()
#plt.show()
#plt.savefig('LeastSquares_Interior_Time_Aug_27.png')
#plt.clf()

#plt.plot(sizes, PGD_simplex_iters, label="PGD on simplex")
#plt.plot(sizes, PGD_hadamard_iters, label="PGD using Hadamard")
#plt.plot(sizes, RGD_hadamard_iters, label="RGD using Hadamard")
#plt.plot(sizes, FW, label="Frank-Wolfe")
#plt.plot(sizes, EDA, label="Exponential Descent Algorithm")
#plt.legend()
#plt.show()
#plt.savefig('LeastSquares_Interior_Iters_Aug_27.png')


myFile = open('Results/LeastSquaresBenchmarkResultsInterior_Oct_12.p', 'wb')
results = {"time_PGD_simplex":time_PGD_simplex, 
           "time_RGD_hadamard": time_RGD_hadamard,
           "time_EDA": time_EDA,
           "time_PFW":time_PFW,
           "err_PGD_simplex": err_PGD_simplex,
           "err_EDA": err_EDA,
           "err_RGD_hadamard": err_RGD_hadamard,
           "err_PFW": err_PFW,
           "num_iters_PGD_simplex": num_iters_PGD_simplex,
           "num_iters_EDA": num_iters_EDA,
           "num_iters_RGD_hadamard": num_iters_RGD_hadamard,
           "num_iters_PFW:": num_iters_PFW,
           "sizes": sizes
           }
pkl.dump(results, myFile)
myFile.close()
    