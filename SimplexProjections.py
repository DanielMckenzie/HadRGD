#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of different algorithms for projection on to the probability simplex.
"""
import numpy as np
import copy as copy
import random as random

def SortProject(y):
    '''
    Based on Matlab code written by Xiaojing Ye.
    See http://arxiv.org/abs/1101.6081
    '''
    
    m = len(y)
    bget = False
    s = -np.sort(-y)
    tmpsum = 0
    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1)/(ii+1)
        if tmax > s[ii+1]:
            bget = True
            break

    if not bget:
        tmax = (tmpsum+s[m-1]-1)/m

    x = np.maximum(y-tmax, 0)
    return x

def CondatProject(y):
    '''
    Algorithm proposed by Laurent Condat in "Fast projection on to the 
    simplex and the ell_1 ball".
    '''
    v = [y[0]]
    v_tilde = []
    rho = y[0] - 1
    N = len(y)
    # Step 2
    for n in range(1,N):
        if y[n] > rho:
            rho = rho + (y[n]-rho)/(len(v)+1)
            if rho > y[n] - 1:
                v.append(y[n])
            else:
                v_tilde = v_tilde + v
                v = [y[n]]
                rho = y[n] - 1
    # Step 3
    for yy in v_tilde:
        if yy > rho:
            v.append(yy)
            rho = rho + (yy-rho)/len(v)
    # Step 4
    v_old = []
    while not (v == v_old):
        v_old = copy.copy(v)
        for yy in v:
            if yy <= rho:
                v.remove(yy)
                rho = rho + (rho-yy)/len(v)
    # Step 5
    tau = rho
    # K = len(v)
    #Step 6
    x = np.maximum(y-tau, 0)
    return x
        
def DuchiProject(v):
    '''
    Implementation of the (supposedly) O(n) simplex projection algorithm
    advertised in "Efficient Projections onto the ell_1 ball ... " by Duchi
    Shalev-Schwartz, Singer and Chandra.
    
    Note we are only considering prob. simplex so z=1
    '''
    n = len(v)
    U = list(range(n))
    s = 0
    rho = 0
    # Loop until U is empty
    while (len(U) != 0):
        k = random.choice(U)
        vk = v[k]
        # Construct G and L
        G = []
        L = []
        for item in U:
            if v[item] >= vk:
                G.append(item)
            else:
                L.append(item)
        # compute delta_rho and delta_s
        delta_rho = len(G)
        delta_s = np.sum(v[G])
        # Check condition
        if (s+delta_s) - (rho+delta_rho)*vk < 1:
            s += delta_s
            rho += delta_rho
            U = L
        else:
            U = G
            U.remove(k)
    # Finish up
    theta = (s-1)/rho
    w = np.maximum(v-theta, 0)
    return w            
                