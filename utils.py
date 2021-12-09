#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some utility functions
"""

import numpy as np

# Small function to uniformly sample from the n-dim simplex
def SampleSimplex(d):
    x = np.zeros(d+1)
    x[0:d-1] = np.random.rand(d-1)
    x[d] = 1
    y = np.sort(x)
    z = np.ediff1d(y) # difference between consecutive elements
    return z

def PowerMethod(A):
    '''
    Finds leading singular value of non-square matrix
    '''
    sing_val = 0
    change = 100
    m,n = np.shape(A)
    if m > n:
        A = A.T  # flip for speed
        
    m,n = np.shape(A)
    x = np.random.randn(m)
    while change >= 1e-6:
        y = A@(A.T@x)
        new_sing_val = np.dot(y,x)/np.linalg.norm(x)
        x = y/np.linalg.norm(y)
        change = abs(new_sing_val - sing_val)
        sing_val = new_sing_val
        
    return np.sqrt(sing_val)
  
def SphereProject(x):
    temp = x/np.linalg.norm(x)
    return temp

def DefineBallProject(alpha):
    '''
    Project on to the ball of radius alpha
    '''
    def BallProject(z):
        norm_val = np.linalg.norm(z)
        if norm_val < alpha:
            temp = z/norm_val
        else:
            temp = z
        return temp
    return BallProject


def LinMinOracle(u, x, active_set):
    '''
    This oracle returns argmin_{x in Simplex} <u,x>.
    For use with Pairwise Frank Wolfe.
    '''
    largest_coordinate = np.argmax(u)
    update_direction = np.zeros(len(u))
    update_direction[largest_coordinate] += 1
    fw_vertex_rep = largest_coordinate
    away_vertex_rep, max_step_size = min(active_set.items(),
                                         key=lambda item: u[item[0]])
    update_direction[away_vertex_rep] -= 1
    return update_direction, fw_vertex_rep, away_vertex_rep, max_step_size

def FindFirstLessThan(L, tol):
    '''
    Find the index of the first element of L less than tol
    '''
    for idx, val in enumerate(L):
        if val <= tol:
            return idx
    return idx  # if no entry in list is less than tol, return final index.