#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for Riemannian gradient descent on the unit sphere
"""

import numpy as np

def ProjTangSpace(z,xi):
    '''
    This function projects an n-dim vector xi onto the tangent space to the (n-1) sphere at x.
    Remove if statement for increased speed.
    '''
    
    if not np.abs(np.linalg.norm(z)-1.) <= 1e-4:
       raise ValueError('x needs to be on the unit sphere')
        
    temp = xi - np.dot(z,xi)*z
    return temp

def ExpSphere(z,xi,t):
    '''
    This function takes a step of length t along the geodesic starting from z
    in the direction of xi on the unit sphere.
    Remove if statements for increased speed.
    '''
    if not np.abs(np.linalg.norm(z)-1.) <= 1e-4:
       raise ValueError('x needs to be on the unit sphere')
        
    if not np.abs(np.dot(z,xi)) <= 1e-4:
       print(np.abs(np.dot(z,xi)))
       raise ValueError('xi needs to be in the tangent space at x')
        
    norm_xi = np.linalg.norm(xi)
    temp = np.cos(t*norm_xi)*z + (np.sin(t*norm_xi)/norm_xi)*xi
    return temp

def RetractionSphere(z, xi, t):
    '''
    This function takes a step t along the standard Retraction on the sphere,
    starting at x and in the direction of xi.
    '''
    if not np.abs(np.linalg.norm(z)-1.) <= 1e-4:
        raise ValueError('x needs to be on the unit sphere')
        
    if not np.abs(np.dot(z,xi)) <= 1e-4:
        raise ValueError('xi needs to be in the tangent space at x')
        
    temp = z + t*xi
    return temp/np.linalg.norm(temp)

def DRetraction(x, s):
   '''
   This function returns the Jacobian of the standard Retraction on the
   sphere based at x but evaluated at s.
   '''
   n = len(x)
   temp_norm = np.linalg.norm(x+s)
   return (np.identity(n) - (1/temp_norm**2)*np.outer(x+s, x+s))/temp_norm

def RGDStep(z, grad, step_size):
    '''
    This function takes a step of RGD, starting from z. grad needs to be the gradient of the function (in R^n, not 
    yet projected to the tangent space of the sphere). step_size is the step size.
    Remove checks for increased speed.
    '''
    if not np.abs(np.linalg.norm(z)-1.) <= 1e-4:
        raise ValueError('x needs to be on the unit sphere')
    
    RiemGrad = ProjTangSpace(z, grad)
    z_plus = ExpSphere(z, -RiemGrad, step_size)
    return z_plus


def ConstructLineFunction(z, xi, objfunc, objfunc_prime):
    '''
    Returns a function phi(t)  = f(Exp(x,xi,t)). This is a 1d function, and 
    so can be passed to the line search.
    '''
    norm_xi = np.linalg.norm(xi)
    def phi(t):
        return objfunc(ExpSphere(z,xi,t))
    
    def phi_prime(t):
        newvec = -norm_xi*np.sin(t*norm_xi)*z + np.cos(t*norm_xi)*xi
        newpt = ExpSphere(z,xi,t)
        fungrad = ProjTangSpace(newpt, objfunc_prime(newpt))
        return np.dot(newvec,fungrad)
    return phi, phi_prime


    
    
        
        
        