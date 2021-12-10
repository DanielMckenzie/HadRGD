#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from baseOptimizer import BaseOptimizer
from Riemannian_utils import *

def RGD(obj_func, obj_func_grad, step_size, x0, num_iters, tol):
    '''
    No-frills implementation of Riemannian gradient descent on the sphere.
    Mainly for timing purposes, RGDStep is recommended for most applications.
    NB: This uses retraction, not geodesics.
    '''
    x = x0
    err = obj_func(x)
    ii = 0
    
    while err > tol and ii <= num_iters:
        grad = obj_func_grad(x)
        RiemGrad = grad - np.dot(x,grad)*x
        xtemp = x - step_size*RiemGrad
        x = xtemp/np.linalg.norm(xtemp)
        err = obj_func(x)
        ii += 1
    
    return err, ii

class PRGD(BaseOptimizer):
    '''
    Class for Perturbed Riemannian gradient descent on the sphere, as described
    in Criscitiello and Boumal 'Efficiently escaping saddle points on manifolds'
    NeurIPS 2019.
    '''
    def __init__(self, objfunc, objfuncGrad, z0, eta, r, cal_T, eps, b):
        self._objfunc = objfunc
        self._objfuncGrad = objfuncGrad
        self._z = z0
        self._n = len(z0)
        self._eta = eta
        self._r = r
        self._cal_T = cal_T
        self._eps = eps
        self._b = b  
        self.Num_Times_Tangent_Steps_Called = 0
        self._num_steps = 0
               
    def TangentSpaceSteps(self, s):
        for j in range(self._cal_T):
            temp_point = RetractionSphere(self._z, s, 1)  # This may need a fix.
            grad = self._objfuncGrad(temp_point)
            RiemGrad = ProjTangSpace(temp_point, grad) # ProjTangSpace(self._z, grad)
            DR = DRetraction(self._z, s) # slow as it requires a full matrix, at least as currently implemented.
            f_hat_grad = np.dot(DR.T, RiemGrad)
            s_temp = s - self._eta*f_hat_grad
            alpha = 0.9
            while np.linalg.norm(s_temp) > self._b:
                s_temp = s - alpha*self._eta*f_hat_grad
                alpha = alpha/2
            s = ProjTangSpace(self._z, s_temp)
        return RetractionSphere(self._z, s, 1)
    
    def step(self):
        grad = self._objfuncGrad(self._z)
        RiemGrad = ProjTangSpace(self._z, grad)
        print(np.linalg.norm(RiemGrad))
        if np.linalg.norm(RiemGrad) > self._eps:
            self._z = RGDStep(self._z, grad, self._eta)
            #self._z = RetractionSphere(self._z, -RiemGrad, self._eta)
            self._num_steps +=1 
        else:
            xi_temp = np.random.randn(self._n)
            xi = ProjTangSpace(self._z, self._r*xi_temp/np.linalg.norm(xi_temp))
            self._z = self.TangentSpaceSteps(self._eta*xi)
            self._num_steps += self._cal_T
            self.Num_Times_Tangent_Steps_Called +=1
            
        return self._objfunc(self._z)


class HadRGD_AW(BaseOptimizer):
    '''
    Hadamard Riemannian Gradient Descent with Armijo-Wolfe backtracking line
    search
    '''
    
    def __init__(self, objfunc, objfuncgrad, z0, default_step_size,
                 rho1=10**(-4), rho2=0.9, beta=0.75):
        self._objfunc = objfunc
        self._objfuncgrad = objfuncgrad
        self._z = z0
        self._default_step_size = default_step_size
        self._rho1 = rho1
        self._rho2 = rho2
        self._beta = beta

    def BackTrackingLineSearch(self, phi, phi_prime):
        Armijo_Cond_Satisfied = False
        Wolfe_Cond_Satisfied = False
        beta = self._beta
        m=0
        if phi_prime(0) >= 0:
            # pass
            raise ValueError('Not a descent direction!')
        while ((not Armijo_Cond_Satisfied) or (not Wolfe_Cond_Satisfied)) and (m<=25):
            alpha = self._default_step_size*(beta**m)
            m += 1
            if phi(alpha) <= phi(0) + self._rho1*alpha*phi_prime(0):
                Armijo_Cond_Satisfied = True
            if phi_prime(alpha) >= self._rho2*phi_prime(0):
                Wolfe_Cond_Satisfied = True
            if Armijo_Cond_Satisfied and Wolfe_Cond_Satisfied:
                pass
        print('number of AW iterations= ' + str(m-1))
        return alpha
        
    def step(self):
        '''
        Riemannian gradient descent with line search, using the Armijo-Wolfe
        step-size condition.
        
        z ................. Base point on sphere
        grad .............. gradient of f(z) (not yet Riemannian gradient)
        '''
        grad = self._objfuncgrad(self._z)
        RiemGrad = ProjTangSpace(self._z, grad)
        phi, phi_prime = ConstructLineFunction(self._z, -RiemGrad,
                                               self._objfunc, self._objfuncgrad)
        step_size = self.BackTrackingLineSearch(phi, phi_prime)
        z_plus = ExpSphere(self._z, -RiemGrad, step_size)
        self._z = z_plus
        return self._objfunc(z_plus)




class HadRGD_BB(BaseOptimizer):
    '''
    Class for Barzilei-Borwein on the sphere with non-monotone line
    search.
    '''
    
    def __init__(self, objfunc, objfuncGrad, alpha_def, rho1, beta, eta, z0):
        '''
        Parameters as in Wen & Yin: A feasible method for optimization with 
        orthogonality constraints.
        '''
        self._objfunc = objfunc
        self._objfuncGrad = objfuncGrad
        self._alpha = alpha_def
        self._z = z0
        self._rho1 = rho1
        self._beta = beta
        self._eta = eta
        self._k = 0 # iterate counter
        self._fVals = [self._objfunc(self._z)]
        self._iterates = [self._z]
        self._gradients = [ProjTangSpace(self._z, self._objfuncGrad(self._z))]
        self._C_list = [1] # used in optimality condition for non-monotone line search
        self._Q_list = [1] # used in optimality condition for non-monotone line search
        
    def GetBBStepSize(self):
        s_k = self._iterates[self._k] - self._iterates[self._k-1]
        y_k = self._gradients[self._k] - self._gradients[self._k-1]
        top = np.dot(s_k,s_k)
        bottom = abs(np.dot(s_k,y_k))
        alpha_BB = top/bottom
        return alpha_BB
    
    def LineSearch(self, z, xi):
        '''
        Non-monotone line search.
        z ............ Base point (almost always current iterate)
        xi ........... Search direction
        '''
        f_prime_0 = np.dot(ProjTangSpace(z, self._objfuncGrad(z)), xi)
        if f_prime_0 > 0:
            raise ValueError("Not a descent direction!")
    
        alpha_k = self._alpha
        f_yk_alpha = self._objfunc(ExpSphere(z, xi, alpha_k))
        C = self._C_list[self._k]
        max_iter = 25
        m = 0
        while f_yk_alpha >= C + self._rho1*alpha_k*f_prime_0 and m <= max_iter:
            alpha_k = self._beta*alpha_k
            m +=1
            
        return alpha_k
    
    def Update_Q(self):
        Qnew = self._eta*self._Q_list[self._k] + 1
        self._Q_list.append(Qnew)
        
    def Update_C(self):
        Cnew1 = self._eta*(self._Q_list[self._k])*(self._C_list[self._k])
        Cnew2 = self._objfunc(self._iterates[self._k+1])
        Cnew3 = self._Q_list[self._k+1]
        Cnew = (Cnew1 + Cnew2)/Cnew3
        self._C_list.append(Cnew)
        
    def step(self):
        '''
        Take a single step.
        '''
        z_k = self._iterates[self._k]
        grad = ProjTangSpace(z_k, self._objfuncGrad(z_k))
        alpha_now = self.LineSearch(z_k, -grad)
        z_k_plus_1 = ExpSphere(z_k, -grad, alpha_now)
        self._iterates.append(z_k_plus_1)
        newGrad = ProjTangSpace(z_k_plus_1, self._objfuncGrad(z_k_plus_1))
        self._gradients.append(newGrad)
        self._z = z_k_plus_1
        tempval = self._objfunc(z_k_plus_1)
        self._fVals.append(tempval)
        
        # Update various parameters
        
        self.Update_Q()
        self.Update_C()
        self._k +=1
        alpha_BB = self.GetBBStepSize()
        self._alpha = np.maximum(np.minimum(alpha_BB, 30.0), 1e-10)
        
        return tempval