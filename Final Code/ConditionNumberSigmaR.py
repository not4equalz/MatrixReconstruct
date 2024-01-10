# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:56:14 2024

@author: noahv
"""

import numpy as np
import AllFunctions as AF
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#seed
np.random.seed(1)

#Extinction or not? True for Extinction, False for no Extinction
Extinction = False

#The following function adds the ones back to the vector of B.
#After this is done you can reshape the new vector into a S by S matrix.
def addones(w, J):
    w2=w
    for i in J:
        w2=np.insert(w2, i, 1)
    return w2
def norm(A):
    return np.linalg.norm(A, ord='fro')
#Vectorization of matrix M. Stacks all the columns of M on top of each other.
def vec(M):
    V=np.matrix.flatten(M.T)
    return V


#number of oringinal species in system
S=20

#number of fixed points that will be measured (x-axis in the plot)
m=S-1

#Random matrix generation, small sigma to ensure no extinction
if Extinction == True:
    B=AF.RandMatGen(S, 1, diag=1, mu=0, sig2=1/(S**(1)))
else:
    B=AF.RandMatGen(S, 1, diag=1, mu=0, sig2=1/(S**(1.5)))      



Sig2R = np.linspace(0.01, S/2, num=100)
ConditionNumbers = []


for Sig in Sig2R:
    #Parameters
    R=np.abs(AF.RandomRMatGen(S, m, sig2=Sig, mu=S))

    #initialize matrix to be filled with fixed points
    M=np.zeros((S,m))


    #find fixed point N for each column in R and put in M
    for i in range(m):
        r=R[:,i]
        def dNdt(N,t):
            D=np.diag(N) 
            G=r-B@N
            return D@G
        #initial condition and timescale
        Nbegin=r
        t=np.linspace(0,1000,num=10000)
        
        #solve system
        Sol = odeint(dNdt, y0=Nbegin, t=t)
        M[:,i]=Sol[-1,:]

    #Set values smaller dan 10^-8 in M to zero <- only happens at extinction

    for i in range(S):
        for j in range(m):
            if np.abs(M[i, j])<1e-2 :
                M[i,j]=0


    #Find rows that have all zeros:
    ZeroRows = [i for i in range(S) if sum(M[i, :]) <= 1e-2]

    #generate random noise and add to measurements

    Noise=np.random.normal(0, 0.001, (S,m))
    #M+=Noise

    #remove columns rows and vector elements 
    NewB = np.delete(B, ZeroRows, axis=0)
    NewB = np.delete(NewB, ZeroRows, axis=1)

    NewM = np.delete(M, ZeroRows, axis=0)
    NewR = np.delete(R, ZeroRows, axis=0)

    #the actual dimension of the problem S-k
    RealDim = S-len(ZeroRows)
    #the set with indices to be removed later from matrix X
    J=[i for i in range(RealDim**2) if i%(RealDim+1)==0]
    
    X = np.kron(np.identity(RealDim), NewM.T) 
    X = np.delete(X, J, axis=1)
    #add condition number to array
    ConditionNumbers.append(np.linalg.cond(X))

#%% Plot
plt.figure(dpi=300)
plt.plot(np.sqrt(Sig2R), ConditionNumbers)
plt.xlabel('variance ' + r'$\sigma _R$')
plt.ylabel('Condition Number ' + r"$\kappa (X)$")
plt.yscale('log')
