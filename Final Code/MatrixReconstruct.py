# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:26:33 2024

@author: noahv
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#seed
np.random.seed(1)

#Extinction or not? True for Extinction, False for no Extinction
Extinction = True

#The following function adds the ones back to the vector of B.
#After this is done you can reshape the new vector into a S by S matrix.

def addones(w, J):
    #This function is used as 
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
def RandomRMatGen(S, m, sig2=0.2, mu=0):
    #S is number of species, m is number of fixed points you want to get
    A = np.random.normal(mu, sig2, (S, m))
    return A
def RandMatGen(n, C, diag=0, mu=0, sig2=1, sym=False):
    """
    Parameters
    ----------
    n : Integer
        Dimension of generated matrix
    C : Probability
        Probability that element is sampled from normal distribution
    diag : float, optional
        Each diagonal element will equal to this number. The default is 0.
    mu : float, optional
        mean of normal distribution. The default is 0.
    sig2 : float>0, optional
        variance of normal distribution. The default is 1.

    This function returns a random matrix of size nxn

    """
    
    A = np.random.normal(mu, np.sqrt(sig2), (n,n))
    CMat = np.reshape(np.random.binomial(1, C, size=n**2), (n,n))
    A = np.multiply(CMat, A)
    for i in range(n):
        A[i, i] = diag
        if sym==True:
            for j in np.arange(i,n):
                A[j,i]=A[i,j]
    return A
#number of oringinal species in system
S=20

#Ridge Coefficients that will show in plot
alphas=[0.001, 0.01, .1]

#number of fixed points that will be measured (x-axis in the plot)
n=np.arange(2, S+60)

#how many fixed points to find as a maximum
m=np.max(n)

#the reconstruction errors of linear regression 
#will be stored here (||B-B_m^*||^2)
errors=[]



#Random matrix generation, small sigma to ensure no extinction
if Extinction == True:
    B=RandMatGen(S, 1, diag=1, mu=0, sig2=1/(S**(1)))
else:
    B=RandMatGen(S, 1, diag=1, mu=0, sig2=1/(S**(1.5)))      


#Parameters
R=np.abs(RandomRMatGen(S, m, sig2=.02, mu=S))

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


#the errors of ridge regression will go in here. each row are the errors
#for a different ridge constant
ridgeerrors=np.zeros((len(alphas), len(n)))

TrueM = np.linalg.inv(NewB) @ NewR[:, :RealDim -1]


#condition numbers
conds=[]
#%%

#Linear Regression
def LinReg(M, R):
    D = RealDim
    X = np.kron(np.identity(D), M.T) 
    X = np.delete(X, J, axis=1)
    conds.append(np.linalg.cond(X))
    y = np.array(vec(R.T - M.T))
    Coeff = np.linalg.pinv(X)@y
    BRecon = np.reshape(addones(Coeff, J), (D,D))
    return BRecon

def RidgeReg(M, R, alpha):
    D = RealDim
    X = np.kron(np.identity(D), M.T) 
    X = np.delete(X, J, axis=1)
    y = np.array(vec(R.T - M.T))
    Coeff = np.linalg.inv(X.T @ X + alpha*np.identity(D*(D-1))) @X.T @y
    BRecon = np.reshape(addones(Coeff, J), (D,D))
    return BRecon


for val in n:
    BRecLin=LinReg(NewM[: , :val], NewR[: , :val])
    errors.append(np.linalg.norm(NewB-BRecLin)**2 / 2)


for val in range(len(n)):
    for i in range(len(alphas)):
        BRecRidge = RidgeReg(NewM[:,:n[val]], NewR[:,:n[val]], alphas[i])
        ridgeerrors[i, val] = np.linalg.norm(NewB-BRecRidge)**2 /(2)    


#%%  Plot of Reconstruction Error against number of fixed points m
plt.figure(dpi=300)
plt.plot(n, errors, label='Linear Regression')
plt.xlabel('Number of fixed points')
plt.ylabel('Error value ' + r'$\|B-B_m^*\|_F^2$')
for i in range(len(alphas)):
    plt.plot(n, ridgeerrors[i,:], label='Ridge ' + r'$\alpha =$' + str(alphas[i]))
plt.ylim(0, 10)
plt.vlines(x=[RealDim-1], ymin=0, ymax=10, color='grey', ls=':', lw=2, label=r'$m = S_{out} -1$')
plt.legend()


#%%             Plot of Condition number of X against m 
plt.figure(dpi=300)
plt.plot(n, conds)
plt.xlabel('number of fixed points ' + r'$m$')
plt.ylabel('Condition number ' + r'$\kappa (X)$')
plt.vlines(x=[RealDim-1], ymin=0, ymax=max(conds), color='grey', ls=':', lw=2, label=r'$m = S_{out} -1$')
plt.legend()
plt.yscale('log')
