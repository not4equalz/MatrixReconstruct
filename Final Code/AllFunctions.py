# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:09:13 2023

@author: noahv
"""
import numpy as np
import math
def RandomRMatGen(S, m, sig2=0.2, mu=0):
    #S is number of species, m is number of fixed points you want to get
    A = np.random.normal(mu, sig2, (S, m))
    return A

def PositiveIndex(N):
    """
    Parameters
    ----------
    N : ARRAY OR LIST
        array containing a fixed point

    Returns The set of indices for which N[i] =/= 0
    -------
    """
    return [i for i, val in enumerate(N) if val != 0]
def bits1(I, n):
    """
    Convert an integer I to a list of binary digits with a length of n.
    Example: bits1(9, 6) returns [0, 0, 1, 0, 0, 1]
    """
    return [int(digit) for digit in f"{I:0{n}b}"]
def SteadyState(n, Ad, r=[]):
    """
    n is the total number of species,
    Ad is an already sampled random matrix with diagonal elements equal to 1
    Returns a matrix where each column is a feasible Steady State
    r is an array of parameters
    """
    Nfeasible = np.zeros((n, 2**n))

    # Iterate through binary vectors from 1 to 2^n-1
    for i in range(2**n - 1):
        binvec = bits1(i + 1, n)  # Convert the integer to binary vector length n
        sumvec = np.sum(binvec)  # Compute the sum of binary vector elements
        indices = np.where(np.array(binvec) == 0)[0]  # Find indices where binary vector is 0

        # Create a mask for the diagonal elements of the matrix
        mask = np.ones(Ad.shape[0], dtype=bool)
        mask[indices] = False

        # Extract submatrix A based on the mask
        A = Ad[np.ix_(mask, mask)]

        # Create a vector of ones with a length equal to the sum of binary vector elements
        if r==[]: 
            onevec = np.ones(sumvec)
            # Solve for Nshort using linear algebra
            Nshort = np.linalg.solve(A, onevec)
        else:
            Positive = PositiveIndex(binvec)
            p = [r[index] for index in Positive]
            Nshort = np.linalg.solve(A, p)
                

        # Initialize N with zeros and populate it based on the binary vector
        N = np.zeros(n)
        mask = np.array(binvec) == 1
        N[mask] = Nshort

        # Store N in the result matrix
        Nfeasible[:, i] = N

    # Remove non-feasible stationary points (where all N values are non-negative)
    mask = (Nfeasible >= 0).all(axis=0)
    Nfeasible = Nfeasible[:, mask]

    return Nfeasible
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
def Normalcdf(x, mu=0,sig2=1):
    """
    calculates the value of the cumulative distribution of a normal random
    variable at x with mean mu and variance sig2
    """
    error=math.erf((x-mu)/((2*sig2)**(0.5)))
    return (1+error)/2
def Matrixcdf(a, C, mu=0, sig2=1):
    
    """
    Calculates the value of the cdf of a matrix element with parameters
    C - connectivity
    mu - mean of normal distribution
    sig2 - variance of normal distribution
    """
    P=C*Normalcdf(a, mu, sig2)
    if a<0:
        return P
    else:
        return 1-C+P
def RemoveTrivialFixPoints(M):
    """
    Parameters
    ----------
    M : Matrix where each column is a Fixed Point

    Returns
    -------
    M without the trivial fixed points, i.e. (1, 0) or (0,0)
    """
    final=[]
    for j in range(np.size(M, axis=1)):
        for i in range(np.size(M, axis=0)):
            if M[i,j]==0 or M[i,j]==1:
                continue
            else:
                final.append(M[:,j])
                break
    return np.array(final).T
def Stability(A, N):
    """
    Finds stability of fixed point N given its matrix A'
    Returns 1 if stable and 0 if unstable
    """
    DA=np.diag(N)@A
    Eig=np.linalg.eig(DA)[0]
    for i in range(len(Eig)):
        if Eig[i]>0:
            return 1
        else:
            continue
    return 0
def rResponseFunc(N, B):
    """
    Parameters
    ----------
    N : ARRAY
        The fixed point
    B : TYPE
        The random matrix of the system

    Returns
    -------
    Returns the response function matrix for a given
    random matrix and one of the fixed points
    """
    n=len(N)
    #initialize response matrix
    nu=np.zeros((n,n))
    Nstar=PositiveIndex(N)
    #find matrix Bstar
    Bstar=B[np.ix_(Nstar, Nstar)]
    #calculate inverse
    BstarInv=np.linalg.inv(Bstar)
    #insert values into nu
    for i in range(len(Nstar)):
        for j in range(len(Nstar)):
            nu[Nstar[i], Nstar[j]]=BstarInv[i,j]
    return nu
def normalpdf(x, mu=0, sigma=1):
    exp=np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    return exp
def laplacepdf(x,mu=0,b=1):
    lpc=np.exp(-np.abs(x-mu)/b)/(2*b)
    return lpc

def DeterMatrixGen(S):
    M=np.identity(S)
    for i in range(S):
        for j in range(S):
            if i>j:
                M[i,j]= 1/S
            elif i<j:
                M[i,j]= -1/S
    return M