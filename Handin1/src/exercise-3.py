#%%
"""
Created on Thu Nov 27 2018
Paths for the GBM and ABM
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp

def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    
    # Fixing random seed
    np.random.seed(1)
        
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = S_0
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma **2 ) * dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
        
    #Compute exponent of ABM
    return X[0], time

def mainCalculation():
    NoOfPaths = 1
    NoOfSteps = 365*7
    T = 7
    K = np.linspace(0, 0.1, 11)

    alpha = 0.05
    beta = 0.1
    sigmax = 0.38
    sigmay = 0.15
    r = 0.06

    X, time = GeneratePathsGBM(NoOfPaths, NoOfSteps, T, alpha, sigmax, 0)
    Y, _ = GeneratePathsGBM(NoOfPaths, NoOfSteps, T, beta, sigmay, 0)
    M, _ = GeneratePathsGBM(NoOfPaths, NoOfSteps, T, r, 0, 1)

    V = np.zeros([K.shape[0], NoOfSteps+1])
    W = np.zeros([K.shape[0], NoOfSteps+1])
    for i in range(NoOfSteps):
        for j in range(len(K)):
            V[j, i+1] = exp(-r*(T-time[i]))/M[i] * max((0.5 * X[i] * exp((alpha+sigmax)*(T-time[i])) - 0.5 * Y[i] * exp((beta + sigmay)*(T-time[i])), K[j]))

    return V

K = np.linspace(0, 0.1, 11)
W = mainCalculation()
plt.figure(1)
plt.title("Option payout price")
plt.xlabel("Day / 7ys")
plt.ylabel("V(t)")
for i in range(len(K)):
    plt.plot(W[i])
plt.legend(K, title="K")
plt.show()