"""
Exercise 3
Based On:
Euler discretization of the GBM
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math
from math import exp

# Fixing random seed
np.random.seed(1)

def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,rho,sigma,kappa,theta,gamma,S_0,R_0):    
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z2 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    W2 = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Asset
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    S[:,0] =S_0
    # Interest rate
    R = np.zeros([NoOfPaths, NoOfSteps+1])
    R[:,0] =R_0
    # Discounted interest rate to present value
    R_discounted = np.ones([NoOfPaths])
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])

        # Correlate noises
        Z2[:,i]= rho * Z1[:,i] + np.sqrt(1.0 - rho**2) * Z2[:,i]

        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        
        R[:,i+1] = R[:,i] + kappa * (theta - R[:, i]) * dt + gamma * (W1[:,i+1] - W1[:,i])
        R_discounted = R_discounted * (1 + R[:, i]/(NoOfSteps))

        S[:,i+1] = S[:,i] + R[:, i] * S[:,i] * dt + sigma * S[:,i] * (W2[:,i+1] - W2[:,i])
        time[i+1] = time[i] + dt
        
    # Return S and R
    paths = {"time":time,"S":S,"R":R, "R_discounted": R_discounted}
    return paths

def compute_convergence_vanilla(NoOfPaths, NoOfSteps, kappa):
    # Initialize parameters

    rho = -0.5
    theta = 0.2
    gamma = 0.7
    sigma = 0.5
    S_0 = 1
    R_0 = theta
    T = 1

    # Compute discretized paths
    Paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,rho,sigma,kappa,theta,gamma,S_0,R_0)

    timeGrid = Paths["time"]
    S = Paths["S"]
    R_discounted = Paths["R_discounted"]

    # Valuate options for a given payoff function
    S_T = S[:,-1]

    def PayoffValuation(S,R_discounted,payoff):
    # S and T are vectors of Monte Carlo samples at T

        return np.mean(R_discounted * payoff(S))
    
    def PayoffValuationStd(S,R_discounted,payoff):
    # S and T are vectors of Monte Carlo samples at T

        return np.std(R_discounted * payoff(S)) / np.sqrt(S.shape)

    # Put option payoff
    K = 1.1
    payoff = lambda S: np.maximum(K-S,0.0)  
    val_t0 = PayoffValuation(S_T, R_discounted, payoff)
    err_t0 = PayoffValuationStd(S_T, R_discounted, payoff)[0]
    return val_t0, err_t0


###############################################################################


#1. Valuation of an European call option with Euler discretization
steps = 1000
paths = np.linspace(1000, 10000, 10, dtype=int)
#steps = [10000]
kappas = [1, 0.2]

values = []
errors = []
for kappa in kappas:
    values_path = []
    errors_path = []
    for path in paths:
        val_t0, err_t0 = compute_convergence_vanilla(path, steps, kappa)
        values_path.append(val_t0)
        errors_path.append(err_t0)
    values.append(values_path)
    errors.append(errors_path)

plt.figure(1)
for i in range(len(kappas)):
    plt.errorbar(paths, values[i], yerr=errors[i], marker='.', capsize=5)
plt.xlabel("Number of paths")
plt.ylabel("Price of the European put option")
plt.title("Pricing of a Put option under the BSV model with Euler Monte Carlo")
plt.legend(kappas, title="kappas")
plt.show()