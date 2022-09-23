"""
Exercise 4
Based On:
Correlated Brownian motions
@author: Lech A. Grzelak
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho,sigma1,sigma2,r,S1_0,S2_0):    
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z2 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W1 = np.zeros([NoOfPaths, NoOfSteps+1])
    W2 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] = S1_0
    S2 = np.zeros([NoOfPaths, NoOfSteps+1])
    S2[:,0] = S2_0
    
    dt = T / float(NoOfSteps)
    time = np.zeros([NoOfSteps+1])
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])
        
        # Correlate noises
        Z2[:,i]= rho * Z1[:,i] + np.sqrt(1.0 - rho**2) * Z2[:,i]
        
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        
        S1[:,i+1] = S1[:,i] * np.exp((r - 0.5*sigma1**2.0) * dt + sigma1 * (W1[:,i+1] - W1[:,i]))
        S2[:,i+1] = S2[:,i] * np.exp((r - 0.5*sigma1**2.0) * dt + sigma2 * (W2[:,i+1] - W2[:,i]))

        time[i+1] = time[i] +dt
        
    #Store the results
    paths = {"time":time,"W1":W1,"W2":W2,"S1":S1,"S2":S2}
    return paths

def valuate_option(NoOfPaths, NoOfSteps, T, rho):
    # Initialize asset parameters
    sigma1 = 0.4
    sigma2 = 0.15
    r = 0.01
    S1_0 = 1
    S2_0 = 1

    # Define discounted payoff function
    def PayoffValuation(S1,S2,T,r,payoff):
        return np.exp(-r*T) * np.mean(payoff(S1,S2))

    payoff = lambda S1,S2: np.maximum(S1,S2) 

    # Generate correlated Monte Carlo paths
    Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho,sigma1,sigma2,r,S1_0,S2_0)
    timeGrid = Paths["time"]
    S1 = Paths["S1"]
    S2 = Paths["S2"]

    # Valuate options for the given payoff function
    S1_T = S1[:,-1]
    S2_T = S2[:,-1]

    val_t0 = PayoffValuation(S1_T,S2_T,T,r,payoff)

    def PayoffValuationStd(S1,S2,T,r,payoff):
        return np.exp(-r*T) * np.std(payoff(S1,S2)) / np.sqrt(np.size(S1))

    std_err = PayoffValuationStd(S1_T,S2_T,T,r,payoff)

    valuation = {'val':val_t0, 'std_err':std_err}
    return valuation


###############################################################################

NoOfPaths = 1000
NoOfSteps = 2000
maturities = np.linspace(0,10,21)

# 1. Valuate options for positive correlation rho=0.9
valuations = []
standard_errors = []
for T in maturities:
    print("Maturity {}".format(T))
    valuation = valuate_option(NoOfPaths, NoOfSteps, T, rho=0.9)
    valuations.append(valuation['val'])
    standard_errors.append(valuation['std_err'])
plt.errorbar(maturities,valuations,yerr=standard_errors, marker='.', capsize=5)
plt.grid()
plt.xlabel('Maturity T')
plt.ylabel('Valuation at t0')
plt.fill_between(maturities, 
    [val-err for val,err in zip(valuations, standard_errors)], 
    [val+err for val,err in zip(valuations, standard_errors)], alpha=0.2)
plt.title('Valuation of the action for positively correlated assets')
plt.show()


# 2. Valuate options for negative correlation rho=-0.9
valuations = []
standard_errors = []
for T in maturities:
    print("Maturity {}".format(T))
    valuation = valuate_option(NoOfPaths, NoOfSteps, T, rho=-0.9)
    valuations.append(valuation['val'])
    standard_errors.append(valuation['std_err'])
plt.errorbar(maturities,valuations,yerr=standard_errors, marker='.', capsize=5)
plt.grid()
plt.xlabel('Maturity T')
plt.ylabel('Valuation at t0')
plt.fill_between(maturities, 
    [val-err for val,err in zip(valuations, standard_errors)], 
    [val+err for val,err in zip(valuations, standard_errors)], alpha=0.2)
plt.title('Valuation of the action for negatively correlated assets')
plt.show()


# def mainCalculation():
#     NoOfPaths = 1
#     NoOfSteps = 500
#     T = 1.0
    
#     ############### Negative correlation ######################
#     rho =-0.9
#     Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)
#     timeGrid = Paths["time"]
#     W1 = Paths["W1"]
#     W2 = Paths["W2"]
    
#     plt.figure(1)
#     plt.plot(timeGrid, np.transpose(W1))   
#     plt.plot(timeGrid, np.transpose(W2))   
#     plt.grid()
#     plt.xlabel("time")
#     plt.ylabel("W(t)")
#     plt.show()

#     ############### Positive correlation ######################
#     rho =0.9
#     Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)
#     timeGrid = Paths["time"]
#     W1 = Paths["W1"]
#     W2 = Paths["W2"]
    
#     plt.figure(2)
#     plt.plot(timeGrid, np.transpose(W1))   
#     plt.plot(timeGrid, np.transpose(W2))   
#     plt.grid()
#     plt.xlabel("time")
#     plt.ylabel("W(t)")
#     plt.show()

#     ############### Zero correlation ######################
#     rho =0.0
#     Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)
#     timeGrid = Paths["time"]
#     W1 = Paths["W1"]
#     W2 = Paths["W2"]
    
#     plt.figure(3)
#     plt.plot(timeGrid, np.transpose(W1))   
#     plt.plot(timeGrid, np.transpose(W2))   
#     plt.grid()
#     plt.xlabel("time")
#     plt.ylabel("W(t)")
#     plt.show()
    
# mainCalculation()