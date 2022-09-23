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
#np.random.seed(2)

def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    # Exact
    S2 = np.zeros([NoOfPaths, NoOfSteps+1])
    S2[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i] * dt + sigma(time[i]) * S1[:,i] * (W[:,i+1] - W[:,i])
        S2[:,i+1] = S2[:,i] * np.exp((r - 0.5*sigma(time[i])**2.0) * dt + sigma(time[i]) * (W[:,i+1] - W[:,i]))
        time[i+1] = time[i] + dt
        
    # Return S1 and S2
    paths = {"time":time,"S1":S1,"S2":S2}
    return paths

def GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    # Exact
    S2 = np.zeros([NoOfPaths, NoOfSteps+1])
    S2[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma(time[i]) * S1[:,i] * (W[:,i+1] - W[:,i]) \
                    + 0.5 * sigma(time[i])**2 * S1[:,i] * (np.power((W[:,i+1] - W[:,i]),2) - dt)
                    
        S2[:,i+1] = S2[:,i] * np.exp((r - 0.5*sigma(time[i])**2.0) *dt + sigma(time[i]) * (W[:,i+1] - W[:,i])) #REVISAR, NO ES RAIZ DE SIGMA?
        time[i+1] = time[i] +dt
        
    # Retun S1 and S2
    paths = {"time":time,"S1":S1,"S2":S2}
    return paths

def compute_convergence_vanilla(NoOfPaths, NoOfSteps, discretization_type):
    # Initialize parameters
    T = 4
    r = 0.05
    S_0 = 1
    K = 1.6

    # Time-dependent volatility specification
    sigma = lambda t: 0.6 - 0.2*exp(-1.5*t)

    # Compute discretized paths
    if discretization_type == 'euler':
        Paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    elif discretization_type == 'milstein':
        Paths = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S_0)

    timeGrid = Paths["time"]
    S1 = Paths["S1"]
    S2 = Paths["S2"]

    # Valuate options for a given payoff function
    S1_T = S1[:,-1]
    S2_T = S2[:,-1]

    def PayoffValuation(S,T,r,payoff):
    # S is a vector of Monte Carlo samples at T
        return np.exp(-r*T) * np.mean(payoff(S))

    payoff = lambda S: np.maximum(S-K,0.0)  
    val_t0_approx = PayoffValuation(S1_T,T,r,payoff)
    val_t0_exact = PayoffValuation(S2_T,T,r,payoff)

    error = 100 * (val_t0_approx - val_t0_exact) / (val_t0_approx + val_t0_exact)

    valuations = {'approx': val_t0_approx, 'exact': val_t0_exact, 'error': error}
    return valuations

def compute_convergence_barrier(NoOfPaths, NoOfSteps, discretization_type, barrier_type, option_type):
    # Initialize parameters
    T = 4
    r = 0.05
    S_0 = 1
    K = 1.6

    # Time-dependent volatility specification
    sigma = lambda t: 0.6 - 0.2*exp(-1.5*t)

    # Compute discretized paths
    if discretization_type == 'euler':
        Paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    elif discretization_type == 'milstein':
        Paths = GeneratePathsGBMMilstein(NoOfPaths,NoOfSteps,T,r,sigma,S_0)

    timeGrid = Paths["time"]
    S1 = Paths["S1"]
    S2 = Paths["S2"]

    # Valuate options for a given path-dependent payoff function (up-and-out and up-and-in barriers)
    def PayoffValuationPathDependent(S,T,r,payoff):
    # S is an array of Monte Carlo samples
        return np.exp(-r*T) * np.mean(np.apply_along_axis(payoff, 0, S))

    def PayoffValuationPathDependentStd(S,T,r,payoff):
    # S is an array of Monte Carlo samples
        return np.exp(-r*T) * np.std(np.apply_along_axis(payoff, 0, S)) / np.sqrt(S.shape[0])

    B = 1.5

    if barrier_type == 'up-and-out':
        if option_type == 'call':
            payoff_barrier = lambda S: np.maximum(S[-1] - K,0.0) if (S <= B).all() else 0
        elif option_type == 'put':
            payoff_barrier = lambda S: np.maximum(K - S[-1],0.0) if (S <= B).all() else 0
        val_t0_approx = PayoffValuationPathDependent(S1,T,r,payoff_barrier)
        err_t0_approx = PayoffValuationPathDependentStd(S1,T,r,payoff_barrier)
        val_t0_exact = PayoffValuationPathDependent(S2,T,r,payoff_barrier)
        if val_t0_approx + val_t0_exact != 0:
            error = 100 * (val_t0_approx - val_t0_exact) / (val_t0_approx + val_t0_exact)
        else:
            error = 0

    elif barrier_type == 'up-and-in':
        if option_type == 'call':  
            payoff_barrier = lambda S: np.maximum(S[-1] - K,0.0) if (S >= B).any() else 0
        elif option_type == 'put':
            payoff_barrier = lambda S: np.maximum(K - S[-1],0.0) if (S >= B).any() else 0
        val_t0_approx = PayoffValuationPathDependent(S1,T,r,payoff_barrier)
        err_t0_approx = PayoffValuationPathDependentStd(S1,T,r,payoff_barrier)
        val_t0_exact = PayoffValuationPathDependent(S2,T,r,payoff_barrier)
        if val_t0_approx + val_t0_exact != 0:
            error = 100 * (val_t0_approx - val_t0_exact) / (val_t0_approx + val_t0_exact)
        else:
            error = 0
    
    if math.isnan(error): error = 0
    valuations = {'val_approx': val_t0_approx, 'err_approx': err_t0_approx, 'val_exact': val_t0_exact, 'error': error}
    return valuations


###############################################################################


# # 1. Valuation of an European call option with Euler discretization

# paths = [1000, 10000, 100000]
# steps = [50, 100, 200, 400]

# error_euler = []
# for path in paths:
#     error_path = []
#     for step in steps:
#         results = compute_convergence_vanilla(path, step, discretization_type='euler')
#         error_path.append(results['error'])
#     error_euler.append(error_path)

# plt.figure(1)
# for err in error_euler:
#     plt.plot(steps, err)
# plt.xlabel("Amount of steps")
# plt.grid()
# plt.ylabel("Error between approximate and exact (%)")
# plt.title("Convergence for Euler method")
# plt.legend(paths, title="Paths")
# plt.show()

# # # 2. Valuation of an European call option with Milstein discretization

# error_euler = []
# for path in paths:
#     error_path = []
#     for step in steps:
#         results = compute_convergence_vanilla(path, step, discretization_type='milstein')
#         error_path.append(results['error'])
#     error_euler.append(error_path)

# plt.figure(1)
# for err in error_euler:
#     plt.plot(steps, err)
# plt.xlabel("Amount of steps")
# plt.ylabel("Error between approximate and exact (%)")
# plt.grid()
# plt.title("Convergence for Milstein method")
# plt.legend(paths, title="Paths")
# plt.show()

# # 3. Valuation of an up-and-out barrier option with Milstein discretization

# paths = [1000, 10000, 100000]
# steps = [50, 100, 200, 400]

# barrier_type = 'up-and-out'
# option_type = 'put'

# vals_upandout = []
# errs_upandout = []
# for path in paths:
#     print("Paths: {}".format(path))
#     vals_path = []
#     errs_path = []
#     for step in steps:
#         results = compute_convergence_barrier(path, step, discretization_type='milstein', barrier_type=barrier_type, option_type=option_type)
#         vals_path.append(results['val_approx'])
#         errs_path.append(results['err_approx'])
#     vals_upandout.append(vals_path)
#     errs_upandout.append(errs_path)

# plt.figure(1)
# for i in range(len(paths)):
#     plt.errorbar(steps, vals_upandout[i], yerr=errs_upandout[i], marker='.', capsize=5)
# plt.xlabel("Amount of steps")
# plt.ylabel("Value")
# plt.grid()
# plt.title("Valuation of an {} barrier {} option with the Milstein method".format(barrier_type, option_type))
# plt.legend(paths, title="Paths")
# plt.show()

# 4. Valuation of an up-and-in barrier option with Milstein discretization

paths = [1000, 10000, 100000]
steps = [50, 100, 200, 400]

barrier_type = 'up-and-in'
option_type = 'call'

vals_upandin = []
errs_upandin = []
for path in paths:
    print("Paths: {}".format(path))
    vals_path = []
    errs_path = []
    for step in steps:
        results = compute_convergence_barrier(path, step, discretization_type='milstein', barrier_type=barrier_type, option_type=option_type)
        vals_path.append(results['val_approx'])
        errs_path.append(results['err_approx'])
    vals_upandin.append(vals_path)
    errs_upandin.append(errs_path)

plt.figure(1)
for i in range(len(paths)):
    plt.errorbar(steps, vals_upandin[i], yerr=errs_upandin[i], marker='.', capsize=5)
plt.xlabel("Amount of steps")
plt.ylabel("Value")
plt.grid()
plt.title("Valuation of an {} barrier {} option with the Milstein method".format(barrier_type, option_type))
plt.legend(paths, title="Paths")
plt.show()



# def mainCalculation():
#     NoOfPaths = 1000
#     NoOfSteps = 50
#     T = 4
#     r = 0.05
#     sigma = lambda t: 0.6 - 0.2*exp(-1.5*t)
#     S_0 = 1
    
#     # Simulated paths
#     Paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
#     timeGrid = Paths["time"]
#     S1 = Paths["S1"]
#     S2 = Paths["S2"]
    
#     # plt.figure(1)
#     # plt.plot(timeGrid, np.transpose(S1),'k')   
#     # plt.plot(timeGrid, np.transpose(S2),'--r')   
#     # plt.grid()
#     # plt.xlabel("time")
#     # plt.ylabel("S(t)")
#     # plt.show()

#     # Payoff setting    
#     S1_T = S1[:,-1]
#     S2_T = S2[:,-1]
#     K  = 1.6
    
#     # Payoff specification
#     payoff = lambda S: np.maximum(S-K,0.0)  
        
#     # Valuation
#     val_t0_exact = PayoffValuation(S2_T,T,r,payoff)
#     print("Exact value of the contract at t0 ={0}".format(val_t0_exact))

#     val_t0_approx = PayoffValuation(S1_T,T,r,payoff)
#     print("Approximate value of the contract at t0 ={0}".format(val_t0_approx))
    
#     B = 1.5
#     payoff_barrier_upandout = lambda S: np.maximum(S[-1] - K,0.0) if (S <= B).all() else 0
#     valBarrierUpAndOut_t0 = PayoffValuationPathDependent(S1,T,r,payoff_barrier_upandout)
#     print("Value of the up-and-out barrier option at t0 ={0}".format(valBarrierUpAndOut_t0))

#     payoff_barrier_upandin = lambda S: np.maximum(S[-1] - K,0.0) if (S >= B).any() else 0
#     valBarrierUpNIn_t0 = PayoffValuation(S1,T,r,payoff_barrier_upandin)
#     print("Value of the up-and-in barrier option at t0 ={0}".format(valBarrierUpNIn_t0))

#     # # Weak and strong convergence
#     # NoOfStepsV = range(1,500,1)
#     # NoOfPaths = 250
#     # errorWeak = np.zeros([len(NoOfStepsV),1])
#     # errorStrong = np.zeros([len(NoOfStepsV),1])
#     # dtV = np.zeros([len(NoOfStepsV),1])
#     # for idx, NoOfSteps in enumerate(NoOfStepsV):
#     #     Paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
#     #     # Get the paths at T
#     #     S1_atT = Paths["S1"][:,-1]
#     #     S2_atT = Paths["S2"][:,-1]
        
#     #     errorWeak[idx] = np.abs(np.mean(S1_atT)-np.mean(S2_atT))
        
#     #     errorStrong[idx] = np.mean(np.abs(S1_atT-S2_atT))
#     #     dtV[idx] = T/NoOfSteps
        
#     # print(errorStrong)    
#     # plt.figure(2)
#     # plt.plot(dtV,errorWeak)
#     # plt.plot(dtV,errorStrong,'--r')
#     # plt.grid()
#     # plt.legend(['weak conv.','strong conv.'])
#     # plt.show()
     
# mainCalculation()