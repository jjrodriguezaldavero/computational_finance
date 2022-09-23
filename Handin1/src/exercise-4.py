#%%
"""
Created on Thu Nov 27 2018
Paths for the GBM and ABM
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt


def GeneratePaths(NoOfPaths,NoOfSteps, T):    
    
    # Fixing random seed
    np.random.seed(1)
        
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps])
    time = np.zeros([NoOfSteps+1])
        
    W[:,0] = 0
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
        
    for i in range(0, NoOfSteps):
        t = time[i]
        X[:,i] = W[:,i] - (t/T) * W[:,int((T - t)/dt)]


    paths = {"time":time,"W":W,"X":X}
    return paths

def mainCalculation():
    NoOfPaths = 100
    NoOfSteps = 10000
    t = 5
    T = 10
    
    Paths = GeneratePaths(NoOfPaths,NoOfSteps, T)
    timeGrid = Paths["time"]
    W = Paths["W"]
    X = Paths["X"]
    
    # plt.figure(1)
    # plt.plot(timeGrid, np.transpose(W))   
    # plt.grid()
    # plt.xlabel("time")
    # plt.ylabel("W(t)")
    # plt.title("Wiener process W(t)")
    # plt.show()

    # plt.figure(2)
    # plt.plot(timeGrid[1:], np.transpose(X))   
    # plt.grid()
    # plt.xlabel("time")
    # plt.ylabel("X(t)")
    # plt.title("Modified stochastic process X(t)")
    # plt.show()
    
    t_step = NoOfSteps * t / T
    variances_W = np.apply_along_axis(np.var, 1, Paths['W'])
    variance_W = np.mean(variances_W)
    variances_X = 2*np.apply_along_axis(np.var, 1, Paths['X'][:, 0:int(t_step)])
    variance_X = np.mean(variances_X)
    variance_analytical = (t**2) * (10-t)/100 + t*abs(t-5)/5
    print("W variance: {}".format(variance_W))
    print("X variance: {}".format(variance_X))
    print("Analytical X variance: {}".format(variance_analytical))

    # steps = np.linspace(1000, 20000, 41)
    # variances = np.zeros(41)
    # for i in range(len(steps)):
    #     Paths = GeneratePaths(500,int(steps[i]), T)
    #     X = Paths["X"]
    #     t_step = NoOfSteps * t / T
    #     variances[i] = np.mean(2*np.apply_along_axis(np.var, 1, Paths['X'][:, 0:int(t_step)]))

    # plt.figure(3)
    # plt.plot(steps, variances)
    # plt.grid()
    # plt.xlabel("Amount of steps")
    # plt.ylabel("Variance of X(t)")
    # plt.title("Increase in variance accuracy as a function of steps")
    # plt.show()

mainCalculation()