"""
Exercise 2
Based On:
---
@author: Lech A. Grzelak
"""

import numpy as np
import matplotlib.pyplot as plt

def GeneratePaths(NoOfPaths,NoOfSteps, T):    
    
    # Fixing random seed
    #np.random.seed(1)
        
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    W[:,0] = 0
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] + dt
        
    for i in range(0, NoOfSteps+1):
        X[:,i] = W[:,i] - (time[i]/(3*T)) * W[:,int((T - time[i])/dt)]

    paths = {"time":time,"W":W,"X":X}
    return paths

def compute_accuracy(NoOfPaths, NoOfSteps, t, T):
    Paths = GeneratePaths(NoOfPaths,NoOfSteps, T)
    W = Paths["W"]
    X = Paths["X"]

    variance_W = np.var(W[:, int(NoOfSteps * t / T)])
    variance_X = np.var(X[:, int(NoOfSteps * t / T)])
    variance_X_analytical = t + np.power(t/(3*T), 2) * (T-t) - 2*t/(3*T) * min(t, T-t)

    accuracy = 100 * (variance_X - variance_X_analytical)/(variance_X + variance_X_analytical)
    return accuracy


###############################################################################

# Analysis of the accuracy as a function of t for T=15
NoOfPaths = 1000
NoOfSteps = 10000
T=15
accuracies = []
ts = np.linspace(0, 15, 16)
for t in ts:
    print("t = {}".format(t))
    accuracy = compute_accuracy(NoOfPaths, NoOfSteps, t, T)
    accuracies.append(accuracy)

plt.plot(ts, accuracies)
plt.grid()
plt.xlabel("t")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy as a function of t")
plt.show()

# Plot some paths for X(t)
NoOfPaths = 20
NoOfSteps = 10000
T = 15
Paths = GeneratePaths(NoOfPaths,NoOfSteps, T)
TimeGrid = Paths['time'][:-1]
X = Paths['X']

plt.plot(TimeGrid, np.transpose(X))
plt.grid()
plt.xlabel("t")
plt.ylabel("X(t)")
plt.title("Monte Carlo paths for X(t)")
plt.show()