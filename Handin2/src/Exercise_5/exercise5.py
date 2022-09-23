"""
Exercise 5
Based On:
The Heston model discretization, Euler scheme vs. AES scheme
@author: Lech A. Grzelak
"""

import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsHestonEuler(NoOfPaths,NoOfSteps,T,kappa,gamma,vbar,v0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        # Truncated boundary condition
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma* np.sqrt(V[:,i]) * (W[:,i+1]-W[:,i])
        V[:,i+1] = np.maximum(V[:,i+1],0.0)

        time[i+1] = time[i] +dt
        
    paths = {"time":time,"V":V}
    return paths

def CIR_Sample(NoOfPaths,kappa,gamma,vbar,s,t,v_s):
    delta = 4.0 *kappa*vbar/gamma/gamma
    c= 1.0/(4.0*kappa)*gamma*gamma*(1.0-np.exp(-kappa*(t-s)))
    kappaBar = 4.0*kappa*v_s*np.exp(-kappa*(t-s))/(gamma*gamma*(1.0-np.exp(-kappa*(t-s))))
    sample = c* np.random.noncentral_chisquare(delta,kappaBar,NoOfPaths)
    return  sample

def GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,kappa,gamma,vbar,v0):    
    V = np.zeros([NoOfPaths, NoOfSteps+1])
    V[:,0]=v0

    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # Exact samles for the variance process
        V[:,i+1] = CIR_Sample(NoOfPaths,kappa,gamma,vbar,0,dt,V[:,i])
        time[i+1] = time[i] +dt
        
    paths = {"time":time,"V":V}
    return paths

###############################################################################

# 1. Show that Euler discretization can reach negative V(t) values

NoOfSteps = 1000
NoOfPaths = 1
T = 10
v0 = 1
# Unsatisfied Feller condition: 2*kappa*vbar < gamma^2
kappa = 1
gamma = 2
vbar = 1

Paths = GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, kappa, gamma, vbar, v0)
timeGrid = Paths["time"]
V = Paths["V"]

plt.figure(1)
plt.plot(timeGrid, np.transpose(V))
plt.grid()
plt.title("CIR process with unsatisfied Feller condition")
plt.xlabel("time t")
plt.ylabel("CIR process v(t)")
plt.show()

# Satisfied Feller condition: 2*kappa*vbar > gamma^2
kappa = 2
gamma = 2
vbar = 2

Paths = GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, kappa, gamma, vbar, v0)
timeGrid = Paths["time"]
V = Paths["V"]

plt.figure(2)
plt.plot(timeGrid, np.transpose(V))
plt.grid()
plt.title("CIR process with satisfied Feller condition")
plt.xlabel("time t")
plt.ylabel("CIR process v(t)")
plt.show()

# 2. Show that exact simulation cannot reach negative values for the same parameter set

NoOfSteps = 1000
NoOfPaths = 1
T = 10
v0 = 1
# Unsatisfied Feller condition: 2*kappa*vbar < gamma^2
kappa = 1
gamma = 2
vbar = 1

Paths = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,kappa,gamma,vbar,v0)
timeGrid = Paths["time"]
V = Paths["V"]

plt.figure(3)
plt.plot(timeGrid, np.transpose(V))
plt.grid()
plt.title("CIR process with unsatisfied Feller condition")
plt.xlabel("time t")
plt.ylabel("CIR process v(t)")
plt.show()

# Satisfied Feller condition: 2*kappa*vbar > gamma^2
kappa = 2
gamma = 2
vbar = 2

Paths = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,kappa,gamma,vbar,v0)
timeGrid = Paths["time"]
V = Paths["V"]

plt.figure(4)
plt.plot(timeGrid, np.transpose(V))
plt.grid()
plt.title("CIR process with satisfied Feller condition")
plt.xlabel("time t")
plt.ylabel("CIR process v(t)")
plt.show()

# Now repeat the same giving expectations and variances
NoOfSteps = 1000
NoOfPaths = 1000
T = 10
v0 = 1
# Unsatisfied Feller condition: 2*kappa*vbar < gamma^2
kappa = 1
gamma = 2
vbar = 1

Paths = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,kappa,gamma,vbar,v0)
timeGrid = Paths["time"]
V = Paths["V"]

V_mean = np.apply_along_axis(np.mean, 0, V)
V_sde = np.apply_along_axis(np.std, 0, V) / np.sqrt(V.shape[0])

plt.figure(5)
plt.errorbar(timeGrid,np.transpose(V_mean),yerr=V_sde, capsize=5, errorevery=50)
plt.grid()
plt.title("CIR process with Unsatisfied Feller condition")
plt.xlabel("time t")
plt.ylabel("CIR process v(t)")
plt.fill_between(timeGrid, 
    [val-err for val,err in zip(V_mean, V_sde)], 
    [val+err for val,err in zip(V_mean, V_sde)], alpha=0.3)
plt.show()

# Greater step size 
NoOfSteps = 10000

Paths = GeneratePathsHestonAES(NoOfPaths,NoOfSteps,T,kappa,gamma,vbar,v0)
timeGrid = Paths["time"]
V = Paths["V"]

V_mean = np.apply_along_axis(np.mean, 0, V)
V_sde = np.apply_along_axis(np.std, 0, V) / np.sqrt(V.shape[0])

plt.figure(5)
plt.errorbar(timeGrid,np.transpose(V_mean),yerr=V_sde, capsize=5, errorevery=500)
plt.grid()
plt.title("CIR process with satisfied Feller condition")
plt.xlabel("time t")
plt.ylabel("CIR process v(t)")
plt.fill_between(timeGrid, 
    [val-err for val,err in zip(V_mean, V_sde)], 
    [val+err for val,err in zip(V_mean, V_sde)], alpha=0.3)
plt.show()