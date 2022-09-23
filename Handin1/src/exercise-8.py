#%%
"""
Created on Thu Nov 27 2018
Paths for the GBM and ABM
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt


def GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    
    # Fixing random seed
    np.random.seed(1)
        
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
        
    X[:,0] = np.log(S_0)
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma **2 ) * dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
        
    #Compute exponent of ABM
    S = np.exp(X)
    paths = {"time":time,"X":X,"S":S}
    return paths

def mainCalculation():
    NoOfPaths = 10
    NoOfSteps = 300
    T = 3
    r = 0.05
    sigma = 0.4
    S_0 = 0.7
    
    Paths = GeneratePathsGBMABM(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
    
    Xsum = np.apply_along_axis(np.square, 1, X)
    Xsum = np.apply_along_axis(np.cumsum, 1, Xsum)
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    plt.show()
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(Xsum))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("Xsum(t)")
    plt.show()

#mainCalculation()

import yfinance as yf
import matplotlib.pyplot as plt

def get_returns(ticker):
    stock = yf.Ticker(ticker)

    hist = stock.history(period="1y")
    #days = hist.index.values
    prices = hist["Close"]

    returns = prices.shift(1) / prices - 1

    return (hist, returns)

(hist1, returns1) = get_returns("MSFT")
(hist2, returns2) = get_returns("WEAT")

Xsum1 = np.cumsum(np.square(returns1))
Xsum2 = np.cumsum(np.square(returns2))

days = hist1.index.values

plt.figure(1)
plt.plot(days, hist1["Close"], '.-')
plt.title("Asset path for MSFT")
plt.ylabel("Sum of square increments")
plt.show()

plt.figure(2)
plt.plot(days, Xsum1, '.-')
plt.title("Running sum of square increments for MSFT")
plt.ylabel("Sum of square increments")
plt.show()

plt.figure(3)
plt.plot(days, hist2["Close"], '.-')
plt.title("Asset path for WEAT")
plt.ylabel("Sum of square increments")
plt.show()

plt.figure(4)
plt.plot(days, Xsum2, '.-')
plt.title("Running sum of square increments for WEAT")
plt.ylabel("Sum of square increments")
plt.show()