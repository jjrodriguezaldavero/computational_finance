"""
Exercise 1
Based On:
Integrated Brownian motion- three cases, W(t), and I(t)
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def ComputeIntegrals(NoOfPaths,NoOfSteps,T):    

    # Fixing random seed
    #np.random.seed(1)

    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    I1 = np.zeros([NoOfPaths, NoOfSteps+1])
    I2 = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        I1[:,i+1] = I1[:,i] + W[:,i]**5 *dt
        I2[:,i+1] = I2[:,i] + W[:,i]**6 *(W[:,i+1]-W[:,i])
        time[i+1] = time[i] + dt
        
    paths = {"time":time,"W":W,"I1":I1,"I2":I2}
    return paths

def main_calculation(NoOfPaths, NoOfSteps, T):
    W_t = ComputeIntegrals(NoOfPaths,NoOfSteps,T)

    W = W_t["W"]
    W7 = np.power(W, 7)
    intW5dt = W_t["I1"]
    intW6dWt = W_t["I2"]
    time = W_t["time"]

    W7_mean = np.apply_along_axis(np.mean, 0, W7)
    intW5dt_mean = np.apply_along_axis(np.mean, 0, intW5dt)
    intW6dWt_mean = np.apply_along_axis(np.mean, 0, intW6dWt)

    accuracy = (1/7) * W7_mean[-1] - (5/2) * intW5dt_mean[-1] - intW6dWt_mean[-1]
    return {"accuracy": accuracy, "W7_mean": W7_mean, "intW5dt_mean": intW5dt_mean, "intW6dWt_mean": intW6dWt_mean, "time": time}


###############################################################################

# Analysis of the asset paths
NoOfSteps = 10000
NoOfPaths = 1000
T = 0.5

results = main_calculation(NoOfPaths, NoOfSteps, T)
W7_mean = results["W7_mean"]
intW5dt_mean = results["intW5dt_mean"]
intW6dWt_mean = results["intW6dWt_mean"]
time = results["time"]

plt.figure(1)
plt.grid()
plt.plot(time, intW6dWt_mean)
plt.plot(time, (1/7) * W7_mean - (5/2)*intW5dt_mean)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Asset paths for the components of the integral")
plt.legend(["LHS", "RHS"])
plt.show()


# Analysis of the accuracy as a function of the Monte Carlo paths for 10000 steps:
NoOfSteps = 10000
T=0.5
accuracies = []
paths = np.linspace(100, 1000, 10)
for n_paths in paths:
    accuracy = main_calculation(int(n_paths), NoOfSteps, T)["accuracy"]
    accuracies.append(accuracy)

plt.plot(paths, accuracies)
plt.grid()
plt.xlabel("Number of paths")
plt.ylabel("Accuracy")
plt.title("Accuracy as a function of Monte Carlo paths")
plt.show()

#Analysis of the accuracy as a function of the number of steps for 1000 paths:
NoOfPaths = 1000
T=0.5
accuracies = []
steps = np.linspace(1000, 10000, 10)
for n_steps in steps:
    accuracy = main_calculation(NoOfPaths, int(n_steps), T)["accuracy"]
    accuracies.append(accuracy)

plt.plot(steps, accuracies)
plt.grid()
plt.xlabel("Number of steps")
plt.ylabel("Accuracy")
plt.title("Accuracy as a function of path steps")
plt.show()