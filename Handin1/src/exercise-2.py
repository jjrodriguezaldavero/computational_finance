# Exercise 2.5

import numpy as np
import matplotlib.pyplot as plt

def integrate_sin(t, n):

    s = np.linspace(0, 5, n)
    ds = np.diff(s)

    W = np.sin(s)
    dW = np.diff(W)

    integral1 = np.sum(W[1:] * ds)
    integral2 = np.sum((t - s[1:]) * dW)

    print("integral 1: ", integral1)
    print("integral2: ", integral2)

    return integral1, integral2

integrate_sin(5, 1000)

iterations = np.linspace(100, 10000, 100)
accuracies = []
for i in iterations:
    accuracy = integrate_sin(5, int(i))[1] - integrate_sin(5, int(i))[0]
    accuracies.append(accuracy)
plt.scatter(iterations, accuracies)
plt.title("Accuracy of the identity for f(x) = sin(x)")
plt.ylabel("Accuracy difference")
plt.show()

def integrate_exponential(t, n):

    s = np.linspace(0, 5, n)
    ds = np.diff(s)

    W = np.exp(s)
    dW = np.diff(W)

    integral1 = np.sum(W[1:] * ds)
    integral2 = np.sum((t - s[1:]) * dW)

    print("integral 1: ", integral1)
    print("integral2: ", integral2)

    return integral1, integral2

integrate_exponential(5, 1000)

iterations = np.linspace(100, 10000, 100)
accuracies = []
for i in iterations:
    accuracy = integrate_exponential(5, int(i))[1] - integrate_exponential(5, int(i))[0]
    accuracies.append(accuracy)
plt.scatter(iterations, accuracies)
plt.title("Accuracy of the identity for f(x) = exp(x)")
plt.ylabel("Accuracy difference")
plt.show()