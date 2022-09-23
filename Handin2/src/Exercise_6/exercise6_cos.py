#%%
"""
LogNormal density recovery using the COS method
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def COSDensity(cf,x,N,a,b):
    i = np.complex(0.0,1.0) #assigning i=sqrt(-1)
    k = np.linspace(0,N-1,N)
    u = np.zeros([1,N])
    u = k * np.pi / (b-a)
        
    #F_k coefficients
    F_k    = 2.0 / (b - a) * np.real(cf(u) * np.exp(-i * u * a));
    F_k[0] = F_k[0] * 0.5; # adjustment for the first term
    
    #Final calculation
    f_X = np.matmul(F_k , np.cos(np.outer(u, x - a )))
        
    # we output only the first row
    return f_X
    
def mainCalculation():


    i = np.complex(0.0, 1.0) #assigning i=sqrt(-1)
    
    # setting for the COS method 
    a = -10
    b = 10
    
    #define the range for the expansion points
    N = [16, 64, 128]
    
    # setting for normal distribution
    kappa = 1
    theta = 0.2
    r0 = theta
    T = 1
    t = 0
    gamma = 0.7
        
    # Define characteristic function for the normal distribution
    psi = lambda u: np.exp(1/kappa * (r0-theta) * (np.exp(-kappa*T) - np.exp(-kappa*t)) 
        - (T-t)*theta) * np.exp(i*u*(theta + (r0 - theta) * np.exp(-kappa * T)))
    A = lambda u: gamma**2 / (2*kappa**3) * (kappa * (T-t) - 2*(1 - np.exp(-kappa * (T-t))) + 0.5 * (1 - np.exp(-2*kappa*(T-t)))) - i*u*gamma**2/(2*kappa**2)*(1-np.exp(-kappa*(T - t)))**2 - u**2*gamma**2/(4*kappa)*(1-np.exp(-2*kappa*(T-t)))
    B = lambda u: i*u*np.exp(-kappa*(T-t)) - (1 - np.exp(-kappa*(T-t)))/kappa
    phi = lambda u: np.exp(A(u) + B(u))

    cF = lambda u : psi(u) * phi(u)
    # define domain for density
    y = np.linspace(0.05,5,1000)
        
    plt.figure(1)
    plt.grid()
    plt.xlabel("y")
    plt.ylabel("$f_Y(y)$")
    for n in N:
        f_Y = 1/y * COSDensity(cF,np.log(y),n,a,b)

        plt.plot(y,f_Y)
    plt.legend(["N=%.0f"%N[0],"N=%.0f"%N[1],"N=%.0f"%N[2]])
    plt.title("Recovered density function for the Vasicek model")
    plt.show()
    
    
mainCalculation()