import autograd.numpy as np
from HyperDrive import Utils as hu

file = "Mises_minimal"
name = "von Mises model - minimal coding version"
ndim = 6
n_y = 1
n_int = 1
n_inp = 1
const = [150.0, 100.0, 1.0]
name_const = ["K", "G", "k"]

def deriv():
    global K, G, k
    K = float(const[0])
    G = float(const[1])
    k = float(const[2])

def g(sig,alp): 
    return -(1.0/(18.0*K))*hu.i1sq_m(sig) - (1.0/(2.0*G))*hu.j2_m(sig) - sig @ alp[0]

def y(eps,sig,alp,chi): return np.array([hu.j2_m(chi[0]) - (k**2)])