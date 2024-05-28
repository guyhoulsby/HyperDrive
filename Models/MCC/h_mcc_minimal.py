import autograd.numpy as np
from HyperDrive import Utils as hu

file = "h_mcc"
name = "Modified Cam Clay" # constant shear modulus variant
ndim = 2
n_int = 1
n_y = 1
#n_const = 6
names = ["t", "eps_v", "eps_s",   "p",   "q"]
units = ["s",     "-",     "-", "kPa", "kPa"]
name_const = [ "pr", "lambda*", "kappa*", "M",    "G", "pco"]
const =      [100.0,       0.2,     0.05, 1.0, 3000.0, 100.0]

def deriv():
    global pr, lambdas, kappas, M, G, pco
    pr = float(const[0])
    lambdas = float(const[1])
    kappas = float(const[2])
    M = float(const[3])
    G = float(const[4])
    pco = float(const[5])

def px(alp): return (pco/2.0) * np.exp(alp[0,0]/(lambdas - kappas))

def f(eps,alp): 
    return pr*kappas*np.exp((eps[0]-alp[0,0])/kappas) + (3.0*G/2.0)*((eps[1]-alp[0,1])**2) - pr*kappas

def g(sig,alp): 
    return -pr*kappas*hu.ilog(sig[0]/pr) - (sig[1]**2)/(6.0*G) - np.einsum("i,ni->",sig,alp)

def y(eps,sig,alp,chi): return np.array([(chi[0,0]-px(alp))**2 + (chi[0,1]/M)**2 - px(alp)**2])