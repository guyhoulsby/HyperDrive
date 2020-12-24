import numpy as np
import HyperUtils as hu

check_eps = 0.3
check_sig = 2.0
check_alp = np.array([0.2, 0.18, 0.16, 0.14, 0.01])
check_chi = np.array([0.3, 0.4, 1.1, 1.2, 1.0])

file = "h1epmk_ser_cbh"
name = "1D Linear Elastic - Plastic with Multisurface Kinematic Hardening - Series HARM"
mode = 0
ndim = 1
n_y = 1
const = [100.0, 4, 0.154932, 100.0, 0.436529, 33.33333, 1.653963, 20.0, 1.169596, 10.0, 0.1]
mu = 0.1

def deriv():
    global E, k, recip_k, H, R, name_const, n_int, n_inp, n_const
    E = float(const[0])
    n_int = int(const[1]) + 1
    n_inp = int(const[1])
    n_const = 2 + 2*n_int + 1
    k = np.array(const[2:2 + 2*n_inp:2])
    H = np.array(const[3:3 + 2*n_inp:2])
    R = float(const[2 + 2*n_inp])
    recip_k = 1.0 / k
    name_const = ["E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))
    name_const.append("R")

deriv()
    
def ep(alp): return sum(alp)

def f(eps,alp): return E*((eps-ep(alp))**2)/2.0 + np.einsum("n,n,n->",H,alp[:n_inp],alp[:n_inp])/2.0
def dfde(eps,alp): return E*(eps-ep(alp))
def dfda(eps,alp): 
    temp = -E*(eps-ep(alp))*np.ones(n_int)
    temp[:n_inp] += np.einsum("n,n->n",H,alp[:n_inp])
    return temp
def d2fdede(eps,alp): return E
def d2fdeda(eps,alp): return -E*np.ones(n_int)
def d2fdade(eps,alp): return -E*np.ones(n_int)
def d2fdada(eps,alp):
    temp = E*np.ones([n_int,n_int])
    for i in range(n_inp): temp[i,i] += H[i]
    return temp

def g(sig,alp): return -(sig**2)/(2.0*E) - sig*ep(alp) + np.einsum("n,n,n->",H,alp[:n_inp],alp[:n_inp])/2.0
def dgds(sig,alp): return -sig/E - ep(alp)
def dgda(sig,alp):
    temp = -sig*np.ones(n_int)
    temp[:n_inp] += np.einsum("n,n->n",H,alp[:n_inp])
    return temp
def d2gdsds(sig,alp): return -1.0/E
def d2gdsda(sig,alp): return -np.ones(n_int)
def d2gdads(sig,alp): return -np.ones(n_int)
def d2gdada(sig,alp):
    temp = np.zeros([n_int,n_int])
    for i in range(n_inp): temp[i,i] += H[i]
    return temp

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): 
    s = dfde(eps,alp)
    return np.array([np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)) - 1.0 + 
                    R*(np.abs(chi[n_inp]) - np.abs(s))])
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int])
    t1 = hu.floor(np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)))
    for i in range(n_inp):
        temp[0,i] += (chi[i]/(k[i]**2)) / t1
    temp[0,n_inp] += R*hu.S(chi[n_inp])
    return temp
def dyde_f(chi,eps,alp): 
    s = dfde(eps,alp)
    return -R*E*hu.S(s)*np.ones(n_y)
def dyda_f(chi,eps,alp):
    s = dfde(eps,alp)
    return R*E*hu.S(s)*np.ones([n_y,n_int])

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): return np.array([np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)) - 1.0 + 
                                      R*(np.abs(chi[n_inp]) - np.abs(sig))])
def dydc_g(chi,sig,alp): 
    temp = np.zeros([n_y,n_int])
    t1 = hu.floor(np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)))
    for i in range(n_inp):
        temp[0,i] += (chi[i]/(k[i]**2)) / t1
    temp[0,n_inp] += R*hu.S(chi[n_inp])
    return temp
def dyds_g(chi,sig,alp): return -R*hu.S(sig)*np.ones(n_y)
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int])

#def w_f(chi,eps,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_f(chi,eps,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
#def w_g(chi,sig,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_g(chi,sig,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
