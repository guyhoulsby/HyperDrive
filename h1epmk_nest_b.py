import numpy as np
import HyperUtils as hu

check_eps = 0.3
check_sig = 2.0
check_alp = np.array([0.2, 0.18, 0.16, 0.14])
check_chi = np.array([0.9, 1.0, 1.1, 1.2])

file = "h1epmk_nest_b"
name = "1D Linear Elastic-Plastic with Multisurface Kinematic Hardening - Nested Boundary"
mode = 0
ndim = 1
n_y = 1
const = [100.0, 4, 0.118965, 100.0, 0.313347, 33.333333, 0.659878, 20.0, 0.820951, 10.0]
mu = 0.1

def deriv():
    global E, k, H, recip_k, name_const, n_int, n_inp, n_const
    E = float(const[0])
    n_int = int(const[1])
    n_inp = int(const[1])
    n_const = 2 + 2*n_int
    k = np.array(const[2:2+2*n_inp:2])
    H = np.array(const[3:3+3*n_inp:2])
    recip_k = 1.0 / k
    name_const = ["E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))

deriv()

def alpdiff(alp): return np.array([(alp[i]-alp[i+1]) for i in range(n_inp-1)])

def f(eps,alp): return ((E*(eps-alp[0])**2)/2.0 + 
                         np.einsum("n,n,n->",H[:n_inp-1],alpdiff(alp),alpdiff(alp))/2.0 +
                         H[n_inp-1]*(alp[n_inp-1]**2)/2.0)
def dfde(eps,alp): return E*(eps-alp[0])
def dfda(eps,alp): 
    temp = np.zeros(n_int)
    temp[0] = -E*(eps-alp[0])
    temp[:n_inp-1] += H[:n_inp-1]*alpdiff(alp)
    temp[1:n_inp] -= H[:n_inp-1]*alpdiff(alp)
    temp[n_inp-1] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2fdede(eps,alp): return E
def d2fdeda(eps,alp): 
    temp = np.zeros(n_int)
    temp[0] = -E
    return temp
def d2fdade(eps,alp): 
    temp = np.zeros(n_int)
    temp[0] = -E
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,n_int])
    temp[0,0] = E
    for i in range(n_inp-1):
        temp[i,i] += H[i]
        temp[i+1,i] -= H[i]
        temp[i,i+1] -= H[i]
        temp[i+1,i+1] += H[i]
    temp[n_inp-1,n_inp-1] += H[n_inp-1]
    return temp

def g(sig,alp): return (-(sig**2)/(2.0*E) - sig*alp[0] + 
                        np.einsum("n,n,n->",H[:n_inp-1],alpdiff(alp),alpdiff(alp))/2.0 +
                        H[n_inp-1]*(alp[n_inp-1]**2)/2.0)
def dgds(sig,alp): return -sig/E - alp[0]
def dgda(sig,alp): 
    temp = np.zeros(n_int)
    temp[0] = -sig
    temp[:n_inp-1] += H[:n_inp-1]*alpdiff(alp)
    temp[1:n_inp] -= H[:n_inp-1]*alpdiff(alp)
    temp[n_inp-1] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2gdsds(sig,alp): return -1.0/E
def d2gdsda(sig,alp):
    temp = np.zeros(n_int)
    temp[0] = -1.0
    return temp
def d2gdads(sig,alp): 
    temp = np.zeros(n_int)
    temp[0] = -1.0
    return temp
def d2gdada(sig,alp): 
    temp = np.zeros([n_int,n_int])
    for i in range(n_inp-1):
        temp[i,i] += H[i]
        temp[i+1,i] -= H[i]
        temp[i,i+1] -= H[i]
        temp[i+1,i+1] += H[i]
    temp[n_inp-1,n_inp-1] += H[n_inp-1]
    return temp

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): 
    return np.array([np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)) - 1.0])
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int])
    t1 = hu.floor(np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)))
    for i in range(n_inp):
        temp[0,i] += (chi[i]/(k[i]**2)) / t1
    return temp
def dyde_f(chi,eps,alp): return np.zeros(n_y)
def dyda_f(chi,sig,alp): return np.zeros([n_y,n_int])

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): return np.array([np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)) - 1.0])
def dydc_g(chi,sig,alp): 
    temp = np.zeros([n_y,n_int])
    t1 = hu.floor(np.sqrt(np.einsum("n,n,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)))
    for i in range(n_inp):
        temp[0,i] += (chi[i]/(k[i]**2)) / t1
    return temp
def dyds_g(chi,sig,alp): return np.zeros(n_y)
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int])