import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [1.1,0.1], [1.2,0.1]])

file = "hnepmk_ser_b"
name = "nD Linear Elastic - Plastic with Multisurface Kinematic Hardening - Series Boundary"
const = [2, 100.0, 4, 0.152589, 100.0, 0.432023, 33.333333, 1.615338, 20.0, 1.142307, 10.0]
mu = 0.1
r = 0.1

def deriv():
    global ndim, E, k, recip_k, H, name_const
    global n_int, n_inp, n_y, n_const
    ndim = int(const[0])
    E = float(const[1])
    n_int = int(const[2])
    n_inp = int(const[2])
    n_y = 1
    n_const = 2 + 2*n_inp
    k = np.array(const[3:3+2*n_inp:2])
    H = np.array(const[4:4+2*n_inp:2])
    recip_k = 1.0 / k
    name_const = ["ndim", "E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))

def ep(alp): return np.einsum("ni->i",alp)

def f(eps,alp): return E*sum((eps-ep(alp))**2)/2.0 + np.einsum("n,ni,ni->",H,alp,alp)/2.0
def dfde(eps,alp): return E*(eps-ep(alp))
def dfda(eps,alp): return -E*(eps-ep(alp)) + np.einsum("n,ni->ni",H,alp)
def d2fdede(eps,alp): 
    return E*np.eye(ndim)
def d2fdeda(eps,alp):
    temp = np.zeros([ndim,n_int,ndim])
    for n in range(n_int):
        temp[:,n,:] = -E*np.eye(ndim)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_int):
        temp[n,:,:] = -E*np.eye(ndim)
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    for i in range(ndim):
        temp[:,i,:,i] = E
    for n in range(n_inp): 
        temp[n,:,n,:] += H[n]*np.eye(ndim)
    return temp

def g(sig,alp): return (-sum(sig**2)/(2.0*E) - sum(sig*ep(alp)) + 
                        np.einsum("n,ni,ni->",H,alp,alp)/2.0)
def dgds(sig,alp): return -sig/E - ep(alp)
def dgda(sig,alp): return -sig + np.einsum("n,ni->ni",H,alp)
def d2gdsds(sig,alp): return -np.diag(np.ones(ndim))/E
def d2gdsda(sig,alp):
    temp = np.zeros([ndim,n_int,ndim])
    for i in range(n_int):
        for j in range(ndim):
           temp[j,i,j] = -1.0
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for i in range(n_int):
        for j in range(ndim):
           temp[i,j,j] = -1.0
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    for n in range(n_inp): 
        for j in range(ndim):
            temp[n,j,n,j] = H[n]
    return temp

def y(eps,sig,alp,chi): 
    return np.array([np.sqrt(np.einsum("ni,ni,n,n->",chi,chi,recip_k,recip_k)) - 1.0])
def dydc(eps,sig,alp,chi): 
    temp = hu.non_zero(np.sqrt(np.einsum("ni,ni,n,n->",chi,chi,recip_k,recip_k)))
    return np.array([np.einsum("ni,n,n->ni",chi,recip_k,recip_k)/temp])
def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyda(eps,sig,alp,chi): return np.zeros([n_y,n_int,ndim])