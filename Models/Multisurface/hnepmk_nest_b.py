import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [1.1,0.1], [1.1,0.1]])

file = "h2epmk_nest_b"
name = "nD Linear Elastic-Plastic with Multisurface Kinematic Hardening - Nested Bounding"
mode = 1
const = [2, 100.0, 4, 0.118965, 100.0, 0.313347, 33.333333, 0.659878, 20.0, 0.820951, 10.0]
mu = 0.1

def deriv():
    global ndim, E, k, H, recip_k, n_y, n_int, n_inp, n_const, name_const
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

def alpdiff(alp):
    temp = np.zeros([n_inp-1,ndim])
    temp = alp[:n_inp-1] - alp[1:n_inp]
    return temp
        
def f(eps,alp): 
    temp  = E*sum((eps - alp[0])**2) / 2.0 
    temp += np.einsum("n,ni,ni->",H[:n_inp-1],alpdiff(alp),alpdiff(alp)) / 2.0
    temp += H[n_inp-1]*sum(alp[n_inp-1]**2) / 2.0
    return temp
def dfde(eps,alp): return E*(eps - alp[0])
def dfda(eps,alp): 
    temp = np.zeros([n_int,ndim])
    temp[0,:] = -E*(eps-alp[0])
    temp[0:n_inp-1,:] += np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[1:n_inp,  :] -= np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[n_inp-1,  :] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2fdede(eps,alp): return E*np.eye(ndim)
def d2fdeda(eps,alp):
    temp = np.zeros([ndim,n_int,ndim])
    temp[:,0,:] = -E*np.eye(ndim)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for i in range(ndim):
        temp[0,i,i] = -E
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    temp[0,:,0,:] = E*np.eye(ndim)
    for n in range(n_inp-1):
        for i in range(ndim):
            temp[n,i,  n,i] += H[n]
            temp[n+1,i,n,i] -= H[n]
            temp[n,i,  n+1,i] -= H[n]
            temp[n+1,i,n+1,i] += H[n]
    for i in range(ndim):
        temp[n_inp-1,i,n_inp-1,i] += H[n_inp-1]
    return temp

def g(sig,alp): return (-sum(sig**2)/(2.0*E) - sum(sig*alp[0]) +
                        np.einsum("n,ni,ni->",H[:n_inp-1],alpdiff(alp),alpdiff(alp))/2.0 +
                        H[n_inp-1]*sum(alp[n_inp-1]**2)/2.0)
def dgds(sig,alp): return -sig/E - alp[0]
def dgda(sig,alp): 
    temp = np.zeros([n_int,ndim])
    temp[0,:] = -sig
    temp[:n_inp-1,:] += np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[1:n_inp,:] -= np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[n_inp-1,:] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2gdsds(sig,alp): return -np.eye(ndim)/E
def d2gdsda(sig,alp):
    temp = np.zeros([ndim,n_int,ndim])
    temp[:,0,:] = -np.eye(ndim)
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0,:,:] = -np.eye(ndim)
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    for n in range(n_inp-1):
        temp[n,  :,n,  :] += H[n]*np.eye(ndim)
        temp[n+1,:,n,  :] -= H[n]*np.eye(ndim)
        temp[n,  :,n+1,:] -= H[n]*np.eye(ndim)
        temp[n+1,:,n+1,:] += H[n]*np.eye(ndim)
    temp[n_inp-1,:,n_inp-1,:] += H[n_inp-1]*np.eye(ndim)
    return temp

def y(eps,sig,alp,chi): 
    return np.array([np.sqrt(np.einsum("ni,ni,n,n->",chi,chi,recip_k,recip_k)) - 1.0])
def dydc(eps,sig,alp,chi): 
    temp = hu.non_zero(np.sqrt(np.einsum("ni,ni,n,n->",chi,chi,recip_k,recip_k)))
    return np.array([np.einsum("ni,n,n->ni",chi,recip_k,recip_k)/temp])
def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyda(eps,sig,alp,chi): return np.zeros([n_y,n_int,ndim])