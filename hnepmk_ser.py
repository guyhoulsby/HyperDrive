import numpy as np
import HyperUtils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [0.1,0.1], [1.2,0.1]])

file = "hnepmk_ser"
name = "nD Linear Elastic-Plastic with Multisurface Kinematic Hardening - Series"
mode = 1
const = [2, 100.0, 4, 0.1, 100.0, 0.3, 33.333333, 0.6, 20.0, 1.0, 10.0]
eta = 0.04
tref = 1.0e6

def deriv():
    global ndim, E, k, recip_k, H, name_const
    global n_int, n_inp, n_y, n_const
    ndim = int(const[0])
    E = float(const[1])
    n_int = int(const[2])
    n_inp = int(const[2])
    n_y = int(const[2])
    n_const = 2 + 2*n_inp
    k = np.array(const[3:3+2*n_inp:2])
    H = np.array(const[4:4+2*n_inp:2])
    recip_k = 1.0 / k
    name_const = ["ndim", "E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))

deriv()
        
def ep(alp): return np.einsum("ni->i",alp)

def f(eps,alp): return E*sum((eps-ep(alp))**2)/2.0 + np.einsum("n,ni,ni->",H,alp,alp)/2.0
def dfde(eps,alp): return E*(eps-ep(alp))
def dfda(eps,alp): return -E*(eps-ep(alp)) + np.einsum("n,ni->ni",H,alp)
def d2fdede(eps,alp): return E*np.eye(ndim)
def d2fdeda(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[:] = -E*np.eye(ndim)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[:] = -E*np.eye(ndim)
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    temp[:,:] = E*np.eye(ndim)
    for n in range(n_inp): temp[n,n] += H[n]*np.eye(ndim)
    return temp

def g(sig,alp): return (-sum(sig**2)/(2.0*E) - sum(sig*ep(alp)) + 
                        np.einsum("n,ni,ni->",H,alp,alp)/2.0)
def dgds(sig,alp): return -sig/E - ep(alp)
def dgda(sig,alp): return -sig + np.einsum("n,ni->ni",H,alp)
def d2gdsds(sig,alp): return -np.eye(ndim)/E
def d2gdsda(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[:] = -np.eye(ndim)
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[:] = -np.eye(ndim)
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    for n in range(n_int): temp[n,n] = H[n]*np.eye(ndim)
    return temp

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): return np.sqrt(np.einsum("ni,ni->n",chi,chi))*recip_k - 1.0
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i]**2)))
        temp[i,i,:] = chi[i,:]*recip_k[i] / t1
    return temp
def dyde_f(chi,eps,alp): return np.zeros([n_y,ndim])
def dyda_f(chi,eps,alp): return np.zeros([n_y,n_int,ndim])

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): return np.sqrt(np.einsum("ni,ni->n",chi,chi))*recip_k - 1.0
def dydc_g(chi,sig,alp):  
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i]**2)))
        temp[i,i,:] = chi[i,:]*recip_k[i] / t1
    return temp
def dyds_g(chi,sig,alp): return np.zeros([n_y,ndim])
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int,ndim])

def w_f(chi,eps,alp): 
    temp = 0.0
    for i in range(n_inp):
        temp += (k[i]/tref) * hu.mac(np.sqrt(sum(chi[i]**2))*recip_k[i] - 1.0)**2 / (2.0*eta)
    return temp
def dwdc_f(chi,eps,alp): 
    temp = np.zeros([n_int,ndim])
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i]**2)))
        temp[i] = chi[i]*hu.mac(np.sqrt(sum(chi[i]**2))*recip_k[i] - 1.0) / (t1*eta*tref)
    return temp
def w_g(chi,sig,alp): 
    temp = 0.0
    for i in range(n_inp):
        temp += (k[i]/tref) * hu.mac(np.sqrt(sum(chi[i]**2))*recip_k[i] - 1.0)**2 / (2.0*eta)
    return temp
def dwdc_g(chi,sig,alp): 
    temp = np.zeros([n_int,ndim])
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i]**2)))
        temp[i] = chi[i]*hu.mac(np.sqrt(sum(chi[i]**2))*recip_k[i] - 1.0) / (t1*eta*tref)
    return temp
