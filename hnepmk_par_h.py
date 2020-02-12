import numpy as np
import HyperUtils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [1.1,0.1], [1.2,0.1], [1.2,0.1]])

file = "hnepmk_par_h"
name = "nD Linear Elastic-Plastic with Multisurface Kinematic Hardening - Parallel HARM"
mode = 1
const = [2, 5.0, 4, 0.050616, 50.0, 0.152501, 30.0, 0.207961, 10.0, 0.305214, 5.0, 0.1]
mu = 0.1

def deriv():
    global E0, Einf, k, recip_k, H, R, name_const, n_int, n_inp, n_y, n_const, ndim
    ndim = int(const[0])
    Einf = float(const[1])
    n_int = int(const[2]) + 1
    n_inp = int(const[2])
    n_y = int(const[2])
    n_const = 3 + 2*n_int + 1
    k = np.array(const[3:3 + 2*n_inp:2])
    H = np.array(const[4:4 + 2*n_inp:2])
    R = float(const[3 + 2*n_inp])
    recip_k = np.array(1.0 / k)
    E0 = Einf + sum(H)
    name_const = ["ndim", "E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))
    name_const.append("R")
    
deriv()

def ep(alp): return np.einsum("ni->i",alp)

def f(eps,alp): 
    temp = eps-alp[:n_inp]-alp[n_inp]
    return np.einsum("i,ij,ij->",H,temp,temp)/2.0 + Einf*sum((eps-alp[n_inp])**2)/2.0
def dfde(eps,alp): return np.einsum("i,ij->j",H,(eps-alp[:n_inp]-alp[n_inp])) + Einf*(eps-alp[n_inp])
def dfda(eps,alp): 
    temp = np.zeros([n_int,ndim])
    temp[:n_inp,:] -= np.einsum("i,ij->ij",H,(eps-alp[:n_inp]-alp[n_inp]))
    temp[n_inp,:] -= np.einsum("i,ij->j",H,eps-alp[:n_inp]-alp[n_inp])
    temp[n_inp,:] -= Einf*(eps-alp[n_inp])
    return temp
def d2fdede(eps,alp): return np.eye(ndim)*E0
def d2fdeda(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_inp): temp[n] = -np.eye(ndim)*H[n]
    temp[n_inp] = -np.eye(ndim)*E0
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_inp): temp[n] = -np.eye(ndim)*H[n]
    temp[n_inp] = -np.eye(ndim)*E0
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    for i in range(n_inp):
        temp[i,i] = np.eye(ndim)*H[i]
        temp[n_inp,i] += np.eye(ndim)*H[i]
        temp[i,n_inp] += np.eye(ndim)*H[i]
    temp[n_inp,n_inp] = np.eye(ndim)*E0
    return temp

def g(sig,alp): 
    temp = sig + np.einsum("n,ni->i",H,alp[:n_inp])
    return -sum(temp**2)/(2.0*E0) - sum(sig*alp[n_inp]) + np.einsum("n,ni,ni->",H,alp[:n_inp],alp[:n_inp])/2.0
def dgds(sig,alp): return -(sig + np.einsum("n,ni->i",H,alp[:n_inp])) / E0 - alp[n_inp]
def dgda(sig,alp): 
    temp = np.zeros([n_int,ndim])
    temp[:n_inp,:] -= np.einsum("i,n->ni",(sig + np.einsum("n,ni->i",H,alp[:n_inp])),H)/E0 
    temp[:n_inp,:] += np.einsum("n,ni->ni",H,alp[:n_inp])
    temp[n_inp] = -sig
    return temp
def d2gdsds(sig,alp): return -np.eye(ndim)/E0
def d2gdsda(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_inp): temp[n] = -np.eye(ndim)*H[n]/E0
    temp[n_inp] = -np.eye(ndim)
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_inp): temp[n] = -np.eye(ndim)*H[n]/E0
    temp[n_inp] = -np.eye(ndim)
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    for n in range(n_inp):
        for m in range(n_inp): temp[n,m] = -H[n]*H[m]*np.eye(ndim)/E0
        temp[n,n] += H[n]*np.eye(ndim)
    return temp

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): 
    s = dfde(eps,alp)
    t1 = np.sqrt(np.einsum("ni,ni->n",chi[:n_inp],chi[:n_inp]))
    t2 = np.sqrt(sum(chi[n_inp]**2))
    t3 = np.sqrt(sum(s**2))
    return t1*recip_k - 1.0 + R*(t2 - t3)
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int,ndim])
    t2 = hu.floor(np.sqrt(sum(chi[n_inp]*chi[n_inp])))
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i,]*chi[i,])))
        temp[i,i,:] = (chi[i,:]/t1)*recip_k[i]
        temp[i,n_inp,:] = R*chi[n_inp]/t2
    return temp
def dyde_f(chi,eps,alp): 
    s = dfde(eps,alp)
    temp = np.zeros([n_y,ndim])
    t3 = hu.floor(np.sqrt(sum(s*s)))
    for i in range(n_y): temp[i] = -R*E0*s/t3
    return temp
def dyda_f(chi,eps,alp):
    s = dfde(eps,alp)
    temp = np.zeros([n_y,n_int,ndim])
    t3 = hu.floor(np.sqrt(sum(s**2)))
    for i in range(n_y):
        for j in range(n_inp): temp[i,j] = R*H[j]*s/t3
        temp[i,n_inp] = R*E0*s/t3
    return temp

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): 
    t1 = np.sqrt(np.einsum("ni,ni->n",chi[:n_inp],chi[:n_inp]))
    t2 = np.sqrt(sum(chi[n_inp]**2))
    t3 = np.sqrt(sum(sig**2))
    return np.einsum("n,n->n",t1,recip_k) - 1.0 + R*(t2 - t3)
def dydc_g(chi,sig,alp): 
    temp = np.zeros([n_y,n_int,ndim])
    t2 = hu.floor(np.sqrt(sum(chi[n_inp]**2)))
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i,]**2)))
        temp[i,i,:] = (chi[i,:]/t1)*recip_k[i]
        temp[i,n_inp,:] = R*chi[n_inp]/t2
    return temp
def dyds_g(chi,sig,alp):
    temp = np.zeros([n_y,ndim])
    t3 = hu.floor(np.sqrt(sum(sig**2)))
    for i in range(n_y): temp[i] = -R*sig/t3
    return temp
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int,ndim])

#def w_f(chi,eps,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_f(chi,eps,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
#def w_g(chi,sig,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_g(chi,sig,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
