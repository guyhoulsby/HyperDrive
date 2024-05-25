import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [1.1,0.1], [1.1,0.1]])

file = "hnepmk_par"
name = "nD Linear Elastic - Plastic with Multisurface Kinematic Hardening - Parallel"
const = [2, 5.0, 4, 0.05, 50.0, 0.15, 30.0, 0.2, 10.0, 0.3, 5.0]
mu = 0.1

def deriv():
    global ndim, Einf, E0, k, H, recip_k, n_y, n_int, n_inp, n_const, name_const
    ndim = int(const[0])
    Einf = float(const[1])
    n_int = int(const[2])
    n_inp = int(const[2])
    n_y = int(const[2])
    n_const = 2 + 2*n_int
    k = np.array(const[3:3+2*n_inp:2])
    H = np.array(const[4:4+2*n_inp:2])
    recip_k = 1.0 / k
    E0 = Einf + sum(H)
    name_const = ["ndim", "Einf", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))
       
def f(eps,alp): return np.einsum("i,ij,ij->",H,(eps-alp),(eps-alp))/2.0 + Einf*np.einsum("i,i->",eps,eps)/2.0
def dfde(eps,alp): return np.einsum("i,ij->j",H,(eps-alp)) + Einf*eps
def dfda(eps,alp): return -np.einsum("i,ij->ij",H,(eps-alp))
def d2fdede(eps,alp): return np.eye(ndim)*E0
def d2fdeda(eps,alp):
    temp = np.zeros([ndim,n_int,ndim])
    for n in range(n_inp): 
        temp[:,n,:] = -H[n]*np.eye(ndim)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_inp): 
        temp[n,:,:] = -H[n]*np.eye(ndim)
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    for n in range(n_inp): 
        temp[n,:,n,:] = H[n]*np.eye(ndim)
    return temp

def g(sig,alp): return (-sum((sig + np.einsum("n,ni->i",H,alp))**2)/(2.0*E0) 
                         + np.einsum("n,ni,ni->",H,alp,alp)/2.0)
def dgds(sig,alp): return -(sig + np.einsum("n,ni->i",H,alp)) / E0
def dgda(sig,alp): return -np.einsum("i,n->ni",(sig + np.einsum("n,ni->i",H,alp)),H)/E0 + np.einsum("n,ni->ni",H,alp)
def d2gdsds(sig,alp): return -np.eye(ndim)/E0
def d2gdsda(sig,alp):
    temp = np.zeros([ndim,n_int,ndim])
    for n in range(n_inp): 
        temp[:,n,:] = -H[n]*np.eye(ndim) / E0
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    for n in range(n_inp):
        temp[n,:,:] = -H[n]*np.eye(ndim) / E0
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    for n in range(n_int):
        for m in range(n_int):
            temp[n,:,m,:] = -H[n]*H[m]*np.eye(ndim) / E0
        temp[n,:,n,:] += H[n]*np.eye(ndim)
    return temp

def y(eps,sig,alp,chi): 
    return np.sqrt(np.einsum("ni,ni->n",chi,chi))*recip_k - 1.0
def dydc(eps,sig,alp,chi): 
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_inp):
        t1 = hu.non_zero(np.sqrt(sum(chi[i,]*chi[i,])))
        temp[i,i,:] = (chi[i,:]/t1)*recip_k[i]
    return temp
def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyda(eps,sig,alp,chi): return np.zeros([n_y,n_int,ndim])