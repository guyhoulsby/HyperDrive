import numpy as np
import HyperUtils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [1.1,0.1], [1.1,0.1]])

file = "h2epmk_nest"
name = "nD Linear Elastic - Plastic with Multisurface Kinematic Hardening - Nested"
mode = 1
const = [2, 100.0, 4, 0.1, 100.0, 0.2, 33.333333, 0.3, 20.0, 0.4, 10.0]
mu = 0.1

def deriv():
    global ndim, E, k, H, recip_k
    global n_y, n_int, n_inp, n_const, name_const
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

def alpdiff(alp):
    temp = np.zeros([n_inp-1,ndim])
    for i in range(n_inp-1): temp[i] = (alp[i]-alp[i+1]) 
    return temp
        
def f(eps,alp): 
    t1 = E*sum((eps-alp[0])**2)/2.0 
    t2 = np.einsum("n,ni,ni->",H[:n_inp-1],alpdiff(alp),alpdiff(alp))/2.0
    t3 = H[n_inp-1]*sum(alp[n_inp-1]**2)/2.0
    return (t1 + t2 + t3)
def dfde(eps,alp): return E*(eps-alp[0])
def dfda(eps,alp): 
    temp = np.zeros([n_int,ndim])
    temp[0,:] = -E*(eps-alp[0])
    temp[:n_inp-1,:] += np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[1:n_inp,:] -= np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[n_inp-1,:] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2fdede(eps,alp): return np.eye(ndim)*E
def d2fdeda(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -E*np.eye(ndim)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -E*np.eye(ndim)
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    temp[0,0] = E*np.eye(ndim)
    for i in range(n_inp-1):
        temp[i,i] += H[i]*np.eye(ndim)
        temp[i+1,i] -= H[i]*np.eye(ndim)
        temp[i,i+1] -= H[i]*np.eye(ndim)
        temp[i+1,i+1] += H[i]*np.eye(ndim)
    temp[n_inp-1,n_inp-1] += H[n_inp-1]*np.eye(ndim)
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
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -np.eye(ndim)
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -np.eye(ndim)
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    for i in range(n_inp-1):
        temp[i,i] += H[i]*np.eye(ndim)
        temp[i+1,i] -= H[i]*np.eye(ndim)
        temp[i,i+1] -= H[i]*np.eye(ndim)
        temp[i+1,i+1] += H[i]*np.eye(ndim)
    temp[n_inp-1,n_inp-1] += H[n_inp-1]*np.eye(ndim)
    return temp

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): 
    t1 = np.sqrt(np.einsum("ni,ni->n",chi,chi))
    return t1*recip_k - 1.0
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i,]**2)))
        temp[i,i,:] = (chi[i,:]/t1)*recip_k[i]
    return temp
def dyde_f(chi,eps,alp): return np.zeros([n_y,ndim])
def dyda_f(chi,eps,alp): return np.zeros([n_y,n_int,ndim])

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp):  
    t1 = np.sqrt(np.einsum("ni,ni->n",chi,chi))
    return t1*recip_k - 1.0
def dydc_g(chi,sig,alp):  
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_inp):
        t1 = hu.floor(np.sqrt(sum(chi[i,]**2)))
        temp[i,i,:] = (chi[i,:]/t1)*recip_k[i]
    return temp
def dyds_g(chi,sig,alp): return np.zeros([n_y,ndim])
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int,ndim])