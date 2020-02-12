import numpy as np
import HyperUtils as hu

check_eps = np.array([0.3,0.05])
check_sig = np.array([8.0,0.5])
check_alp = np.array([[0.2,0.1], [0.18,0.1], [0.16,0.1], [0.14,0.1], [0.14,0.1]])
check_chi = np.array([[0.9,0.1], [1.0,0.1], [1.1,0.1], [1.2,0.1], [1.2,0.1]])

file = "hnepmk_nest_cbh"
name = "nD Linear Elastic-Plastic with Multisurface Kinematic Hardening - Nested Bounding HARM"
mode = 1
const = [2, 100.0, 4, 0.11955, 100.0, 0.317097, 33.33333, 0.679352, 20.0, 0.848504, 10.0, 0.1]
mu = 0.1

def deriv():
    global E, k, recip_k, H, R, name_const, n_int, n_inp, n_y, n_const, ndim
    ndim = int(const[0])
    E = float(const[1])
    n_int = int(const[2]) + 1
    n_inp = int(const[2])
    n_y = 1
    n_const = 3 + 2*n_int + 1
    k = np.array(const[3:3 + 2*n_inp:2])
    H = np.array(const[4:4 + 2*n_inp:2])
    recip_k = np.array(1.0 / k)
    R = float(const[3 + 2*n_inp])
    name_const = ["ndim", "E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))
    name_const.append("R")
    
deriv()

def alpdiff(alp):
    temp = np.zeros([n_inp-1,ndim])
    for i in range(n_inp-1): temp[i] = (alp[i]-alp[i+1]) 
    return temp
        
def f(eps,alp): 
    t1 = E*sum((eps-alp[0]-alp[n_inp])**2)/2.0 
    t2 = np.einsum("n,ni,ni->",H[:n_inp-1],alpdiff(alp),alpdiff(alp))/2.0
    t3 = H[n_inp-1]*sum(alp[n_inp-1]**2)/2.0
    return (t1 + t2 + t3)
def dfde(eps,alp): return E*(eps-alp[0]-alp[n_inp])
def dfda(eps,alp): 
    temp = np.zeros([n_int,ndim])
    temp[0,:] = -E*(eps-alp[0]-alp[n_inp])
    temp[n_inp,:] = -E*(eps-alp[0]-alp[n_inp])
    temp[:n_inp-1,:] += np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[1:n_inp,:] -= np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[n_inp-1,:] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2fdede(eps,alp): return np.eye(ndim)*E
def d2fdeda(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -E*np.eye(ndim)
    temp[n_inp] = -E*np.eye(ndim)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -E*np.eye(ndim)
    temp[n_inp] = -E*np.eye(ndim)
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,n_int,ndim,ndim])
    temp[0,0] = E*np.eye(ndim)
    temp[0,n_inp] = E*np.eye(ndim)
    temp[n_inp,0] = E*np.eye(ndim)
    temp[n_inp,n_inp] = E*np.eye(ndim)
    for i in range(n_inp-1):
        temp[i,i] += H[i]*np.eye(ndim)
        temp[i+1,i] -= H[i]*np.eye(ndim)
        temp[i,i+1] -= H[i]*np.eye(ndim)
        temp[i+1,i+1] += H[i]*np.eye(ndim)
    temp[n_inp-1,n_inp-1] += H[n_inp-1]*np.eye(ndim)
    return temp

def g(sig,alp): return (-sum(sig**2)/(2.0*E) - sum(sig*(alp[0] + alp[n_inp])) +
                        np.einsum("n,ni,ni->",H[:n_inp-1],alpdiff(alp),alpdiff(alp))/2.0 +
                        H[n_inp-1]*sum(alp[n_inp-1]**2)/2.0)
def dgds(sig,alp): return -sig/E - alp[0] - alp[n_inp]
def dgda(sig,alp): 
    temp = np.zeros([n_int,ndim])
    temp[0,:] = -sig
    temp[n_inp,:] = -sig
    temp[:n_inp-1,:] += np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[1:n_inp,:] -= np.einsum("n,ni->ni",H[:n_inp-1],alpdiff(alp))
    temp[n_inp-1,:] += H[n_inp-1]*alp[n_inp-1]
    return temp
def d2gdsds(sig,alp): return -np.eye(ndim)/E
def d2gdsda(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -np.eye(ndim)
    temp[n_inp] = -np.eye(ndim)
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0] = -np.eye(ndim)
    temp[n_inp] = -np.eye(ndim)
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
    s = dfde(eps,alp)
    t1 = np.sqrt(np.einsum("ni,ni,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k))
    t2 = np.sqrt(sum(chi[n_inp]*chi[n_inp]))
    t3 = np.sqrt(sum(s*s))
    return np.array([t1 - 1.0 + R*(t2 - t3)])
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int,ndim])
    t1 = hu.floor(np.sqrt(np.einsum("ni,ni,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)))
    t2 = hu.floor(np.sqrt(sum(chi[n_inp]*chi[n_inp])))
    temp[0,:n_inp,:] += np.einsum("ni,n,n->ni",chi[:n_inp],recip_k,recip_k)/t1
    temp[0,n_inp,:] = R*chi[n_inp]/t2
    return temp
def dyde_f(chi,eps,alp): 
    s = dfde(eps,alp)
    temp = np.zeros([n_y,ndim])
    t3 = hu.floor(np.sqrt(sum(s*s)))
    for i in range(n_y):
        temp[i] = -R*E*s/t3
    return temp
def dyda_f(chi,eps,alp):
    s = dfde(eps,alp)
    s = dfde(eps,alp)
    temp = np.zeros([n_y,n_int,ndim])
    t3 = hu.floor(np.sqrt(sum(s*s)))
    for i in range(n_y):
        temp[i,0] = R*E*s/t3
        temp[i,n_inp] = R*E*s/t3
    return temp

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): 
    t1 = np.sqrt(np.einsum("ni,ni,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k))
    t2 = np.sqrt(sum(chi[n_inp]*chi[n_inp]))
    t3 = np.sqrt(sum(sig*sig))
    return np.array([t1 - 1.0 + R*(t2 - t3)])
def dydc_g(chi,sig,alp): 
    temp = np.zeros([n_y,n_int,ndim])
    t1 = hu.floor(np.sqrt(np.einsum("ni,ni,n,n->",chi[:n_inp],chi[:n_inp],recip_k,recip_k)))
    t2 = hu.floor(np.sqrt(sum(chi[n_inp]*chi[n_inp])))
    temp[0,:n_inp,:] += np.einsum("ni,n,n->ni",chi[:n_inp],recip_k,recip_k)/t1
    temp[0,n_inp,:] = R*chi[n_inp]/t2
    return temp
def dyds_g(chi,sig,alp):
    temp = np.zeros([n_y,ndim])
    t3 = hu.floor(np.sqrt(sum(sig*sig)))
    for i in range(n_y):
        temp[i] = -R*sig/t3
    return temp
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int,ndim])

#def w_f(chi,eps,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_f(chi,eps,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
#def w_g(chi,sig,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_g(chi,sig,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
