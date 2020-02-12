import numpy as np
import HyperUtils as hu

check_eps = 0.3
check_sig = 2.0
check_alp = np.array([0.2, 0.18, 0.16, 0.14, 0.01])
check_chi = np.array([0.9, 1.0, 1.1, 1.2, 1.0])

file = "h1epmk_par_h"
name = "1D Linear Elastic-Plastic with Multisurface Kinematic Hardening - Parallel HARM"
mode = 0
ndim = 1
const = [5.0, 4, 0.050616, 50.0, 0.152501, 30.0, 0.207961, 10.0, 0.305214, 5.0, 0.1]
mu = 0.1

def deriv():
    global E0, Einf, k, recip_k, H, R, name_const, n_int, n_inp, n_y, n_const
    Einf = float(const[0])
    n_int = int(const[1]) + 1
    n_inp = int(const[1])
    n_y = int(const[1])
    n_const = 2 + 2*n_int + 1
    k = np.array(const[2:2 + 2*n_inp:2])
    H = np.array(const[3:3 + 2*n_inp:2])
    R = float(const[2 + 2*n_inp])
    recip_k = 1.0 / k
    E0 = Einf + sum(H)
    name_const = ["Einf", "N"]
    for i in range(n_int):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))
    name_const.append("R")

deriv()

def f(eps,alp): 
    temp = (eps-alp[:n_inp]-alp[n_inp])
    return np.einsum("i,i,i->",H,temp,temp)/2.0 + Einf*((eps-alp[n_inp])**2)/2.0
def dfde(eps,alp): return np.einsum("i,i->",H,(eps-alp[:n_inp]-alp[n_inp])) + Einf*(eps-alp[n_inp])
def dfda(eps,alp): 
    temp = np.zeros(n_int)
    temp[:n_inp] -= np.einsum("i,i->i",H,(eps-alp[:n_inp]-alp[n_inp]))
    temp[n_inp] -= np.einsum("i,i->",H,(eps-alp[:n_inp]-alp[n_inp]))
    temp[n_inp] -= Einf*(eps-alp[n_inp])
    return temp
def d2fdede(eps,alp): return E0
def d2fdeda(eps,alp): 
    temp = np.zeros(n_int)
    temp[:n_inp] -= H
    temp[n_inp] -= E0
    return temp
def d2fdade(eps,alp):
    temp = np.zeros(n_int)
    temp[:n_inp] -= H
    temp[n_inp] -= E0
    return temp
def d2fdada(eps,alp): 
    temp = np.zeros([n_int,n_int])
    for i in range(n_inp):
        temp[i,i] = H[i]
        temp[i,n_inp] += H[i] 
        temp[n_inp,i] += H[i] 
    temp[n_inp,n_inp] = E0
    return temp

def g(sig,alp): return -((sig + np.einsum("n,n->",H,alp[:n_inp]))**2)/(2.0*E0) - sig*alp[n_inp] + np.einsum("n,n,n->",H,alp[:n_inp],alp[:n_inp])/2.0
def dgds(sig,alp): return -(sig + np.einsum("n,n->",H,alp[:n_inp])) / E0 - alp[n_inp]
def dgda(sig,alp): 
    temp = np.zeros(n_int)
    temp[:n_inp] = -(sig + np.einsum("n,n->",H,alp[:n_inp]))*H/E0 + np.einsum("n,n->n",H,alp[:n_inp])
    temp[n_inp] = -sig
    return temp
def d2gdsds(sig,alp): return -1.0/E0
def d2gdsda(sig,alp):
    temp = np.zeros(n_int)
    temp[:n_inp] = -H/E0
    temp[n_inp] = -1.0
    return temp
def d2gdads(sig,alp):
    temp = np.zeros(n_int)
    temp[:n_inp] = -H/E0
    temp[n_inp] = -1.0
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,n_int])
    temp[:n_inp,:n_inp] = -np.einsum("n,m->nm",H,H)/E0 + np.diag(H)
    return temp

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): 
    s = dfde(eps,alp)
    return abs(chi[:n_inp])*recip_k - 1.0 + R*(np.abs(chi[n_inp]) - np.abs(s))
def dydc_f(chi,eps,alp): 
    temp = np.zeros([n_y,n_int])
    for i in range(n_inp):
        temp[i,i] += hu.S(chi[i])*recip_k[i]
        temp[i,n_inp] = R*hu.S(chi[n_inp])
    return temp
def dyde_f(chi,eps,alp): 
    s = dfde(eps,alp)
    return -R*E0*hu.S(s)*np.ones(n_y)
def dyda_f(chi,eps,alp): 
    s = dfde(eps,alp)
    temp = R*hu.S(s)*np.ones([n_y,n_int])
    for i in range(n_inp):
        for j in range(n_inp):
            temp[i,j] = temp[i,j]*H[j]
        temp[i,n_inp] = temp[i,n_inp]*E0
    return temp

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): return abs(chi[:n_inp])*recip_k - 1.0 + R*(np.abs(chi[n_inp]) - np.abs(sig))
def dydc_g(chi,sig,alp): 
    temp=np.zeros([n_y,n_int])
    for i in range(n_inp):
        temp[i,i] += hu.S(chi[i])*recip_k[i]
        temp[i,n_inp] = R*hu.S(chi[n_inp])
    return temp
def dyds_g(chi,sig,alp): return -R*hu.S(sig)*np.ones(n_y)
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int])

#def w_f(chi,eps,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_f(chi,eps,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
#def w_g(chi,sig,alp): return sum([(mac(abs(chi[i]) - k[i])**2)/(2.0*mu) for i in range(n_int)]) 
#def dwdc_g(chi,sig,alp): return np.array([S(chi[i])*mac(abs(chi[i]) - k[i])/mu for i in range(n_int)])
