import numpy as np
import HyperUtils as hu

check_eps = 0.3
check_sig = 2.0
check_alp = np.array([0.2, 0.18, 0.16, 0.2])
check_chi = np.array([0.9, 1.0, 1.1, 1.2])

file = "h1epmk_par"
name = "1D Linear Elastic-Plastic with Multisurface Kinematic Hardening - Parallel"
mode = 0
ndim = 1
const = [5.0, 4, 0.05, 50.0, 0.15, 30.0, 0.2, 10.0, 0.3, 5.0]
eta = 0.04
tref = 1.0e6

def deriv():
    global Einf, E0, k, H, recip_k, name_const, n_int, n_inp, n_y, n_const
    Einf = float(const[0])
    n_int = int(const[1])
    n_inp = int(const[1])
    n_y = int(const[1])
    n_const = 2 + 2*n_int
    k = np.array(const[2:2+2*n_inp:2])
    H = np.array(const[3:3+2*n_inp:2])
    recip_k = 1.0 / k
    E0 = Einf + sum(H)
    name_const = ["E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))

deriv()

def f(eps,alp): return np.einsum("i,i,i->",H,(eps-alp),(eps-alp))/2.0 + Einf*(eps**2)/2.0
def dfde(eps,alp): return np.einsum("i,i->",H,(eps-alp)) + Einf*eps
def dfda(eps,alp): return -np.einsum("i,i->i",H,(eps-alp))
def d2fdede(eps,alp): return E0
def d2fdeda(eps,alp): return -H
def d2fdade(eps,alp): return -H
def d2fdada(eps,alp): return np.diag(H)

def g(sig,alp): return -((sig + np.einsum("n,n->",H,alp))**2)/(2.0*E0) + np.einsum("n,n,n->",H,alp,alp)/2.0
def dgds(sig,alp): return -(sig + np.einsum("n,n->",H,alp)) / E0
def dgda(sig,alp): return -(sig + np.einsum("n,n->",H,alp))*H/E0 + np.einsum("n,n->n",H,alp)
def d2gdsds(sig,alp): return -1.0/E0
def d2gdsda(sig,alp): return -H/E0
def d2gdads(sig,alp): return -H/E0
def d2gdada(sig,alp): return -np.einsum("n,m->nm",H,H)/E0 + np.diag(H)

#def d_f(alpr,eps,alp): return k*abs(alpr)

def y_f(chi,eps,alp): return abs(chi)*recip_k - 1.0
def dydc_f(chi,eps,alp):
    temp = np.zeros([n_y,n_int])
    for i in range(n_y):
        temp[i,i] = hu.S(chi[i])/k[i]
    return temp
def dyde_f(chi,eps,alp): return np.zeros(n_y)
def dyda_f(chi,eps,alp): return np.zeros([n_y,n_int])

#def d_g(alpr,eps,alp): return k*abs(alpr)

def y_g(chi,sig,alp): return abs(chi)*recip_k - 1.0
def dydc_g(chi,sig,alp):
    temp = np.zeros([n_y,n_int])
    for i in range(n_y):
        temp[i,i] = hu.S(chi[i])/k[i]
    return temp
def dyds_g(chi,sig,alp): return np.zeros(n_y)
def dyda_g(chi,sig,alp): return np.zeros([n_y,n_int])

def w_f(chi,eps,alp): return sum([(k[i]/tref)*(hu.mac(abs(chi[i])/k[i] -1.0)**2)/(2.0*eta) for i in range(n_inp)]) 
def dwdc_f(chi,eps,alp): return np.array([hu.S(chi[i])*hu.mac(abs(chi[i])/k[i] - 1.0)/(eta*tref) for i in range(n_inp)])
def w_g(chi,sig,alp): return sum([(k[i]/tref)*(hu.mac(abs(chi[i])/k[i] -1.0)**2)/(2.0*eta) for i in range(n_inp)]) 
def dwdc_g(chi,sig,alp): return np.array([hu.S(chi[i])*hu.mac(abs(chi[i])/k[i] - 1.0)/(eta*tref) for i in range(n_inp)])
