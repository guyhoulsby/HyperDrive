import numpy as np
import HyperUtils as hu

check_eps = 0.3
check_sig = 2.0
check_alp = np.array([0.2, 0.18, 0.16, 0.14])
check_chi = np.array([0.9, 1.0, 1.1, 1.2])

file = "h1epmk_ser"
name = "1D Linear Elastic - Plastic with Multisurface Kinematic Hardening - Series"
mode = 0
ndim = 1
const = [100.0, 4, 0.1, 100.0, 0.3, 33.333333, 0.6, 20.0, 1.0, 10.0]
eta = 0.04
tref = 1.0e6
rp = 0.0

def deriv():
    global E, k, recip_k, H, name_const, rk
    global n_int, n_inp, n_y, n_const
    E = float(const[0])
    n_int = int(const[1])
    n_inp = int(const[1])
    n_y = int(const[1])
    n_const = 2 + 2*n_inp
    k = np.array(const[2:2+2*n_inp:2])
    H = np.array(const[3:3+3*n_inp:2])
    recip_k = 1.0 / k
    rk = np.sqrt(sum(k**2))
    name_const = ["E", "N"]
    for i in range(n_inp):
        name_const.append("k"+str(i+1))
        name_const.append("H"+str(i+1))

deriv()

def ep(alp): return sum(alp)

def f(eps,alp): return E*((eps-ep(alp))**2)/2.0 + np.einsum("n,n,n->",H,alp,alp)/2.0
def dfde(eps,alp): return E*(eps-ep(alp))
def dfda(eps,alp): return -E*(eps-ep(alp)) + np.einsum("n,n->n",H,alp)
def d2fdede(eps,alp): return E
def d2fdeda(eps,alp): return -E*np.ones(n_int)
def d2fdade(eps,alp): return -E*np.ones(n_int)
def d2fdada(eps,alp): return E + np.diag(H)

def g(sig,alp): return -(sig**2)/(2.0*E) - sig*ep(alp) + np.einsum("n,n,n->",H,alp,alp)/2.0
def dgds(sig,alp): return -sig/E - ep(alp)
def dgda(sig,alp): return -sig + np.einsum("n,n->n",H,alp)
def d2gdsds(sig,alp): return -1.0/E
def d2gdsda(sig,alp): return -np.ones(n_int)
def d2gdads(sig,alp): return -np.ones(n_int)
def d2gdada(sig,alp): return np.diag(H)

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

def w_f(chi,eps,alp): return sum([(k[i]**(2.0-rp)/(tref*rk**(1.0-rp)))*(hu.mac(abs(chi[i])/k[i] -1.0)**2)/(2.0*eta) for i in range(n_inp)]) 
def dwdc_f(chi,eps,alp): return np.array([(k[i]**(1.0-rp)/(rk**(1.0-rp)))*hu.S(chi[i])*hu.mac(abs(chi[i])/k[i] - 1.0)/(eta*tref) for i in range(n_inp)])
def w_g(chi,sig,alp): return sum([(k[i]/tref)*(hu.mac(abs(chi[i])/k[i] -1.0)**2)/(2.0*eta) for i in range(n_inp)]) 
def dwdc_g(chi,sig,alp): return np.array([hu.S(chi[i])*hu.mac(abs(chi[i])/k[i] - 1.0)/(eta*tref) for i in range(n_inp)])

