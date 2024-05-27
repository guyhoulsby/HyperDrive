import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.01,0.02,0.03,0.04,0.05,0.03])
check_sig = np.array([1.01,1.02,1.03,2.04,1.05,1.03])
check_alp = np.array([[0.001,0.002,0.003,0.004,0.005,0.006]])
check_chi = np.array([[1.01,1.02,1.03,2.04,1.05,3.06]])

file = "Mises"
name = "von Mises model"
n_y = 1
n_int = 1
n_inp = 1
ndim = 6
const = [150.0, 100.0, 1.0]
name_const = ["K", "G", "k"]
global K, G, k

def deriv():
    global K, G, k
    K = float(const[0])
    G = float(const[1])
    k = float(const[2])

deriv()
        
def ee(eps,alp): return eps - alp[0]
estiff = (K/2.0)*hu.d2i1sq_m() + 2.0*G*hu.d2j2_m()

def f(eps,alp): 
    epse = ee(eps,alp)
    return (K/2.0)*hu.i1sq_m(epse) + 2.0*G*hu.j2_m(epse)
def dfde(eps,alp): 
    epse = ee(eps,alp)
    return (K/2.0)*hu.di1sq_m(epse) + 2.0*G*hu.dj2_m(epse)
def dfda(eps,alp): 
    epse = ee(eps,alp)
    return -np.array([(K/2.0)*hu.di1sq_m(epse) + 2.0*G*hu.dj2_m(epse)])
def d2fdede(eps,alp): 
    return estiff
def d2fdeda(eps,alp):
    temp = -estiff
    return temp.reshape(6,1,6)
def d2fdade(eps,alp):
    temp = -estiff
    return temp.reshape(1,6,6)
def d2fdada(eps,alp): 
    temp = estiff
    return temp.reshape(1,6,1,6)

def g(sig,alp): 
    return -(1.0/(18.0*K))*hu.i1sq_m(sig) - (1.0/(2.0*G))*hu.j2_m(sig) - sig @ alp[0]
def dgds(sig,alp): 
    return -(1.0/(18.0*K))*hu.di1sq_m(sig) - (1.0/(2.0*G))*hu.dj2_m(sig) - alp[0]
def dgda(sig,alp): 
    return np.array([-sig])
def d2gdsds(sig,alp): 
    return -(1.0/(18.0*K))*hu.d2i1sq_m() - (1.0/(2.0*G))*hu.d2j2_m()
def d2gdsda(sig,alp):
    temp = -np.eye(6)
    return temp.reshape(6,1,6)
def d2gdads(sig,alp):
    temp = -np.eye(6)
    return temp.reshape(1,6,6)
def d2gdada(sig,alp): 
    return np.zeros([1,6,1,6])

def y(eps,sig,alp,chi): return np.array([hu.j2_m(chi[0]) - (k**2)])
def dydc(eps,sig,alp,chi): 
    temp = hu.dj2_m(chi[0])
    return temp.reshape(1,1,6)
def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyda(eps,sig,alp,chi): return np.zeros([n_y,n_int,ndim])
