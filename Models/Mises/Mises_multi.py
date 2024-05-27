import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.01,0.04,0.06,0.10,0.06,0.04])
check_sig = np.array([1.01,2.04,3.06,1.05,1.03,1.02])
check_alp = np.array([[0.001,0.004,0.006,0.01,0.006,0.004],
                      [0.001,0.004,0.006,0.01,0.006,0.004],
                      [0.001,0.004,0.006,0.01,0.006,0.004]])
check_chi = np.array([[1.01,2.04,3.06,1.05,1.03,1.02],
                      [1.01,2.04,3.06,1.05,1.03,1.02],
                      [1.01,2.04,3.06,1.05,1.03,1.02]])

file = "Mises_multi_m"
name = "Three surface hardening von Mises - Mandel form"
mode = 1
ndim = 6
n_y = 3
n_int = 3
n_inp = 3
const = [1000.0, 750.0, 1.0, 250.0, 2.0, 50.0, 3.0, 10.0]
name_const = ["K", "G", "k1", "H1", "k2", "H2", "k3", "H3"]

def deriv():
    global K, G, k, H
    k = np.zeros(3)
    H = np.zeros(3)
    K    = float(const[0])
    G    = float(const[1])
    k[0] = float(const[2])
    H[0] = float(const[3])
    k[1] = float(const[4])
    H[1] = float(const[5])
    k[2] = float(const[6])
    H[2] = float(const[7])
        
def ep(alp): return np.einsum("ij->j",alp)
def ee(eps,alp): return eps - ep(alp)

def f(eps,alp): 
    temp = (K/2.0)*hu.i1_m(ee(eps,alp))**2 + 2.0*G*hu.j2_m(ee(eps,alp))
    for i in range(n_int):
        temp += 2.0*H[i]*hu.j2_m(alp[i,:])
    return temp
def dfde(eps,alp): 
    return K*hu.i1_m(ee(eps,alp))*hu.delta_m + 2.0*G*hu.dj2_m(ee(eps,alp))
def dfda(eps,alp): 
    temp = np.zeros([n_int,6])
    for i in range(n_int):
        temp[i,:] -= K*hu.i1_m(ee(eps,alp))*hu.delta_m + 2.0*G*hu.dj2_m(ee(eps,alp))
        temp[i,:] += 2.0*H[i]*hu.dj2_m(alp[i,:])
    return temp
def d2fdede(eps,alp): 
    return K*hu.II_m + 2.0*G*hu.d2j2_m()
def d2fdeda(eps,alp):
    temp = np.zeros([6,n_int,6])
    for i in range(n_int):
       temp[:,i,:] = -K*hu.II_m - 2.0*G*hu.d2j2_m()
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,6,6])
    for i in range(n_int):
       temp[i,:,:] = -K*hu.II_m - 2.0*G*hu.d2j2_m()
    return temp
def d2fdada(eps,alp): 
    temp = np.zeros([n_int,6,n_int,6])
    for i in range(n_int):
        for j in range(n_int):
            temp[i,:,j,:] = K*hu.II_m + 2.0*G*hu.d2j2_m()
        temp[i,:,i,:] += 2.0*H[i]*hu.d2j2_m(alp[i,:])
    return temp

def g(sig,alp): 
    temp = -(hu.i1_m(sig)**2)/(18.0*K) - hu.j2_m(sig)/(2.0*G)
    temp -= hu.cont_m(sig,ep(alp))
    for i in range(n_int):
        temp += 2.0*H[i]*hu.j2_m(alp[i,:])
    return temp
def dgds(sig,alp):
    return -hu.i1_m(sig)*hu.delta_m/(9.0*K) - hu.dj2_m(sig)/(2.0*G) - ep(alp)
def dgda(sig,alp):
    temp = np.zeros([n_int,6])
    for i in range(n_int):
        temp[i,:] = -sig + 2.0*H[i]*hu.dj2_m(alp[i,:])
    return temp
def d2gdsds(sig,alp): 
    return -hu.II_m/(9.0*K) - hu.d2j2_m()/(2.0*G)
def d2gdsda(sig,alp):
    temp = np.zeros([6,n_int,6])
    for i in range(6):
        temp[i,:,i] = -1.0
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,6,6])
    for i in range(6):
        temp[:,i,i] = -1.0
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,6,n_int,6])
    for i in range(n_int):
        temp1 = 2.0*H[i]*hu.d2j2_m()
        temp[i,:,i,:] = temp1
    return temp

y_exclude = True

def y(eps,sig,alp,chi):
    temp = np.zeros(n_y)
    for i in range(n_y):
        temp[i] = hu.j2_m(chi[i,:]) - k[i]**2
    return temp
def dydc(eps,sig,alp,chi): 
    temp = np.zeros([n_y,n_int,6])
    for i in range(n_y):
        temp[i,i,:] = hu.dj2_m(chi[i,:])
    return temp
def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyda(eps,sig,alp,chi): return np.zeros([n_y,n_int,ndim])