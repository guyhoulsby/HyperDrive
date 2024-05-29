import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([0.41])
check_sig = np.array([0.9])
check_alp = np.array([[0.01], [0.11], [0.12], [0.14], [0.02]])
check_chi = np.array([[0.31], [0.61], [0.51], [1.11], [0.89]])

file = "habb1"
name = "HABB model - Series HARM"
ndim = 1
const = [85.0, 1.0, 1.0, 3.2, 40.0, 0.05, 4.5, 5.0, 0.1, 4, 0.1, 1.0]
name_const = ["E0", "kU", "epspU", "mh", "R0", "beta0", "mr", "ms", "mk", "Ns", "mu", "Rfac"]

def deriv():
    global E0, kU, epspU, mh, R0, beta0, mr, ms, mk, Ns, mu, Rfac
    global n_int, n_inp, n_y, n_const, k0, H
    E0    = float(const[0])
    kU    = float(const[1])
    epspU = float(const[2])
    mh    = float(const[3])
    R0    = float(const[4])
    beta0 = float(const[5])
    mr    = float(const[6])
    ms    = float(const[7])
    mk    = float(const[8])
    Ns    = int(const[9])
    mu    = float(const[10])
    Rfac  = float(const[11])
    n_int = Ns + 1
    n_inp = Ns
    n_y = Ns
    n_const = 12
    epsc = np.zeros(Ns+1)
    sigc = np.zeros(Ns+1)
    k0   = np.zeros(Ns)
    H    = np.zeros(Ns)
    for n in range(Ns+1):
        sigc[n] = kU * float(n)/float(Ns)
        epsc[n] = sigc[n]/E0 + epspU*((sigc[n]/kU)**mh)
    for n in range(Ns):
        k0[n] = kU* float(n+1)/float(Ns)
        H[n] = (kU/(float(Ns)*epspU)) * (Ns**mh) / \
               (float(n+2)**mh - 2.0*float(n+1)**mh + float(n)**mh)
    
def up_kR(sig,alp):
    beta = beta0 + alp[Ns,0]
    k    = k0 * ((beta/beta0)**mk)
    R    = np.zeros(Ns)
    for n in range(Ns):
        R[n] = (Rfac*R0 * (k[n]/kU) * ((beta/beta0)**(-mr)) * ((np.abs(sig[0])/kU)**ms))
    return k, R

def up_kR_dev(sig,alp):
    beta = beta0 + alp[Ns,0]
    k, R = up_kR(sig, alp)
    dkdb = mk*k / beta
    dRds = np.zeros(Ns)
    dRdb = np.zeros(Ns)
    for n in range(Ns):
        if sig[0] != 0.0:
            dRds[n] = (ms*R[n] / sig[0])
        dRdb[n] = ((R[n]/k[n])*dkdb[n] - mr*R[n]/beta)
    return dkdb, dRds, dRdb

def ep(alp): return np.einsum("ni->",alp)

def f(eps,alp): 
    temp = E0*((eps[0] - ep(alp))**2)/2.0 
    temp += np.einsum("n,ni,ni->",H,alp[:n_inp,:],alp[:n_inp,:])/2.0
    print(temp)
    return temp
def dfde(eps,alp): 
    temp = np.zeros(ndim)
    temp[0] = E0*(eps[0] - ep(alp))
    return temp
def dfda(eps,alp): 
    temp = np.zeros([n_int,ndim])
    temp[:,0] = -E0*(eps[0] - ep(alp))
    temp[:n_inp,:] += np.einsum("n,ni->ni",H,alp[:n_inp,:])
    return temp
def d2fdede(eps,alp): 
    temp = np.zeros([ndim,ndim])
    temp[0,0] = E0
    return temp
def d2fdeda(eps,alp): 
    temp = np.zeros([ndim,n_int,ndim])
    temp[0,:,0] = -E0
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[:,0,0] = -E0
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    temp[:,0,:,0] = E0
    for i in range(n_inp): 
        temp[i,0,i,0] += H[i]
    return temp

def g(sig,alp): 
    temp = -(sig[0]**2) / (2.0*E0) 
    temp += -sig[0]*ep(alp) 
    temp += np.einsum("n,ni,ni->",H,alp[:n_inp,:],alp[:n_inp,:]) / 2.0
    return temp
def dgds(sig,alp): 
    temp = np.zeros(ndim)
    temp[0] = -sig[0]/E0 - ep(alp)
    return temp
def dgda(sig,alp):
    temp = np.zeros([n_int,ndim])
    temp[:,0] = -sig[0]
    temp[:n_inp,:] += np.einsum("n,ni->ni",H,alp[:n_inp,:])
    return temp
def d2gdsds(sig,alp):
    temp = np.zeros([ndim,ndim])
    temp[0,0] = -1.0 / E0
    return temp
def d2gdsda(sig,alp):
    temp = np.zeros([ndim,n_int,ndim])
    temp[0,:,0] = -1.0
    return temp
def d2gdads(sig,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[:,0,0] = -1.0
    return temp
def d2gdada(sig,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    for i in range(n_inp): 
        temp[i,0,i,0] = H[i]
    return temp

y_exclude = 1
w_exclude = 1
def y(eps,sig,alp,chi): 
    k, R = up_kR(sig,alp)
    temp = np.zeros(n_y)
    temp[:] = abs(chi[:n_inp,0]) - k[:] + R[:]*(abs(chi[n_inp,0]) - abs(sig[0]))
    return np.array(temp)
def dydc(eps,sig,alp,chi): 
    k, R = up_kR(sig,alp)
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_y):
        temp[i,i,0]     = hu.S(chi[i,0])
        temp[i,n_inp,0] = R[i]*hu.S(chi[n_inp,0])
    return temp
def dyde(eps,sig,alp,chi): 
    return np.zeros([n_y,ndim])
def dyds(eps,sig,alp,chi):
    temp = np.zeros([n_y,ndim])
    k, R = up_kR(sig,alp)
    dkdb, dRds, dRdb = up_kR_dev(sig,alp)
    temp[:,0] = -R[:]*hu.S(sig[0]) + dRds[:]*(abs(chi[n_inp,0]) - abs(sig[0]))
    return temp
def dyda(eps,sig,alp,chi):
    k, R = up_kR(sig,alp)
    dkdb, dRds, dRdb = up_kR_dev(sig,alp)
    temp = np.zeros([n_y,n_int,ndim])
    for i in range(n_y):
       temp[i,n_inp,0] = -dkdb[i] + dRdb[i]*(abs(chi[n_inp,0]) - abs(sig[0]))
    return temp

def w(eps,sig,alp,chi): 
    yloc = y(eps,sig,alp,chi)
    temp = 0.0
    for i in range(n_y):
        temp += hu.mac(yloc[i])**2
    temp = temp / (2.0*mu)
    return temp 
def dwdc(eps,sig,alp,chi): 
    yloc = y(eps,sig,alp,chi)
    k, R = up_kR(sig,alp)
    temp = np.zeros([n_int,ndim])
    for i in range(n_y):
        temp[i,0]     += hu.mac(yloc[i])*hu.S(chi[i,0])
        temp[n_inp,0] += hu.mac(yloc[i])*R[i]*hu.S(chi[n_inp,0])
    temp = temp / mu
    return temp 