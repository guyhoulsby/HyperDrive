import autograd.numpy as np
from HyperDrive import Utils as hu

check_eps = np.array([-0.1, 0.05])
check_sig = np.array([40.0, 20.0])
check_alp = np.array([[0.02, 0.01]])
check_chi = np.array([[80.0, 20.0]])

file = "hmcc"
name = "Modified Cam Clay" # constant shear modulus variant
ndim = 2
n_int = 1
n_y = 1
n_const = 6
names = ["t", "eps_v", "eps_s",   "p",   "q"]
units = ["s",     "-",     "-", "kPa", "kPa"]
name_const = [ "pr", "lambda*", "kappa*", "M",    "G", "pco"]
const =      [100.0,       0.2,     0.05, 1.0, 3000.0, 100.0]

def deriv():
    global pr, lambdas, kappas, M, G, pco
    pr = float(const[0])
    lambdas = float(const[1])
    kappas = float(const[2])
    M = float(const[3])
    G = float(const[4])
    pco = float(const[5])

def px(alp): return (pco/2.0) * np.exp(alp[0,0]/(lambdas - kappas))
def dpxda(alp): return px(alp) / (lambdas - kappas)
def estiff(eps,alp): return np.array([[pr*np.exp((eps[0]-alp[0,0])/kappas)/kappas, 0.0], [0.0, 3.0*G]])

def f(eps,alp): 
    return pr*kappas*np.exp((eps[0]-alp[0,0])/kappas) + (3.0*G/2.0)*((eps[1]-alp[0,1])**2) - pr*kappas
def dfde(eps,alp): 
    return np.array([pr*np.exp((eps[0]-alp[0,0])/kappas), 3.0*G*(eps[1]-alp[0,1])])
def dfda(eps,alp): 
    return -np.array([[pr*np.exp((eps[0]-alp[0,0])/kappas), 3.0*G*(eps[1]-alp[0,1])]])
def d2fdede(eps,alp): 
    return estiff(eps,alp)
def d2fdeda(eps,alp):
    temp = np.zeros([ndim,n_int,ndim])
    temp[:,0,:] = -estiff(eps,alp)
    return temp
def d2fdade(eps,alp):
    temp = np.zeros([n_int,ndim,ndim])
    temp[0,:,:] = -estiff(eps,alp)
    return temp
def d2fdada(eps,alp):
    temp = np.zeros([n_int,ndim,n_int,ndim])
    temp[0,:,0,:] = estiff(eps,alp)
    return temp

def g(sig,alp): return -pr*kappas*hu.ilog(sig[0]/pr) - (sig[1]**2)/(6.0*G) - np.einsum("i,ni->",sig,alp)
def dgds(sig,alp): return -np.array([kappas*np.log(sig[0]/pr), sig[1]/(3.0*G)]) - alp[0]
def dgda(sig,alp): return -np.array([sig])
def d2gdsds(sig,alp): return -np.array([[kappas/sig[0], 0.0], [0.0, 1.0/(3.0*G)]])
def d2gdsda(sig,alp): return -np.array([[[1.0, 0.0]],[[0.0, 1.0]]])
def d2gdads(sig,alp): return -np.array([[[1.0, 0.0],  [0.0, 1.0]]])
def d2gdada(sig,alp): return np.zeros([n_int,ndim,n_int,ndim])

# variant with yield as quadratic function, no back-stress
#def y(eps,sig,alp,chi): return np.array([(chi[0,0]-px(alp))**2 + (chi[0,1]/M)**2 - px(alp)**2])
#def dydc(eps,sig,alp,chi): return np.array([[[2.0*(chi[0,0]-px(alp)), 2.0*chi[0,1]/(M**2)]]])
#def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
#def dyda(eps,sig,alp,chi): return np.array([[[(-2.0*chi[0,0])*dpxda(alp), 0.0]]])
#def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])

# variant with yield dimensionless, no back-stress
def y(eps,sig,alp,chi): return np.array([np.sqrt((chi[0,0]/px(alp)-1.0)**2 + (chi[0,1]/(M*px(alp)))**2) - 1.0])
def dydc(eps,sig,alp,chi): 
    temp = np.sqrt((chi[0,0]/px(alp)-1.0)**2 + (chi[0,1]/(M*px(alp)))**2)
    return np.array([[[(chi[0,0]/px(alp)-1.0)/px(alp), chi[0,1]/((px(alp)*M)**2)]]]) / temp
def dyde(eps,sig,alp,chi): return np.zeros([n_y,ndim])
def dyda(eps,sig,alp,chi): 
    temp = np.sqrt((chi[0,0]/px(alp)-1.0)**2 + (chi[0,1]/(M*px(alp)))**2)
    return np.array([[[((-(chi[0,0]/px(alp)-1.0)*chi[0,0]/(px(alp)**2)
                         -(chi[0,1]**2)/((M**2)*(px(alp)**3)))/temp)*dpxda(alp), 0.0]]])
def dyds(eps,sig,alp,chi): return np.zeros([n_y,ndim])
