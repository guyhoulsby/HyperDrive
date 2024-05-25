# HyperDrive
# Version 3.1.0
# (c) G.T. Houlsby, 2018-2023
#
# Routines to run the HyperDrive system for implementing hyperplasticity models 

# set the following line to auto = False if autograd is not available
auto = True

if auto:
#    import jax as ag       # ignore this. Not yet implemented
#    import jax.numpy as np # ignore this. Not yet implemented
    import autograd as ag
    import autograd.numpy as np
else:
    import numpy as np
    
import matplotlib.pyplot as plt
import os
from copy      import deepcopy
from importlib import import_module
from re        import split
from scipy     import optimize
from sys       import exit as sysexit
from time      import process_time

def weight_dydc(dydc):
    temp = np.zeros([hj.n_y, hj.n_int, hj.n_dim])
    for i in range(hj.n_int):
        temp[:,i,:] = hj.rwt[i]*dydc[:,i,:]
    return temp

class Ein: # Einsein summation shorthand, several products now replaced by @
    chi  = "N,Ni->Ni"   # for weight calculation
    dydc = "N,MNi->MNi" # for weight calculation
    b = "ki,i->k"
    c = "imj,mj->i"
    d = "N,Nmi->mi"
    #e = "Ni,i->N"
    f = "Nj,jk->Nk"
    g = "Nk,knl->Nnl"
    h = "Nmi,mij->Nj"
    i = "Nmi,minj->Nnj"
    j = "Nni,Mni->NM"
class Time:
    start = 0.0
    last  = 0.0
class State:
    t = 0.0
    eps = np.zeros(1)
    sig = np.zeros(1)
    alp = np.zeros([1,1])
    chi = np.zeros([1,1])
s    = State() # current state
save = State() # saved version of state
si   = State() # increments for numerical differentiation
si.eps = 0.0001
si.sig = 0.001
si.alp = 0.0001
si.chi = 0.001

class Large:
    F = np.eye(3)
    large_strain_def = "Hencky"
    sig = np.zeros(6)
hl = Large()

class Job: # stores common values for the current job
    title = ""
    model = "undefined"
    n_dim = 1
    n_int = 1
    n_y   = 1
    rwt   = np.ones(1) # reciprocal weights in internal variables * generalised stress
    prefs = ["analytical", "automatic", "numerical"] # preference for differentials
    fform = False
    gform = False
    rate  = False # rate option
    large = False # large strain option
    quiet = False
    voigt = False # option to use Voigt notation for input and output
                  # internal calculations use Mandel vectors
    acc  = 0.5       # acceleration factor
    ytol = -0.00001  # tolerance on triggering yield
    auto_t_step = False # option for setting automatic timestep, not fully tested
    at_step     = 0.001
    start_inc = True
    recording = True
    curr_test = 0
    rec = [[]]    # record of test(s)
    colour = "b"  # current colour for plotting
    test = False
    test_rec = []
    test_col = [] # colour for each test
    hstart = 0    # for highlighting
    high = [[]]
    test_high = []
    epsref = np.zeros(1) # strain reference point
hj = Job()

class RKDP: # required for Runge-Kutta-Dorland-Price option (testing only)
    tfac = np.array([[   1.0/5.0,     0.0,        0.0,         0.0,        0.0      ],
                     [   3.0/40.0,    9.0/40.0,   0.0,         0.0,        0.0      ],
                     [   3.0/10.0,   -9.0/10.0,   6.0/5.0,     0.0,        0.0      ],
                     [ 226.0/729.0, -25.0/27.0, 880.0/729.0,  55.0/729.0,  0.0      ],
                     [-181.0/270.0,   5.0/2.0, -226.0/297.0, -91.0/27.0, 189.0/55.0]])
    t5th = np.array([19.0/216.0, 0.0, 1000.0/2079.0, -125.0/216.0, 81.0/88.0,  5.0/56.0 ])
    terr = np.array([11.0/360.0, 0.0,  -10.0/63.0,     55.0/72.0, -27.0/40.0, 11.0/280.0])
    opt  = False

udef = "undefined"
def undef(arg1=[],arg2=[],arg3=[],arg4=[]): return udef

# for large strain - currently unused
#def C(F):
#    return np.einsum("ki,kj->ij",F,F)
#def b(F):
#    return np.einsum("ik,jk->ij",F,F)
#def R(F):
#    A,lam,Bt = np.linalg.svd(F)
#    return A @ Bt
def Hencky(F):
    A,lam,Bt = np.linalg.svd(F)
    return np.einsum("ki,k,kj->ij",Bt,np.log(lam),Bt)
def Green(F):
    A,lam,Bt = np.linalg.svd(F)
    return 0.5*(np.einsum("ki,k,k,kj->ij",Bt,lam,lam,Bt) - Utils.delta)
def gammaInt(lam):
    temp = np.ones([3,3])
    for m in range(3):
        for n in range(3):
            if m != n and lam[m] != lam[n]:
                r = lam[m] / lam[n]
                temp[m,n] = 2.0*r*np.log(r) / (r**2 - 1.0)
    return temp
def LL_H(F): # linear operator for Hencky strain: H~ = LL:d 
    A,lam,Bt = np.linalg.svd(F)
    B        = Bt.T
    return np.einsum("im,jn,km,ln,mn->ijkl",B,B,A,A,gammaInt(lam))
def LL_G(F): # linear operator for Green strain: E~ = LL:d 
    A,lam,Bt = np.linalg.svd(F)
    B        = Bt.T
    return np.einsum("im,jn,km,ln,m,n->ijkl",B,B,A,A,lam,lam)

# required for shortcuts to Voigt notation tests
S_txl_d = np.array([[0.0, -1.0,  1.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  1.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  1.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  1.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 1.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0]])
E_txl_d = np.array([[0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [1.0,  0.0,  0.0,  0.0,  0.0, 0.0]])
S_txl_u = np.array([[0.0, -1.0,  1.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  1.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  1.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 1.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0]])
E_txl_u = np.array([[0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [1.0,  1.0,  1.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [1.0,  0.0,  0.0,  0.0,  0.0, 0.0]])
S_dss   = np.array([[0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  1.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 1.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0]])
E_dss   = np.array([[1.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  1.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  1.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  0.0,  0.0, 0.0],
                    [0.0,  0.0,  0.0,  1.0,  0.0, 0.0]])

def qprint(*args): # suppressible print
    if not hj.quiet: print(*args)

def sig_from_voigt(sig_in): return Utils.vs_to_m(sig_in)  if hj.voigt else sig_in
def sig_to_voigt(sig_out):  return Utils.m_to_vs(sig_out) if (hj.voigt or hj.large) else sig_out
def eps_from_voigt(eps_in): return Utils.ve_to_m(eps_in)  if hj.voigt else eps_in
def eps_to_voigt(eps_out):  return Utils.m_to_ve(eps_out) if hj.voigt else eps_out

def substeps(nsub, tinc):
    return (int(tinc / hj.at_step) + 1) if hj.auto_t_step else nsub

def error(text = "Unspecified error"):
    print(text)
    sysexit()
def pause(message = ''):
    if len(message) > 0: print(message)
    text = input("Processing paused: hit ENTER to continue (x to exit)... ")
    if text == "x" or text == "X": sysexit()

#printing routine for matrices
def pprint(x, label, form="15.8"):
    if hj.quiet: return
    def ftext(x, formin="15.8"):
        form = "{:"+formin+"g}"
        return form.format(x)
    if type(x) == str:
        print(label,x)
        return
    if hasattr(x,"shape"):
        if len(x.shape) == 0:
            print(label+ftext(x,form))
        elif len(x.shape) == 1:
            text = label+" ["
            for i in range(x.shape[0]):
                text = text+ftext(x[i],form)
            text = text+"]"
            print(text)
        elif len(x.shape) == 2:
            for i in range(x.shape[0]):
                if i == 0:
                    text = label+" ["
                    lenstart = len(text)
                else:
                    text = " " * lenstart
                text = text+"["
                for j in range(x.shape[1]):
                    text = text+ftext(x[i,j],form)
                text = text+"]"
                if i == x.shape[0]-1:
                    text = text+"]"                
                print(text)
        elif len(x.shape) == 3:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if i == 0 and j == 0:
                        text = label+" ["
                        lenstart = len(text)
                    else:
                        text = " " * lenstart
                    if j == 0:
                        text = text+"["
                    else:
                        text = text+" "
                    text = text+"["
                    for k in range(x.shape[2]):
                        text = text+ftext(x[i,j,k],form)
                    text = text+"]"
                    if j == x.shape[1]-1:
                        text = text+"]"                
                        if i == x.shape[0]-1:
                            text = text+"]"                
                    print(text)
        elif len(x.shape) == 4:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        if i == 0 and j == 0 and k == 0:
                            text = label+" ["
                            lenstart = len(text)
                        else:
                            text = " " * lenstart
                        if j == 0 and k == 0:
                            text = text+"["
                        else:
                            text = text+" "
                        if k == 0:
                            text = text+"["
                        else:
                            text = text+" "
                        text = text+"["
                        for l in range(x.shape[3]):
                            text = text+ftext(x[i,j,k,l],form)
                        text = text+"]"
                        if k == x.shape[2]-1:
                            text = text+"]"      
                            if j == x.shape[1]-1:
                                text = text+"]"                
                                if i == x.shape[0]-1:
                                    text = text+"]"                
                        print(text)
        else:
            print(label)
            print(x)
    else:
        print(label+ftext(x,form))
    return

#Utility routines
class Utils:
    huge  = 1.0e12 
    big   = 1.0e6
    small = 1.0e-6
    tiny  = 1.0e-12
    root2 = np.sqrt(2.0)
    rooth = np.sqrt(0.5)

    def mac(x): # Macaulay bracket
        return 0.5*(x + np.abs(x)) # iterable version
    def macm(x): # Macaulay bracket, with rounding near the origin
        delta = Utils.small
        if x <= -delta: return 0.0
        if x >= delta: return x
        else: return ((x + delta)**2) / (4.0*delta)
    def S(x): # modified Signum function
        return x / (Utils.tiny + np.abs(x)) # iterable version
    def non_zero(x): # return a finite small value if argument close to zero
        return x if np.abs(x) >= Utils.tiny else Utils.tiny
    def ilog(x): # Integral of log, with ilog(1) = 0
        return 1.0 + x*np.log(x) - x
    def Ineg(x): # approximation to Indicator function for set of negative reals
        return 0.0 if x <= 0.0 else Utils.big * 0.5*(x**2)
    def Nneg(x): # approximation to Normal Cone for set of negative reals
        return 0.0 if x <= 0.0 else Utils.big * x
    def w_rate_lin(y, mu): # convert first canonical y to linear w function
        return (Utils.mac(y)**2) / (2.0*mu) 
    def w_rate_lind(y, mu): # differential of w_rate_lin
        return Utils.mac(y) / mu
    def w_rate_rpt(y, mu, r): # convert first canonical y to Rate Process Theory w function
        return mu*(r**2) * (np.cosh(Utils.mac(y) / (mu*r)) - 1.0)
    def w_rate_rptd(y, mu,r): # differential of w_rate_rpt
        return r * np.sinh(Utils.mac(y) / (mu*r))

    # Tensor utilities for 3x3 tensors - used for development and checking
    delta = np.eye(3) # Kronecker delta, unit tensor
    def dev(t): # deviator
        return t - (Utils.tr1(t)/3.0)*Utils.delta
    def trans(t): # transpose
        return t.T
    def sym(t): # symmetric part
        return (t + t.T)/2.0
    def skew(t): # skew (antisymmetric) part
        return (t - t.T)/2.0
    # shorthand for various products
    def cont(t1,t2): # double contraction of two tensors
        return np.einsum("ij,ij->",t1,t2)
    def iprod(t1,t2): # inner product of two tensors
        #return np.einsum("ij,jk->ik",t1,t2)
        return t1 @ t2
    # 4th order products
    def pijkl(t1,t2): return np.einsum("ij,kl->ijkl",t1,t2)
    def pikjl(t1,t2): return np.einsum("ik,jl->ijkl",t1,t2)
    def piljk(t1,t2): return np.einsum("il,jk->ijkl",t1,t2)
    # 4th order unit and projection tensors
    II    = pikjl(delta, delta)  # 4th order unit tensor, contracts with t to give t
    IIb   = piljk(delta, delta)  # 4th order unit tensor, contracts with t to give t-transpose
    IIbb  = pijkl(delta, delta)  # 4th order unit tensor, contracts with t to give tr(t).delta
    IIsym = (II + IIb) / 2.0     # 4th order unit tensor (symmetric)
    PP    = II    - (IIbb / 3.0) # 4th order projection tensor, contracts with t to give dev(t)
                                 # also equal to differtential d_dev(t) / dt
    PPb   = IIb   - (IIbb / 3.0) # 4th order projection tensor, contracts with t to give dev(t)-transpose
    PPsym = IIsym - (IIbb / 3.0) # 4th order projection tensor (symmetric)
    # traces
    # _e variants are einsum versions for checking
    def tr1_e(t): return np.trace(t)
    def tr2_e(t): return np.trace(t @ t)
    def tr3_e(t): return np.trace(t @ t @ t) 
    # mixed invariants of two tensors
    def trm_ab(a,b):   return np.trace(a @ b)
    def trm_a2b(a,b):  return np.trace(a @ a @ b)
    def trm_ab2(a,b):  return np.trace(a @ b @ b)
    def trm_a2b2(a,b): return np.trace(a @ a @ b @ b)
    def trm_abab(a,b): return np.trace(a @ b @ a @ b)
    # (maybe?) faster versions, hard coded
    def tr1(t): return t[0,0] + t[1,1] + t[2,2] # trace
    def tr2(t): # trace of square
        return t[0,0]*t[0,0] + t[0,1]*t[1,0] + t[0,2]*t[2,0] + \
               t[1,0]*t[0,1] + t[1,1]*t[1,1] + t[1,2]*t[2,1] + \
               t[2,0]*t[0,2] + t[2,1]*t[1,2] + t[2,2]*t[2,2]
    def tr3(t): # trace of cube
        return t[0,0]*(t[0,0]*t[0,0] + t[0,1]*t[1,0] + t[0,2]*t[2,0]) + \
               t[0,1]*(t[1,0]*t[0,0] + t[1,1]*t[1,0] + t[1,2]*t[2,0]) + \
               t[0,2]*(t[2,0]*t[0,0] + t[2,1]*t[1,0] + t[2,2]*t[2,0]) + \
               t[1,0]*(t[0,0]*t[0,1] + t[0,1]*t[1,1] + t[0,2]*t[2,1]) + \
               t[1,1]*(t[1,0]*t[0,1] + t[1,1]*t[1,1] + t[1,2]*t[2,1]) + \
               t[1,2]*(t[2,0]*t[0,1] + t[2,1]*t[1,1] + t[2,2]*t[2,1]) + \
               t[2,0]*(t[0,0]*t[0,2] + t[0,1]*t[1,2] + t[0,2]*t[2,2]) + \
               t[2,1]*(t[1,0]*t[0,2] + t[1,1]*t[1,2] + t[1,2]*t[2,2]) + \
               t[2,2]*(t[2,0]*t[0,2] + t[2,1]*t[1,2] + t[2,2]*t[2,2]) 
    # invariants - use basic definitions
    def i1(t): return Utils.tr1(t) # 1st invariant
    def i2(t): # 2nd invariant (NB some sources define this with opposite sign)
        return (Utils.tr2(t) - Utils.tr1(t)**2)/2.0 
    def i3(t): # 3rd invariant
        return (2.0*Utils.tr3(t) - 3.0*Utils.tr2(t)*Utils.tr1(t) + Utils.tr1(t)**3)/6.0 
    def j2(t): # 2nd invariant of deviator
        return (3.0*Utils.tr2(t) - Utils.tr1(t)**2)/6.0 
    def j2_a(t): # 2nd invariant of deviator, alternative form for checking
        return Utils.i2(Utils.dev(t)) 
    def j3(t): # 3rd invariant of deviator
        return (9.0*Utils.tr3(t) - 9.0*Utils.tr2(t)*Utils.tr1(t) + 2.0*Utils.tr1(t)**3)/27.0 
    def j3_a(t): # 3rd invariant of deviator, alternative form for checking
        return Utils.i3(Utils.dev(t)) 
    def det(t): # determinant - should be same as 3rd invariant
        return t[0,0]*(t[1,1]*t[2,2] - t[1,2]*t[2,1]) + \
               t[0,1]*(t[1,2]*t[2,0] - t[1,0]*t[2,2]) + \
               t[0,2]*(t[1,0]*t[2,1] - t[1,1]*t[2,0])
    def i1sq(t): # square of first invariant
        return Utils.i1(t)**2
    
    # differentials
    def dtr1(t=0.0):  return Utils.delta # differential of trace
    def di1(t=0.0):   return Utils.delta # differential of 1st invariant
    def di1sq(t): return 2.0*Utils.i1(t)*Utils.delta # differential of square of 1st invariant
    def dj2(t):   return Utils.dev(t) # differential of 2nd invariant of deviator
    
    # conversions to and from Voigt vectors
    def t_to_ve(t):  return np.array([    t[0,0],     t[1,1],     t[2,2],
                                      2.0*t[1,2], 2.0*t[2,0], 2.0*t[0,1]])
    def ve_to_t(t):  return np.array([[    t[0], 0.5*t[5], 0.5*t[4]], 
                                      [0.5*t[5],     t[1], 0.5*t[3]], 
                                      [0.5*t[4], 0.5*t[3],    t[2]]])
    def t_to_vs(t):  return np.array([t[0,0], t[1,1], t[2,2],
                                      t[1,2], t[2,0], t[0,1]])
    def vs_to_t(t):  return np.array([[t[0], t[5], t[4]], 
                                      [t[5], t[1], t[3]], 
                                      [t[4], t[3], t[2]]])
    # conversions to and from Mandel vectors
    def t_to_m(t):  return np.array([t[0,0], t[1,1], t[2,2],
                                     Utils.root2*t[1,2], Utils.root2*t[2,0], Utils.root2*t[0,1]])
    def m_to_t(t):  return np.array([[            t[0], Utils.rooth*t[5], Utils.rooth*t[4]], 
                                     [Utils.rooth*t[5],             t[1], Utils.rooth*t[3]], 
                                     [Utils.rooth*t[4], Utils.rooth*t[3],             t[2]]])
    # conversions between Voigt and Mandel vectors
    def m_to_ve(tm): 
        return np.array([tm[0], tm[1], tm[2], Utils.root2*tm[3], Utils.root2*tm[4], Utils.root2*tm[5]])
    def m_to_vs(tm): 
        return np.array([tm[0], tm[1], tm[2], Utils.rooth*tm[3], Utils.rooth*tm[4], Utils.rooth*tm[5]])
    def ve_to_m(ve): 
        return np.array([ve[0], ve[1], ve[2], Utils.rooth*ve[3], Utils.rooth*ve[4], Utils.rooth*ve[5]])
    def vs_to_m(vs): 
        return np.array([vs[0], vs[1], vs[2], Utils.root2*vs[3], Utils.root2*vs[4], Utils.root2*vs[5]])
    
    # Mandel routines
    delta_m = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    def dev_m(t): # deviator
        return t - (Utils.tr1_m(t)/3.0)*Utils.delta_m
    def cont_m(t1,t2): # full contraction t1_ij*t2_ij
        return t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2] + t1[3]*t2[3] + t1[4]*t2[4] + t1[5]*t2[5]
    def square_m(t): # t^2
        return np.array([t[0]**2 + 0.5*(t[5]**2 + t[4]**2),
                         t[1]**2 + 0.5*(t[3]**2 + t[5]**2),
                         t[2]**2 + 0.5*(t[4]**2 + t[3]**2),
                         Utils.rooth*t[5]*t[4] + t[3]*(t[1] + t[2]),
                         Utils.rooth*t[3]*t[5] + t[4]*(t[2] + t[0]),
                         Utils.rooth*t[4]*t[3] + t[5]*(t[0] + t[1])])

    def pij_m(t1,t2): return np.einsum("i,j->ij",t1,t2) # dyadic product
    II_m = pij_m(delta_m,delta_m)
    PP_m = np.eye(6) - II_m/3.0 # projection tensor, also differential d_dev(t) / dt
    # traces
    def tr1_m(t): # Mandel trace
        return t[0] + t[1] + t[2] 
    def tr2_m(t): # Mandel trace of square
        return t[0]**2 + t[1]**2 + t[2]**2 + t[3]**2 + t[4]**2 + t[5]**2
    def tr3_m(t): # Mandel trace of cube
        return t[0]**3 + t[1]**3 + t[2]**3 + \
               1.5*(t[0]*(t[4]**2 + t[5]**2) + t[1]*(t[5]**2 + t[3]**2) + t[2]*(t[3]**2 + t[4]**2)) + \
               3.0*Utils.rooth*t[3]*t[4]*t[5] 
    # invariants
    def i1_m(t): # 1st invariant
        return t[0] + t[1] + t[2]
    def i1sq_m(t): # square of 1st invariant
        return (t[0] + t[1] + t[2])**2
    def i2_m(t): # 2nd invariant
        return (Utils.tr2_m(t) - Utils.tr1_m(t)**2)/2.0 
    def i3_m(t): # 3rd invariant
        return (2.0*Utils.tr3_m(t) - 3.0*Utils.tr2_m(t)*Utils.tr1_m(t) + Utils.tr1_m(t)**3)/6.0 
    def j2_m(t): # 2nd invariant of deviator
        return (3.0*Utils.tr2_m(t) - Utils.tr1_m(t)**2)/6.0 
    def j3_m(t): # 3rd invariant of deviator
        return (9.0*Utils.tr3_m(t) - 9.0*Utils.tr2_m(t)*Utils.tr1_m(t) + 2.0*Utils.tr1_m(t)**3)/27.0 
    def det_m(t): # determinant (equal to i3)
        return t[0]*t[1]*t[2] + Utils.rooth*t[3]*t[4]*t[5] \
               - 0.5*(t[0]*(t[3]**2) + t[1]*(t[4]**2) + t[2]*(t[5]**2))  
    
    #mixed invariants
    def trm_ab_m(a,b): 
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5]
    def trm_a2b_m(a,b): 
        return b[0]*(a[0]**2 + 0.5*(a[5]**2 + a[4]**2)) + \
               b[1]*(a[1]**2 + 0.5*(a[3]**2 + a[5]**2)) + \
               b[2]*(a[2]**2 + 0.5*(a[4]**2 + a[3]**2)) + \
               b[3]*(a[1]*a[3] + a[2]*a[3] + Utils.rooth*a[4]*a[5]) + \
               b[4]*(a[2]*a[4] + a[0]*a[4] + Utils.rooth*a[5]*a[3]) + \
               b[5]*(a[0]*a[5] + a[1]*a[5] + Utils.rooth*a[3]*a[4])
    def trm_ab2_m(a,b): 
        return a[0]*(b[0]**2 + 0.5*(b[5]**2 + b[4]**2)) + \
               a[1]*(b[1]**2 + 0.5*(b[3]**2 + b[5]**2)) + \
               a[2]*(b[2]**2 + 0.5*(b[4]**2 + b[3]**2)) + \
               a[3]*(b[1]*b[3] + b[2]*b[3] + Utils.rooth*b[4]*b[5]) + \
               a[4]*(b[2]*b[4] + b[0]*b[4] + Utils.rooth*b[5]*b[3]) + \
               a[5]*(b[0]*b[5] + b[1]*b[5] + Utils.rooth*b[3]*b[4])
    def trm_a2b2_m(a,b): 
        return (a[0]**2)*(b[0]**2 + 0.5*(b[4]**2 + b[5]**2)) + \
               (a[1]**2)*(b[1]**2 + 0.5*(b[5]**2 + b[3]**2)) + \
               (a[2]**2)*(b[2]**2 + 0.5*(b[3]**2 + b[4]**2)) + \
               a[0]*a[4]*((b[0] + b[2])*b[4] + Utils.rooth*b[3]*b[5]) + \
               a[1]*a[5]*((b[1] + b[0])*b[5] + Utils.rooth*b[4]*b[3]) + \
               a[2]*a[3]*((b[2] + b[1])*b[3] + Utils.rooth*b[5]*b[4]) + \
               a[0]*a[5]*((b[0] + b[1])*b[5] + Utils.rooth*b[3]*b[4]) + \
               a[1]*a[3]*((b[1] + b[2])*b[3] + Utils.rooth*b[4]*b[5]) + \
               a[2]*a[4]*((b[2] + b[0])*b[4] + Utils.rooth*b[5]*b[3]) + \
               (a[3]**2)*(0.5*(b[1]**2 + b[2]**2 + b[3]**2) + 0.25*(b[4]**2 + b[5]**2)) + \
               (a[4]**2)*(0.5*(b[2]**2 + b[0]**2 + b[4]**2) + 0.25*(b[5]**2 + b[3]**2)) + \
               (a[5]**2)*(0.5*(b[0]**2 + b[1]**2 + b[5]**2) + 0.25*(b[3]**2 + b[4]**2)) + \
               a[3]*a[4]*(Utils.rooth*((b[0] + b[1])*b[5]) + 0.5*b[3]*b[4]) + \
               a[4]*a[5]*(Utils.rooth*((b[1] + b[2])*b[3]) + 0.5*b[4]*b[5]) + \
               a[5]*a[3]*(Utils.rooth*((b[2] + b[0])*b[4]) + 0.5*b[5]*b[3])
    def trm_abab_m(a,b):
        return (a[0]**2)*(b[0]**2) + (a[1]**2)*(b[1]**2) + (a[2]**2)*(b[2]**2) + \
               a[0]*a[1]*(b[5]**2) + a[1]*a[2]*(b[3]**2) + a[2]*a[0]*(b[4]**2) + \
               a[0]*a[3]*(Utils.root2*b[4]*b[5]) + \
               a[1]*a[4]*(Utils.root2*b[5]*b[3]) + \
               a[2]*a[5]*(Utils.root2*b[3]*b[4]) + \
               a[0]*a[4]*2.0*b[0]*b[4] + \
               a[1]*a[5]*2.0*b[1]*b[5] + \
               a[2]*a[3]*2.0*b[2]*b[3] + \
               a[0]*a[5]*2.0*b[0]*b[5] + \
               a[1]*a[3]*2.0*b[1]*b[3] + \
               a[2]*a[4]*2.0*b[2]*b[4] + \
               (a[3]**2)*(b[1]*b[2] + 0.5*(b[3]**2)) + \
               (a[4]**2)*(b[2]*b[0] + 0.5*(b[4]**2)) + \
               (a[5]**2)*(b[0]*b[1] + 0.5*(b[5]**2)) + \
               a[3]*a[4]*(Utils.root2*b[2]*b[5] + b[3]*b[4]) + \
               a[4]*a[5]*(Utils.root2*b[0]*b[3] + b[4]*b[5]) + \
               a[5]*a[3]*(Utils.root2*b[1]*b[4] + b[5]*b[3])

    # derivatives of traces and invariants
    def dtr1_m(t):  return Utils.delta_m
    def dtr2_m(t):  return 2.0*t
    def dtr3_m(t):  return 3.0*Utils.square_m(t)
    def di1_m(t):   return Utils.delta_m
    def di1sq_m(t): return 2.0*Utils.i1_m(t)*Utils.delta_m
    def di2_m(t):   return t - Utils.di1_m(t)*Utils.tr1_m(t)
    def di3_m(t):
        return np.array([t[1]*t[2] - 0.5*(t[3]**2),
                         t[2]*t[0] - 0.5*(t[4]**2),
                         t[0]*t[1] - 0.5*(t[5]**2),
                         Utils.rooth*t[4]*t[5] - t[3]*t[0],
                         Utils.rooth*t[5]*t[3] - t[4]*t[1],
                         Utils.rooth*t[3]*t[4] - t[5]*t[2]])
    def dj2_m(t): return t - Utils.di1_m(t)*Utils.tr1_m(t)/3.0
    def dj3_m(t):
        return (  3.0*Utils.dtr3_m(t) 
                - 3.0*Utils.dtr2_m(t)*Utils.tr1_m(t) 
                - 3.0*Utils.tr2_m(t)*Utils.dtr1_m(t) 
                + 2.0*(Utils.tr1_m(t)**2)*Utils.dtr1_m(t) ) / 9.0 
    
    #differentials of mixed invariants
    def dtrm_ab_a_m(a,b): # d_tr(ab) / da
        return b
    def dtrm_ab_b_m(a,b): # d_tr(ab) / db
        return a
    def dtrm_a2b_a_m(a,b): # d_tr(aab) / da
        return np.array([2.0*a[0]*b[0] + a[4]*b[4] + a[5]*b[5],
                         2.0*a[1]*b[1] + a[5]*b[5] + a[3]*b[3],
                         2.0*a[2]*b[2] + a[3]*b[3] + a[4]*b[4],
                         (a[1] + a[2])*b[3] + a[3]*(b[1] + b[2]) + Utils.rooth*(a[4]*b[5] + a[5]*b[4]),
                         (a[2] + a[0])*b[4] + a[4]*(b[2] + b[0]) + Utils.rooth*(a[5]*b[3] + a[3]*b[5]),
                         (a[0] + a[1])*b[5] + a[5]*(b[0] + b[1]) + Utils.rooth*(a[3]*b[4] + a[4]*b[3])])
    def dtrm_a2b_b_m(a,b): # d_tr(aab) / db
        return np.array([a[0]**2 + 0.5*(a[5]**2 + a[4]**2),          \
                         a[1]**2 + 0.5*(a[3]**2 + a[5]**2),          \
                         a[2]**2 + 0.5*(a[4]**2 + a[3]**2),          \
                         (a[1] + a[2])*a[3] + Utils.rooth*a[4]*a[5], \
                         (a[2] + a[0])*a[4] + Utils.rooth*a[5]*a[3], \
                         (a[0] + a[1])*a[5] + Utils.rooth*a[3]*a[4]])
    def dtrm_ab2_a_m(a,b): # d_tr(abb) / da
        return np.array([b[0]**2 + 0.5*(b[5]**2 + b[4]**2),          \
                         b[1]**2 + 0.5*(b[3]**2 + b[5]**2),          \
                         b[2]**2 + 0.5*(b[4]**2 + b[3]**2),          \
                         (b[1] + b[2])*b[3] + Utils.rooth*b[4]*b[5], \
                         (b[2] + b[0])*b[4] + Utils.rooth*b[5]*b[3], \
                         (b[0] + b[1])*b[5] + Utils.rooth*b[3]*b[4]])
    def dtrm_ab2_b_m(a,b): # d_tr(abb) / db
        return np.array([2.0*b[0]*a[0] + b[4]*a[4] + b[5]*a[5],
                         2.0*b[1]*a[1] + b[5]*a[5] + b[3]*a[3],
                         2.0*b[2]*a[2] + b[3]*a[3] + b[4]*a[4],
                         (a[1] + a[2])*b[3] + a[3]*(b[1] + b[2]) + Utils.rooth*(a[4]*b[5] + a[5]*b[4]),
                         (a[2] + a[0])*b[4] + a[4]*(b[2] + b[0]) + Utils.rooth*(a[5]*b[3] + a[3]*b[5]),
                         (a[0] + a[1])*b[5] + a[5]*(b[0] + b[1]) + Utils.rooth*(a[3]*b[4] + a[4]*b[3])])
    def dtrm_a2b2_a_m(a,b): # d_tr(aabb) / da
        return np.array([2.0*a[0]*(b[0]**2 + 0.5*(b[4]**2 + b[5]**2)) + \
                         a[4]*((b[0] + b[2])*b[4] + Utils.rooth*b[3]*b[5]) + \
                         a[5]*((b[0] + b[1])*b[5] + Utils.rooth*b[3]*b[4]),
                         2.0*a[1]*(b[1]**2 + 0.5*(b[5]**2 + b[3]**2)) + \
                         a[5]*((b[1] + b[0])*b[5] + Utils.rooth*b[4]*b[3]) + \
                         a[3]*((b[1] + b[2])*b[3] + Utils.rooth*b[4]*b[5]),
                         2.0*a[2]*(b[2]**2 + 0.5*(b[3]**2 + b[4]**2)) + \
                         a[3]*((b[2] + b[1])*b[3] + Utils.rooth*b[5]*b[4]) + \
                         a[4]*((b[2] + b[0])*b[4] + Utils.rooth*b[5]*b[3]),
                         (a[1] + a[2])*((b[1] + b[2])*b[3] + Utils.rooth*b[4]*b[5]) + \
                         2.0*a[3]*(0.5*(b[1]**2 + b[2]**2 + b[3]**2) + 0.25*(b[4]**2 + b[5]**2)) + \
                         a[4]*(Utils.rooth*(b[0] + b[1])*b[5] + 0.5*b[3]*b[4]) + \
                         a[5]*(Utils.rooth*(b[2] + b[0])*b[4] + 0.5*b[5]*b[3]),
                         (a[2] + a[0])*((b[0] + b[2])*b[4] + Utils.rooth*b[3]*b[5]) + \
                         2.0*a[4]*(0.5*(b[2]**2 + b[0]**2 + b[4]**2) + 0.25*(b[5]**2 + b[3]**2)) + \
                         a[3]*(Utils.rooth*(b[0] + b[1])*b[5] + 0.5*b[3]*b[4]) + \
                         a[5]*(Utils.rooth*(b[1] + b[2])*b[3] + 0.5*b[4]*b[5]),
                         (a[0] + a[1])*((b[0] + b[1])*b[5] + Utils.rooth*b[3]*b[4]) + \
                         2.0*a[5]*(0.5*(b[0]**2 + b[1]**2 + b[5]**2) + 0.25*(b[3]**2 + b[4]**2)) + \
                         a[4]*(Utils.rooth*(b[1] + b[2])*b[3] + 0.5*b[4]*b[5]) + \
                         a[3]*(Utils.rooth*(b[2] + b[0])*b[4] + 0.5*b[5]*b[3])])
    def dtrm_a2b2_b_m(a,b): # d_tr(aabb) / db
        return np.array([2.0*b[0]*(a[0]**2 + 0.5*(a[4]**2 + a[5]**2)) + \
                         b[4]*((a[0] + a[2])*a[4] + Utils.rooth*a[3]*a[5]) + \
                         b[5]*((a[0] + a[1])*a[5] + Utils.rooth*a[3]*a[4]),
                         2.0*b[1]*(a[1]**2 + 0.5*(a[5]**2 + a[3]**2)) + \
                         b[5]*((a[1] + a[0])*a[5] + Utils.rooth*a[4]*a[3]) + \
                         b[3]*((a[1] + a[2])*a[3] + Utils.rooth*a[4]*a[5]),
                         2.0*b[2]*(a[2]**2 + 0.5*(a[3]**2 + a[4]**2)) + \
                         b[3]*((a[2] + a[1])*a[3] + Utils.rooth*a[5]*a[4]) + \
                         b[4]*((a[2] + a[0])*a[4] + Utils.rooth*a[5]*a[3]),
                         (b[1] + b[2])*((a[1] + a[2])*a[3] + Utils.rooth*a[4]*a[5]) + \
                         2.0*b[3]*(0.5*(a[1]**2 + a[2]**2 + a[3]**2) + 0.25*(a[4]**2 + a[5]**2)) + \
                         b[4]*(Utils.rooth*(a[0] + a[1])*a[5] + 0.5*a[3]*a[4]) + \
                         b[5]*(Utils.rooth*(a[2] + a[0])*a[4] + 0.5*a[5]*a[3]),
                         (b[2] + b[0])*((a[0] + a[2])*a[4] + Utils.rooth*a[3]*a[5]) + \
                         2.0*b[4]*(0.5*(a[2]**2 + a[0]**2 + a[4]**2) + 0.25*(a[5]**2 + a[3]**2)) + \
                         b[3]*(Utils.rooth*(a[0] + a[1])*a[5] + 0.5*a[3]*a[4]) + \
                         b[5]*(Utils.rooth*(a[1] + a[2])*a[3] + 0.5*a[4]*a[5]),
                         (b[0] + b[1])*((a[0] + a[1])*a[5] + Utils.rooth*a[3]*a[4]) + \
                         2.0*b[5]*(0.5*(a[0]**2 + a[1]**2 + a[5]**2) + 0.25*(a[3]**2 + a[4]**2)) + \
                         b[4]*(Utils.rooth*(a[1] + a[2])*a[3] + 0.5*a[4]*a[5]) + \
                         b[3]*(Utils.rooth*(a[2] + a[0])*a[4] + 0.5*a[5]*a[3])])
    def dtrm_abab_a_m(a,b): # d_trm(abab) / da
        return np.array([2.0*a[0]*(b[0]**2) + a[1]*(b[5]**2) + a[2]*(b[4]**2) + \
                         a[3]*(Utils.root2*b[4]*b[5]) + 2.0*b[0]*(a[4]*b[4] + a[5]*b[5]),
                         2.0*a[1]*(b[1]**2) + a[2]*(b[3]**2) + a[0]*(b[5]**2) + \
                         a[4]*(Utils.root2*b[5]*b[3]) + 2.0*b[1]*(a[5]*b[5] + a[3]*b[3]),
                         2.0*a[2]*(b[2]**2) + a[0]*(b[4]**2) + a[1]*(b[3]**2) + \
                         a[5]*(Utils.root2*b[3]*b[4]) + 2.0*b[2]*(a[3]*b[3] + a[4]*b[4]),
                         a[0]*(Utils.root2*b[4]*b[5]) + 2.0*b[3]*(a[1]*b[1] + a[2]*b[2]) + \
                         2.0*a[3]*(b[1]*b[2] + 0.5*(b[3]**2)) + \
                         a[4]*(Utils.root2*b[2]*b[5] + b[3]*b[4]) + a[5]*(Utils.root2*b[1]*b[4] + b[5]*b[3]),
                         a[1]*(Utils.root2*b[5]*b[3]) + 2.0*b[4]*(a[2]*b[2] + a[0]*b[0]) + \
                         2.0*a[4]*(b[2]*b[0] + 0.5*(b[4]**2)) + \
                         a[5]*(Utils.root2*b[0]*b[3] + b[4]*b[5]) + a[3]*(Utils.root2*b[2]*b[5] + b[3]*b[4]),
                         a[2]*(Utils.root2*b[3]*b[4]) + 2.0*b[5]*(a[0]*b[0] + a[1]*b[1]) + \
                         2.0*a[5]*(b[0]*b[1] + 0.5*(b[5]**2)) + \
                         a[3]*(Utils.root2*b[1]*b[4] + b[5]*b[3]) + a[4]*(Utils.root2*b[0]*b[3] + b[4]*b[5])])                                  
    def dtrm_abab_b_m(a,b): # d_trm(abab) / db
        return np.array([2.0*b[0]*(a[0]**2) + b[1]*(a[5]**2) + b[2]*(a[4]**2) + \
                         b[3]*(Utils.root2*a[4]*a[5]) + 2.0*a[0]*(b[4]*a[4] + b[5]*a[5]),
                         2.0*b[1]*(a[1]**2) + b[2]*(a[3]**2) + b[0]*(a[5]**2) + \
                         b[4]*(Utils.root2*a[5]*a[3]) + 2.0*a[1]*(b[5]*a[5] + b[3]*a[3]),
                         2.0*b[2]*(a[2]**2) + b[0]*(a[4]**2) + b[1]*(a[3]**2) + \
                         b[5]*(Utils.root2*a[3]*a[4]) + 2.0*a[2]*(b[3]*a[3] + b[4]*a[4]),
                         b[0]*(Utils.root2*a[4]*a[5]) + 2.0*a[3]*(b[1]*a[1] + b[2]*a[2]) + \
                         2.0*b[3]*(a[1]*a[2] + 0.5*(a[3]**2)) + \
                         b[4]*(Utils.root2*a[2]*a[5] + a[3]*a[4]) + b[5]*(Utils.root2*a[1]*a[4] + a[5]*a[3]),
                         b[1]*(Utils.root2*a[5]*a[3]) + 2.0*a[4]*(b[2]*a[2] + b[0]*a[0]) + \
                         2.0*b[4]*(a[2]*a[0] + 0.5*(a[4]**2)) + \
                         b[5]*(Utils.root2*a[0]*a[3] + a[4]*a[5]) + b[3]*(Utils.root2*a[2]*a[5] + a[3]*a[4]),
                         b[2]*(Utils.root2*a[3]*a[4]) + 2.0*a[5]*(b[0]*a[0] + b[1]*a[1]) + \
                         2.0*b[5]*(a[0]*a[1] + 0.5*(a[5]**2)) + \
                         b[3]*(Utils.root2*a[1]*a[4] + a[5]*a[3]) + b[4]*(Utils.root2*a[0]*a[3] + a[4]*a[5])])                                  
    
    # second differentials - define others when needed
    def d2i1sq_m(t=0): # d2(i1^2) / dtdt
        return 2.0*Utils.II_m
    def d2j2_m(t=0): # d2j2 / dtdt
        return Utils.PP_m
    
    # tensorial signum of deviator and its differential
    def S_dev_m(t):
        return Utils.dev_m(t) / (Utils.tiny + np.sqrt(2.0*Utils.j2_m(t)))
    def dS_dev_m(t):
        if Utils.j2_m(t) < Utils.tiny: 
            temp = np.zeros[6,6]
        else:
            temp  =  Utils.PP_m / np.sqrt(2.0*Utils.j2_m(t))
            temp += -Utils.pij_m(Utils.dev_m(t),Utils.dev_m(t)) / ((2.0*Utils.j2_m(t))**1.5)
        return temp

class Commands: # each of these functions corresponds to a command in Hyperdrive
    def check(): # run checks
        check()
    def title(title): # job title
        hj.title = title
        print("Title: ", hj.title)
    def voigt(): # set Voigt mode
        hj.n_dim = 6
        hj.voigt = True
        print("Setting Voigt mode (n_dim = 6)")
    def model(model_temp): # specify model
        global hm
        print("Current directory:",os.getcwd())
        print("Current path directory:",os.path.curdir)
        if os.path.isfile("./" + model_temp + ".py"):
            hj.model = model_temp + ""
            print("Importing hyperplasticity model: ", hj.model)
            hm = import_module(hj.model)
            if hasattr(hm, "deriv"): hm.deriv()
            print_constants(hm)
            print("Description: ", hm.name)
            hj.n_dim = hm.ndim
        else:
            error("Model not found:" + model_temp)
        for fun in ["f", "g", "y", "w"]:
            if not hasattr(hm, fun): print("Function",fun,"not present in",hj.model)
        # override numerical differentiation increments if specified
        if hasattr(hm, "epsi"): si.eps = hm.epsi
        if hasattr(hm, "sigi"): si.sig = hm.sigi
        if hasattr(hm, "alpi"): si.alp = hm.alpi
        if hasattr(hm, "chii"): si.chi = hm.chii
        set_up_analytical()
        set_up_auto()
        set_up_num()
        choose_diffs()
        
# commands for setting options
    def prefs(p1, p2, p3): # set preferences for method of differentiation
        print("Setting differential preferences:")
        hj.prefs[0:3] = [p1, p2, p3]
        for i in range(3):
            print("Preference", i+1, hj.prefs[i])
        choose_diffs()
    def f_form(): # use f-functions for preference
        print("Setting f-form")
        hj.fform = True
        hj.gform = False
    def g_form(): # use g-functions for preference
        print("Setting g-form")
        hj.gform = True
        hj.fform = False
    def rate(): # use rate-dependent algorithm
        global hm
        print("Setting rate dependent analysis")
        hj.rate = True
    def rateind(): # use rate-independent algorithm
        print("Setting rate independent analysis")
        hj.rate = False
    def small(): # use small strain algorithms
        print("Setting small strain analysis")
        hj.large = False
    def large(): # use large strain algorithms
        global hm
        print("Setting finite strain analysis")
        hj.large = True
    def green(): # use Green large strain
        global hl
        print("Setting Green strain")
        hl.large_strain_def = "Green"
    def hencky(): # use Hencky large strain
        global hl
        print("Setting Hencky strain")
        hl.large_strain_def = "Hencky"
    def acc(accv): # set acceleration factor for rate-independent algorithm
        hj.acc = accv
        print("Setting acceleration factor:", hj.acc)
    def RKDP(): # set RKDP for general increment
        RKDP.opt = True
        print("Setting RKDP mode")
    def no_RKDP(): # unset RKDP for general increment (default)
        RKDP.opt = False
        print("Unsetting RKDP mode")
    def quiet(): # set quiet mode
        hj.quiet = True
        print("Setting quiet mode")
    def unquiet(): # unset quiet mode (default)
        hj.quiet = False
        print("Unsetting quiet mode")
    def auto_t_step(at_step): # set auto_t_step mode
        hj.auto_t_step = True
        hj.at_step = at_step
        print("Setting auto_t_step mode")
    def no_auto_t_step(): # unset auto_t_step (default)
        hj.auto_t_step = False 
        print("Unsetting auto_t_step mode")
    def colour(col): # select plot colour
        hj.colour = col
        print("Setting plot colour:", hj.colour)

# commands for setting constants
    def const(constv): # read model constants
        global hm
        if hj.model != "undefined":
            hm.const = constv
            print("Constants:", hm.const)
            if hasattr(hm, "deriv"): hm.deriv()
            hj.n_dim = hm.ndim
            print_constants(hm)
        else:
            pause("Cannot read constants: model undefined")
    def tweak(const_tweak, val_tweak): # tweak a single model constant
        global hm
        if hj.model != "undefined":
            for i in range(len(hm.const)):
                if const_tweak == hm.name_const[i]: 
                    hm.const[i] = val_tweak 
                    print("Tweaked constant: "+hm.name_const[i]+", value set to", hm.const[i])
            if hasattr(hm, "deriv"): hm.deriv()
            hj.n_dim = hm.ndim
            print_constants(hm)            
        else:
            pause("Cannot tweak constant: model undefined")
    def const_from_points(modtype, points, Einf=0.0, epsmax=0.0, HARM_R=0.0):
        global hm
        npt = len(points)
        epsd = np.zeros(npt)
        sigd = np.zeros(npt)
        for ipt in range(npt):
            epsd[ipt] = float(points[ipt][0])
            sigd[ipt] = float(points[ipt][1])
        const = derive_from_points(modtype,epsd,sigd,Einf,epsmax,HARM_R)
        print("Constants from points:", const)
        hm.const = const
        hm.const[0] = hj.n_dim
        if hasattr(hm, "deriv"): hm.deriv()
        print_constants(hm)
    def const_from_curve(modtype, curve, npt, sigmax, HARM_R, parami):
        global hm
        param = np.array(parami)
        epsd = np.zeros(npt)
        sigd = np.zeros(npt)
        print("Calculated points from curve:")
        for ipt in range(npt):
            sigd[ipt] = sigmax*float(ipt+1)/float(npt)
            if curve == "power":
                Ei     = param[0]
                epsmax = param[1]
                power  = param[2]
                epsd[ipt] = sigd[ipt]/Ei + (epsmax-sigmax/Ei)*(sigd[ipt]/sigmax)**power
            if curve == "jeanjean":                    
                Ei     = param[0]
                epsmax = param[1]
                A      = param[2]
                epsd[ipt] = sigd[ipt]/Ei + (epsmax-sigmax/Ei)*((np.atanh(np.tanh(A)*sigd[ipt]/sigmax)/A)**2)
            if curve == "PISA":
                Ei     = param[0]
                epsmax = param[1]
                n      = param[2]
                epspmax = epsmax - sigmax/Ei
                A = n*sigmax/(Ei*epspmax)
                B = -2.0*A*sigd[ipt]/sigmax + (1.0-n)*((1.0 + sigmax/(Ei*epspmax))**2)*(sigd[ipt]/sigmax - 1.0)
                C = A*(sigd[ipt]/sigmax)**2
                D = np.max(B**2-4.0*A*C, 0.0)
                epsd[ipt] = sigd[ipt]/Ei + 2.0*epspmax*C/(-B + np.sqrt(D))
            print(epsd[ipt],sigd[ipt])
        Einf = 0.5*(sigd[npt-1]-sigd[npt-2]) / (epsd[npt-1]-epsd[npt-2])
        epsmax = epsd[npt-1]
        hm.const = derive_from_points(modtype, epsd, sigd, Einf, epsmax, HARM_R)
        print("Constants from curve:", hm.const)
        if hasattr(hm, "deriv"): hm.deriv()
        print_constants(hm)
    def const_from_data(modtype, file, npt, maxsig, HARM_R): #not yet implemented
        global hm
        if hj.model != "undefined":
            data_text = Commands.read_csv(file)
            ttest   = np.array(len(data_text))
            epstest = np.array(len(data_text))
            sigtest = np.array(len(data_text))
            for i in range(len(data_text)):
                data_split = split(r'[ ,;]',data_text[i])
                ttest[i]   = float(data_split[0])
                epstest[i] = float(data_split[1])
                sigtest[i] = float(data_split[2])
            epsd = np.zeros(npt)
            sigd = np.zeros(npt)
            print("Calculating const from data")
            for ipt in range(npt):
                sigd[ipt] = maxsig*float(ipt+1)/float(npt)
                for i in range(len(data_text)-1):
                    if sigd[ipt] >= sigtest[i] and sigd[ipt] <= sigtest[i+1]:
                        epsd[ipt] = epstest[i] + (epstest[i+1]-epstest[i])*(sigtest[i+1]-sigd[ipt]) / \
                                                 (sigtest[i+1]-sigtest[i])
                print(epsd[ipt],sigd[ipt])
            Einf = 0.5*(sigd[npt-1]-sigd[npt-2])/(epsd[npt-1]-epsd[npt-2])
            epsmax = epsd[npt-1]
            hm.const = derive_from_points(modtype, epsd, sigd, Einf, epsmax, HARM_R)
            print("Constants from data:", hm.const)
            if hasattr(hm, "deriv"): hm.deriv()
            print_constants(hm)
        else:
            pause("Cannot obtain constants, model undefined")

# commands for starting and stopping processing
    def start(): # clear record and start
        global hm
        hj.curr_test = 0
        print("Starting a new test series")
        print("Test", hj.curr_test+1)
        hj.rec      = [[]]
        hj.test_rec = []
        hj.test_col = []
        hj.test_col.append(hj.colour)
        hj.high = [[]]
        hj.test_high = []
        s.t = 0.0
        if hasattr(hm, "deriv"): hm.deriv()
        calc_init()
        if hasattr(hm, "setalp"): hm.setalp(s.alp)
        record(s.eps, s.sig)
        Time.start = process_time()
        Time.last  = process_time()
    def restart(): # restart without clearing old tests from record
        global hm
        hj.curr_test += 1
        print("Restarting a new test")
        print("Test", hj.curr_test+1)
        hj.rec.append([])
        hj.test_col.append(hj.colour)
        hj.high.append([])
        s.t = 0.0
        if hasattr(hm, "deriv"): hm.deriv()
        calc_init()
        if hasattr(hm, "setalp"): hm.setalp(s.alp)
        record(s.eps, s.sig)
        time_now = process_time()
        print("Total time:",time_now - Time.start,", increment", time_now - Time.last)
        Time.last = process_time()
    def rec():
        print("Recording data")
        hj.recording = True
    def stoprec():
        temp = np.full(hj.n_dim, np.nan)
        record(temp, temp) # write a line of nan to split plots
        print("Stop recording data")
        hj.recording = False
    def end():
        time_now = process_time()
        print("Total time:",time_now - Time.start,", increment", time_now - Time.last)
        print("End of test")

# commands for initialisation
    def init_stress(sigin):
        sigi = np.array(sigin)
        s.sig = sig_from_voigt(sigi) # convert if necessary
        s.eps = eps_g(s.sig,s.alp)
        s.chi = chi_g(s.sig,s.alp)
        print("Initialising stress")
        print("Initial stress:", s.sig)
        print("Initial strain:", s.eps)
        delrec()
        record(s.eps, s.sig)
    def init_strain(epsin):
        epsi = np.array(epsin)
        s.eps = eps_from_voigt(epsi) # convert if necessary
        s.sig = sig_f(s.eps,s.alp)
        s.chi = chi_f(s.eps,s.alp)
        print("Initialising strain")
        print("Initial strain:", s.eps)
        print("Initial stress:", s.sig)
        delrec()
        record(s.eps, s.sig)
    def set_strain_reference():
        print("Setting strain reference point")
        hj.epsref =  deepcopy(s.eps)
        delrec()
        record(s.eps, s.sig)
    def save_state():
        global save
        save = deepcopy(s)
        print("State saved")
    def restore_state():
        global s
        s = deepcopy(save)
        delrec()
        record(s.eps, s.sig)
        print("State restored")
        
# commands for applying stress and strain increments
    def v_run(dt, deps, nprint, nsub, Smatin, Ematin, text1, text2): # utility used by v_txl_d, v_txl_u and v_dss
        Smat = deepcopy(Smatin)
        Emat = deepcopy(Ematin)
        Tdt = np.zeros(6)
        Tdt[5] = deps
        qprint(text1)
        qprint(text2, Tdt[5])
        run_general_inc(Smat, Emat, Tdt, dt, nprint, nsub)
    def v_txl_d(dt, deps1, nprint, nsub):
        Commands.v_run(dt, deps1,  nprint, nsub, S_txl_d, E_txl_d, "Drained triaxial test:",   "deps11 =")
    def v_txl_u(dt, deps1, nprint, nsub):
        Commands.v_run(dt, deps1,  nprint, nsub, S_txl_u, E_txl_u, "Undrained triaxial test:", "deps11 =")
    def v_dss(dt, dgamma, nprint, nsub):
        Commands.v_run(dt, dgamma, nprint, nsub, S_dss,   E_dss,   "Direct shear test:",       "dgam23 =")
    def general_inc(Smati, Emati, Tdti, dt, nprint, nsub):
        Smat = np.array(Smati)
        Emat = np.array(Emati)
        Tdt  = np.array(Tdti)
        qprint("General control increment:")
        qprint("S   =", Smat)
        qprint("E   =", Emat)
        qprint("Tdt =", Tdt)
        run_general_inc(Smat, Emat, Tdt, dt, nprint, nsub)
    def general_cyc(Smati, Emati, Tdti, tper, ctype, ncyc, nprint, nsub):
        if nprint%2 == 1: nprint += 1
        hnprint = int((nprint+0.5)/2)
        Smat = np.array(Smati)
        Emat = np.array(Emati)
        Tdt  = np.array(Tdti)
        qprint("General control cycles:")
        qprint("S   =", Smat)
        qprint("E   =", Emat)
        qprint("Tdt =", Tdt)
        hj.start_inc = True
        if ctype == "saw":
            dTdt  = Tdt  / float(hnprint) / float(nsub)
            dtper = tper / float(hnprint) / float(nsub)
            qprint("General cycles (saw): tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(int(hnprint)):
                    for isub in range(nsub): apply_general_inc(Smat, Emat, dTdt, dtper)
                    record(s.eps, s.sig)
                dTdt = -dTdt
        if ctype == "sine":
            dtper = tper / float(nprint) / float(nsub)
            qprint("General cycles (sine): tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)  /float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dTdt = Tdt * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): apply_general_inc(Smat, Emat, dTdt, dtper)
                    record(s.eps, s.sig)
        if ctype == "haversine":
            dtper = tper / float(nprint) / float(nsub)
            qprint("General cycles (haversine): tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)  /float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dTdt = Tdt * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): apply_general_inc(Smat, Emat, dTdt, dtper)
                    record(s.eps, s.sig)
        qprint("Cycles complete")
    def strain_it(t_inc, eps_vali, nprint, nsub, eback, text1, text2):
        eps_val = eps_from_voigt(eps_vali) # convert if necessary
        deps = (eps_val - eback) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        qprint(text1, nprint, "steps,", nsub, "substeps")
        qprint(text2, eps_val)
        qprint("deps =", deps)
        qprint("t_inc =", t_inc, ", dt =",dt)
        hj.start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): apply_strain_inc(deps, dt)
            record(s.eps, s.sig)
        qprint("Increment complete\n")
    def strain_inc(dt, deps, nprint, nsub):
        eps_val = np.array(deps)
        Commands.strain_it(dt, eps_val, nprint, nsub, 0.0, "Strain increment:", "eps_inc")
    def strain_targ(dt, epst, nprint, nsub):
        eps_val = np.array(epst)
        Commands.strain_it(dt, eps_val, nprint, nsub, s.eps, "Strain target:", "eps_targ")
    def strain_cyc(tper, eps_in, ctype, ncyc, nprint, nsub):
        eps_cyci = np.array(eps_in)
        eps_cyc = eps_from_voigt(eps_cyci) # convert if necessary
        if nprint%2 == 1: nprint += 1
        hnprint = int((nprint+0.5)/2)
        dt = tper / float(nprint*nsub)
        hj.start_inc = True
        if ctype == "saw":
            deps = eps_cyc / float(hnprint) / float(nsub)
            qprint("Strain cycle (saw):", eps_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(int(hnprint)):
                    for isub in range(nsub): apply_strain_inc(deps,dt)
                    record(s.eps, s.sig)
                deps = -deps
        if ctype == "sine":
            qprint("Strain cycle (sine):", eps_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    deps = eps_cyc * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): apply_strain_inc(deps,dt)
                    record(s.eps, s.sig)
        if ctype == "haversine":
            qprint("Strain cycle (haversine):", eps_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    deps = eps_cyc * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): apply_strain_inc(deps,dt)
                    record(s.eps, s.sig)
        qprint("Cycles complete")
    def stress_it(t_inc, sig_vali, nprint, nsub, sback, text1, text2): # utility used by stress_inc and stress_targ
        sig_val = sig_from_voigt(sig_vali) # convert if necessary
        dsig = (sig_val - sback) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        qprint(text1, nprint, "steps,", nsub, "substeps")
        qprint(text2, sig_val)
        qprint("dsig =", dsig)
        qprint("t_inc =", t_inc, ", dt =",dt)
        hj.start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): apply_stress_inc(dsig, dt)
            record(s.eps, s.sig)
        qprint("Increment complete\n")
    def large_inc(t_inc, Finn, nprint, nsub, mult=False, prop=False):
        from scipy.linalg import fractional_matrix_power
        Fin = np.array(Finn)
        if mult:
            #Ftarg = np.einsum("ik,kj->ij",Fin,hl.F)
            Ftarg = Fin @ hl.F
        else:
            Ftarg = Fin
        pprint(Ftarg,"Ftarg")
        if prop:
            if mult: 
                Ffac = Fin
            else:
                #Ffac = np.einsum("ik,kj->ij",Ftarg,np.linalg.inv(hl.F))
                Ffac = Ftarg @ np.linalg.inv(hl.F)
            Fdfac = fractional_matrix_power(Ffac, 1.0/float(nprint*nsub))
            Fdfac = np.real(Fdfac)
        else:
            dF = (Ftarg - hl.F) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        qprint("Large strain increment", nprint, "steps,", nsub, "substeps")
        #pprint(Ftarg,"Ftarg")
        #qprint("dF =", dF)
        qprint("t_inc =", t_inc, ", dt =",dt)
        hj.start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub):
                if prop:
                    #Fnew = np.einsum("ik,kj->ij",Fdfac,hl.F)
                    Fnew = Fdfac @ hl.F
                else:
                    Fnew = dF + hl.F
                if hj.fform:
                    apply_large_strain_inc_f(Fnew, dt)
                else:
                    apply_large_strain_inc_g(Fnew, dt)
            record(s.eps, s.sig)
        qprint("Increment complete\n")
    def stress_inc(dt, dsig, nprint, nsub):
        sig_val = np.array(dsig)
        Commands.stress_it(dt, sig_val, nprint, nsub, 0.0, "Stress increment:", "sig_inc")
    def stress_targ(dt, sigt, nprint, nsub):
        sig_val = np.array(sigt)
        Commands.stress_it(dt, sig_val, nprint, nsub, s.sig, "Stress target:", "sig_targ")
    def stress_cyc(tper, sig_in, ctype, ncyc, nprint, nsub):
        sig_cyci = np.array(sig_in)
        sig_cyc = sig_from_voigt(sig_cyci) # convert if necessary
        if nprint%2 == 1: nprint += 1
        hnprint = int((nprint+0.5)/2)
        dt = tper / float(nprint*nsub)
        hj.start_inc = True
        if ctype == "saw":
            dsig = sig_cyc / float(hnprint) / float(nsub)
            qprint("Stress cycle (saw):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(hnprint):
                    for isub in range(nsub): apply_stress_inc(dsig,dt)
                    record(s.eps, s.sig)
                dsig = -dsig
        if ctype == "sine":
            qprint("Stress cycle (sine):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dsig = sig_cyc * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): apply_stress_inc(dsig,dt)
                    record(s.eps, s.sig)
        if ctype == "haversine":
            qprint("Stress cycle (haversine):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dsig = sig_cyc * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): apply_stress_inc(dsig,dt)
                    record(s.eps, s.sig)
        qprint("Cycles complete")
    def test(filename, nsub, ptype): # hidden command used by stress_test and strain_test
        hj.test = True
        test_text = Commands.read_csv(filename)
        hj.test_rec = []
        hj.start_inc = True
        for iprint in range(len(test_text)):
            if np.mod(iprint,1000) == 0: print(" Step",iprint,"of",len(test_text))
            ltext = test_text[iprint].replace(" ","")
            test_split = split(r'[ ,;]',ltext)
            ttest = float(test_split[0])
            epstest = np.array([float(tex) for tex in test_split[1:hj.n_dim+1]])
            sigtest = np.array([float(tex) for tex in test_split[hj.n_dim+1:2*hj.n_dim+1]])
            if hj.start_inc:
                s.eps = eps_from_voigt(epstest)
                s.sig = sig_from_voigt(sigtest)
                delrec()
                record(s.eps, s.sig)
                recordt(epstest, sigtest)
                hj.start_inc= False
            else:
                dt = (ttest - s.t) / float(nsub)
                if ptype == "strain":
                    deps = (eps_from_voigt(epstest) - s.eps) / float(nsub)
                    for isub in range(nsub): apply_strain_inc(deps,dt)
                elif ptype == "stress":
                    dsig = (sig_from_voigt(sigtest) - s.sig) / float(nsub)
                    for isub in range(nsub): apply_stress_inc(dsig,dt)                    
                recordt(epstest, sigtest)
                record(s.eps, s.sig)
    def strain_test(filename, nsub):
        Commands.test(filename, nsub, "strain")
    def stress_test(filename, nsub):
        Commands.test(filename, nsub, "stress")
    def read_csv(filename): # hidden command used by _test and _path
        if filename[-4:] != ".csv": filename = filename + ".csv"
        print("Reading from file:", filename)
        test_file = open(filename,"r")
        test_text = test_file.readlines()
        test_file.close()
        return test_text
    def path(filename, nsub, ptype): # hidden command used by stress_path and strain_path
        test_text = Commands.read_csv(filename)
        hj.start_inc = True
        for iprint in range(len(test_text)):
            ltext = test_text[iprint].replace(" ","")
            test_split = split(r'[ ,;]',ltext)
            ttest = float(test_split[0])
            val = np.array([float(tex) for tex in test_split[1:hj.n_dim+1]])
            if hj.start_inc:
                if ptype == "strain":
                    s.eps = eps_from_voigt(val)
                elif ptype == "stress":
                    s.sig = sig_from_voigt(val)
                delrec()
                record(s.eps, s.sig)
                hj.start_inc= False
            else:
                dt = (ttest - s.t) / float(nsub)
                if ptype == "strain":
                    deps = (eps_from_voigt(val) - s.eps) / float(nsub)
                    for isub in range(nsub): apply_strain_inc(deps,dt)
                elif ptype == "stress":
                    dsig = (sig_from_voigt(val) - s.sig) / float(nsub)
                    for isub in range(nsub): apply_stress_inc(dsig,dt)
                record(s.eps, s.sig)
    def strain_path(filename, nsub):
        Commands.path(filename, nsub, "strain")
    def stress_path(filename, nsub):
        Commands.path(filename, nsub, "stress")
        
# commands for plotting and printing
    def printrec(oname = "hout_" + hj.model):
        results_print(oname)
    def pstress(form = "10.4"):
        text = "sig: ["
        for sigma in s.sig:
            text = text + ("{:"+form+"f}").format(sigma)
        text = text + "]"
        print(text)
    def pstrain(form = "10.6"):
        text = "eps: ["
        for epsilon in s.eps:
            text = text + ("{:"+form+"f}").format(epsilon)
        text = text + "]"
        print(text)
    def csv(oname = "hcsv_" + hj.model):
        results_csv(oname)
    def specialprint(oname = "hout_" + hj.model):
        if hasattr(hm,"specialprint"): 
            hm.specialprint(oname, s.eps, s.sig, s.alp, s.chi)
    def printstate():
        print("t   =", s.t)
        print("eps =", s.eps)
        print("sig =", s.sig)
        print("alp =", s.alp)
        print("chi =", s.chi)
    def plot(pname = "hplot_" + hj.model):
        results_plot(pname)
    def plotCS(pname = "hplot_" + hj.model):
        results_plotCS(pname)
    def graph(xax, yax, pname = "hplot_" + hj.model, xsize = 6, ysize = 6):
        axes = [xax, yax]
        results_graph(pname, axes, xsize, ysize)
    def specialplot(pname = "hplot_" + hj.model):
        if hasattr(hm,"specialplot"): 
            hm.specialplot(pname, hj.title, hj.rec, s.eps, s.sig, s.alp, s.chi)
    def high():
        print("Start highlighting plot")
        hj.hstart = len(hj.rec[hj.curr_test])-1
    def unhigh():
        print("Stop highlighting plot")
        hend = len(hj.rec[hj.curr_test])
        hj.high[hj.curr_test].append([hj.hstart,hend])
    def pause(*args):
        pause(*args)
        
    def open_csv(oname):
        global out_file
        if oname[-4:] != ".csv": oname = oname + ".csv"
        out_file = open(oname, 'w')
        titles = "t"
        for i in range(len(s.eps)):
            titles=titles+",eps_"+str(i+1)
        for i in range(len(s.sig)):
            titles=titles+",sig_"+str(i+1)
        titles = titles+"\n"
        out_file.write(titles)
    def write_csv():
        #time
        out_file.write(str(s.t)+",")
        #strain
        out_file.write(",".join([str(num) for num in s.eps])+",")
        #strain
        out_file.write(",".join([str(num) for num in s.sig])+"\n")
    def close_csv():
        out_file.close()       
        
def set_up_analytical():
    global udfde, udfda, ud2fdede, ud2fdeda, ud2fdade, ud2fdada
    global udgds, udgda, ud2gdsds, ud2gdsda, ud2gdads, ud2gdada
    global udyde, udyds, udyda, udydc, udwdc
    def set_h(name):
        qprint("  "+name+"...")
        if hasattr(hm,name): 
            hname = getattr(hm,name)
            qprint("        ...found")
        else:
            hname = undef
            qprint("        ...undefined")
        return hname
    print("Setting up analytical differentials (if available)")
    udfde = set_h("dfde")
    udfda = set_h("dfda")
    ud2fdede = set_h("d2fdede")
    ud2fdeda = set_h("d2fdeda")
    ud2fdade = set_h("d2fdade")
    ud2fdada = set_h("d2fdada")
    udgds = set_h("dgds")
    udgda = set_h("dgda")
    ud2gdsds = set_h("d2gdsds")
    ud2gdsda = set_h("d2gdsda")
    ud2gdads = set_h("d2gdads")
    ud2gdada = set_h("d2gdada")
    udyde = set_h("dyde")
    udyds = set_h("dyds")
    udyda = set_h("dyda")
    udydc = set_h("dydc")
    udwdc = set_h("dwdc")
    
def set_up_auto():
    global adfde, adfda, ad2fdede, ad2fdeda, ad2fdade, ad2fdada
    global adgds, adgda, ad2gdsds, ad2gdsda, ad2gdads, ad2gdada
    global adyde, adyds, adyda, adydc, adwdc    
    adfde = undef
    adfda = undef
    ad2fdede = undef
    ad2fdeda = undef
    ad2fdade = undef
    ad2fdada = undef
    adgds = undef
    adgda = undef
    ad2gdsds = undef
    ad2gdsda = undef
    ad2gdads = undef
    ad2gdada = undef
    adyde = undef
    adyds = undef
    adyda = undef
    adydc = undef
    adwdc = undef
    if not auto:
        print("\nAutomatic differentials not available")
        return
    print("\nSetting up automatic differentials")
    if hasattr(hm,"f"):
        if hasattr(hm,"f_exclude"):
            qprint("f excluded from auto-differentials in", hm.file)
        else:
            qprint("Setting up auto-differentials of f")
            qprint("  dfde...")
            adfde = ag.jacobian(hm.f,0)
            qprint("  dfda...")
            adfda = ag.jacobian(hm.f,1)
            qprint("  d2fdede...")
            ad2fdede = ag.jacobian(adfde,0)
            qprint("  d2fdeda...")
            ad2fdeda = ag.jacobian(adfde,1)
            qprint("  d2fdade...")
            ad2fdade = ag.jacobian(adfda,0)
            qprint("  d2fdada...")
            ad2fdada = ag.jacobian(adfda,1)
    else:
        qprint("f not specified in", hm.file)
    if hasattr(hm,"g"):
        if hasattr(hm,"g_exclude"):
            qprint("g excluded from auto-differentials in", hm.file)
        else:
            qprint("Setting up auto-differentials of g")
            qprint("  dgds...")
            adgds = ag.jacobian(hm.g,0)
            qprint("  dgda...")
            adgda = ag.jacobian(hm.g,1)
            qprint("  d2gdsds...")
            ad2gdsds = ag.jacobian(adgds,0)
            qprint("  d2gdsda...")
            ad2gdsda = ag.jacobian(adgds,1)
            qprint("  d2gdads...")
            ad2gdads = ag.jacobian(adgda,0)
            qprint("  d2gdada...")
            ad2gdada = ag.jacobian(adgda,1)
    else:
        qprint("g not specified in", hm.file)   
    if hasattr(hm,"y"):
        if hasattr(hm,"y_exclude"):
            qprint("y excluded from auto-differentials in", hm.file)
        else:
            qprint("Setting up auto-differentials of y")
            qprint("  dyde...")
            adyde = ag.jacobian(hm.y,0)
            qprint("  dyds...")
            adyds = ag.jacobian(hm.y,1)
            qprint("  dyda...")
            adyda = ag.jacobian(hm.y,2)
            qprint("  dydc...")
            adydc = ag.jacobian(hm.y,3)
    else:
        qprint("y not specified in", hm.file)    
    if hasattr(hm,"w"):
        if hasattr(hm,"w_exclude"):
            qprint("w excluded from auto-differentials in", hm.file)
        else:
            qprint("Setting up auto-differential of w")
            qprint("  dwdc...")
            adwdc = ag.jacobian(hm.w,3)
    else:
        qprint("w not specified in", hm.file)    

def set_up_num():
    global ndfde, ndfda, nd2fdede, nd2fdeda, nd2fdade, nd2fdada
    global ndgds, ndgda, nd2gdsds, nd2gdsda, nd2gdads, nd2gdada
    global ndyde, ndyds, ndyda, ndydc, ndwdc
    ndfde = undef
    ndfda = undef
    nd2fdede = undef
    nd2fdeda = undef
    nd2fdade = undef
    nd2fdada = undef
    ndgds = undef
    ndgda = undef
    nd2gdsds = undef
    nd2gdsda = undef
    nd2gdads = undef
    nd2gdada = undef
    ndyde = undef
    ndyds = undef
    ndyda = undef
    ndydc = undef
    ndwdc = undef
    print("\nSetting up numerical differentials")
    if hasattr(hm,"f"):
        qprint("Setting up numerical differentials of f")
        qprint("  dfde...")
        def ndfde(eps,alp): return numdiff_1(hm.f, eps, alp, si.eps)
        qprint("  dfda...")
        def ndfda(eps,alp): return numdiff_2(hm.f, eps, alp, si.alp)
        qprint("  d2fdede...")
        def nd2fdede(eps,alp): return numdiff2_1(hm.f, eps, alp, si.eps)
        qprint("  d2fdeda...")
        def nd2fdeda(eps,alp): return numdiff2_2(hm.f, eps, alp, si.eps, si.alp)
        qprint("  d2fdade...")
        def nd2fdade(eps,alp): return numdiff2_3(hm.f, eps, alp, si.eps, si.alp)
        qprint("  d2fdada...")
        def nd2fdada(eps,alp): return numdiff2_4(hm.f, eps, alp, si.alp)
    else:
        qprint("f not specified in", hm.file)
    if hasattr(hm,"g"):
        qprint("Setting up numerical differentials of g")
        qprint("  dgds...")
        def ndgds(sig,alp): return numdiff_1(hm.g, sig, alp, si.sig)
        qprint("  dgda...")
        def ndgda(sig,alp): return numdiff_2(hm.g, sig, alp, si.alp)
        qprint("  d2gdsds...")
        def nd2gdsds(sig,alp): return numdiff2_1(hm.g, sig, alp, si.sig)
        qprint("  d2gdsda...")
        def nd2gdsda(sig,alp): return numdiff2_2(hm.g, sig, alp, si.sig, si.alp)
        qprint("  d2gdads...")
        def nd2gdads(sig,alp): return numdiff2_3(hm.g, sig, alp, si.sig, si.alp)
        qprint("  d2gdada...")
        def nd2gdada(sig,alp): return numdiff2_4(hm.g, sig, alp, si.alp)
    else:
        qprint("g not specified in", hm.file)
    if hasattr(hm,"y"):
        qprint("Setting up numerical differentials of y")
        qprint("  dydc...")
        def ndydc(eps,sig,alp,chi): return numdiff_3(hm.y, eps,sig,alp,chi, si.chi)
        qprint("  dyde...")
        def ndyde(eps,sig,alp,chi): return numdiff_4e(hm.y, eps,sig,alp,chi, si.eps)
        qprint("  dyds...")
        def ndyds(eps,sig,alp,chi): return numdiff_4s(hm.y, eps,sig,alp,chi, si.sig)
        qprint("  dyda...")
        def ndyda(eps,sig,alp,chi): return numdiff_5(hm.y, eps,sig,alp,chi, si.alp)
    else:
        qprint("y not specified in", hm.file)
    if hasattr(hm,"w"):
        qprint("Setting up numerical differential of w")
        qprint("  dwdc...")
        def ndwdc(eps,sig,alp,chi): return numdiff_6(hm.w, eps,sig,alp,chi, si.chi)
    else:
        qprint("w not specified in", hm.file)

def choose_diffs():
    global dfde, dfda, d2fdede, d2fdeda, d2fdade, d2fdada
    global dgds, dgda, d2gdsds, d2gdsda, d2gdads, d2gdada
    global dyde, dyds, dyda, dydc, dwdc    
    def choose(name, ud, nd, ad):
        d = udef
        for i in range(3):
            if hj.prefs[i] == "analytical" and ud != udef: d = ud
            if hj.prefs[i] == "automatic"  and ad != udef: d = ad 
            if hj.prefs[i] == "numerical"  and nd != udef: d = nd
            if d != undef:
                qprint(name+":", hj.prefs[i])
                return d
        d = undef
        qprint(name+": undefined - will not run if this is required")
        return d        
    print("\nChoosing preferred differential methods")
    dfde    = choose("dfde",    udfde,    ndfde,    adfde)
    dfda    = choose("dfda",    udfda,    ndfda,    adfda)
    d2fdede = choose("d2fdede", ud2fdede, nd2fdede, ad2fdede)
    d2fdeda = choose("d2fdeda", ud2fdeda, nd2fdeda, ad2fdeda)
    d2fdade = choose("d2fdade", ud2fdade, nd2fdade, ad2fdade)
    d2fdada = choose("d2fdada", ud2fdada, nd2fdada, ad2fdada)
    dgds    = choose("dgds",    udgds,    ndgds,    adgds)
    dgda    = choose("dgda",    udgda,    ndgda,    adgda)
    d2gdsds = choose("d2gdsds", ud2gdsds, nd2gdsds, ad2gdsds)
    d2gdsda = choose("d2gdsda", ud2gdsda, nd2gdsda, ad2gdsda)
    d2gdads = choose("d2gdads", ud2gdads, nd2gdads, ad2gdads)
    d2gdada = choose("d2gdada", ud2gdada, nd2gdada, ad2gdada)
    dyde    = choose("dyde",    udyde,    ndyde,    adyde)
    dyds    = choose("dyds",    udyds,    ndyds,    adyds)
    dyda    = choose("dyda",    udyda,    ndyda,    adyda)
    dydc    = choose("dydc",    udydc,    ndydc,    adydc)
    dwdc    = choose("dwdc",    udwdc,    ndwdc,    adwdc)
    print("")

def sig_f(e, a): 
    return dfde(e, a)
def chi_f(e, a): # updated for weights
    return -np.einsum("N,Ni->Ni", hj.rwt, dfda(e, a))
def eps_g(s, a): 
    return -dgds(s, a)
def chi_g(s, a): # updated for weights
    return -np.einsum("N,Ni->Ni", hj.rwt, dgda(s, a))

class Check:
    npass = 0
    nfail = 0
    fails = []
    nzero = 0
    zeros = []
    nmiss = 0

def testch(text1, val1, text2, val2, failtext):
    print(text1, val1)
    print(text2, val2)
    if hasattr(val1, "shape") and hasattr(val2, "shape"):
        if val1.shape != val2.shape:
            print("\x1b[0;31mArrays different dimensions:",val1.shape,"and",val2.shape,"\x1b[0m")
            Check.nfail += 1
            Check.fails.append(failtext)
            pause()
            return
    maxv = np.maximum(np.max(val1),np.max(val2))
    minv = np.minimum(np.min(val1),np.min(val2))
    mv = np.maximum(maxv,-minv)
    #print("mv =",mv)
    testmat = np.isclose(val1, val2, rtol=0.0001, atol=0.000001*mv)
    if all(testmat.reshape(-1)): 
        print("\x1b[1;32m***PASSED***\n\x1b[0m")
        Check.npass += 1
    else:
        print("\x1b[1;31m",testmat,"\x1b[0m")
        print("\x1b[1;31m***FAILED***\n\x1b[0m")
        Check.nfail += 1
        Check.fails.append(failtext)
        pause()

def testch2(ch, v1, v2, failtext):
    try:
        _ = (e for e in v1)
        v1_iterable = True
    except TypeError:
        v1_iterable = False
    try:
        _ = (e for e in v2)
        v2_iterable = True
    except TypeError:
        v2_iterable = False
    if not v1_iterable:
        if v1 == "undefined":
            print(ch,"first variable undefined, comparison not possible")
            Check.nmiss += 1
            return
    if not v2_iterable:
        if v2 == "undefined":
            print(ch,"second variable undefined, comparison not possible")
            Check.nmiss += 1
            return
    if v1_iterable != v2_iterable:
        print(ch,"variables of different types, comparison not possible")
        Check.nfail += 1
        Check.fails.append(failtext)
        pause()
        return
    if v1_iterable and v2_iterable:
        if hasattr(v1,"shape") and hasattr(v2,"shape"):
            if v1.shape != v2.shape:
                print("\x1b[0;31mArrays different dimensions:",v1.shape,"and",v2.shape,"\x1b[0m")
                Check.nfail += 1
                Check.fails.append(failtext)
                pause()
                return
    maxv = np.maximum(np.max(v1),np.max(v2))
    minv = np.minimum(np.min(v1),np.min(v2))
    mv = np.maximum(maxv,-minv)
    testmat = np.isclose(v1, v2, rtol=0.0001, atol=0.001*mv)
    if all(testmat.reshape(-1)): 
        print(ch,"\x1b[1;32m***PASSED***\x1b[0m")
        Check.npass += 1
        if np.max(v1) == 0.0 and np.min(v1) == 0.0:
            Check.nzero +=1
            Check.zeros.append(failtext)
    else:
        print(ch,"\x1b[1;31m***FAILED***\x1b[0m")
        print("test\x1b[1;31m",testmat,"\x1b[0m")
        pprint(v1,"va11","14.6")
        pprint(v2,"val2","14.6")
        Check.nfail += 1
        Check.fails.append(failtext)
        pause()
    
def testch1(text, hval, aval, nval):
    pprint(hval,"Analytical "+text, "14.6")
    pprint(aval,"Automatic  "+text, "14.6")
    pprint(nval,"Numerical  "+text, "14.6")
    def ex(fun):
        if hasattr(fun,"shape"):
            exists = True
        else:
            exists = (fun != "undefined")
        return exists
    if ex(hval) and ex(aval):
        testch2("analytical v. automatic: ", hval, aval, "analytical v. automatic:  "+text)
    if ex(aval) and ex(nval):
        testch2("automatic  v. numerical: ", aval, nval, "automatic  v. numerical:  "+text)
    if ex(nval) and ex(hval):
        testch2("numerical  v. analytical:", nval, hval, "numerical  v. analytical: "+text)

def check(arg="unknown"):
    global hm
    
    Check.npass = 0
    Check.nfail = 0
    Check.fails = []
    Check.nzero = 0
    Check.zeros = []
    Check.nmiss = 0
    hj.model = hm.file
    hj.n_dim = hm.ndim   
    # override numerical differentiation increments if specified
    if hasattr(hm, "epsi"): si.eps = hm.epsi
    if hasattr(hm, "sigi"): si.sig = hm.sigi
    if hasattr(hm, "alpi"): si.alp = hm.alpi
    if hasattr(hm, "chii"): si.chi = hm.chii
    
    set_up_analytical()
    set_up_auto()
    set_up_num()
    choose_diffs()
    
    sig, eps, chi, alp = calc_init()
    
    if hasattr(hm, "check_eps"): 
        eps = hm.check_eps
    else:
        text = input("Input test strain: ",)
        eps = np.array([float(item) for item in split(r'[ ,;]',text)])
    pprint(eps, "eps =", "12.6")
    
    if hasattr(hm, "check_sig"): 
        sig = hm.check_sig
    else:
        text = input("Input test stress: ",)
        sig = np.array([float(item) for item in split(r'[ ,;]',text)])
    pprint(sig, "sig =", "12.4")
    
    if hasattr(hm, "check_alp"): 
        alp = hm.check_alp
    pprint(alp, "alp =", "12.6")
    
    if hasattr(hm, "check_chi"): 
        chi = hm.check_chi
    pprint(chi, "chi =", "12.4")
    
    print_constants(hm)   
    input("Hit ENTER to start checks",)
    
    if hasattr(hm, "f") and hasattr(hm, "g"):
        print("Checking consistency of f and g formulations ...\n")
        print("Checking inverse relations...")
        sigt = sig_f(eps, alp)
        epst = eps_g(sig, alp)
        print("Stress from strain:", sigt)
        print("Strain from stress:", epst)
        testch("Strain:             ", eps, "-> stress -> strain:", eps_g(sigt, alp),"strain v stress -> strain")
        testch("Stress:             ", sig, "-> strain -> stress:", sig_f(epst, alp),"stress v strain -> stress")
    
        print("Checking chi from different routes...")
        testch("from eps and f:", chi_f(eps, alp), "from g:        ", chi_g(sigt, alp),"chi from eps")
        testch("from sig and g:", chi_g(sig, alp), "from f:        ", chi_f(epst, alp),"chi from sig")
    
        print("Checking Legendre transform...")
        W = np.einsum("i,i->", sigt, eps)
        testch("from eps: f + (-g) =", hm.f(eps,alp) - hm.g(sigt,alp), "sig.eps =           ", W,"f + (-g): 1")
        W = np.einsum("i,i->", sig, epst)
        testch("from sig: f + (-g) =", hm.f(epst,alp) - hm.g(sig,alp), "sig.eps =           ", W,"f + (-g): 2")
    
        print("Checking elastic stiffness and compliance at specified strain...")
        unit = np.eye(hj.n_dim)
        D =  d2fdede(eps ,alp)
        C = -d2gdsds(sigt,alp)
        DC = np.einsum("ij,jk->ik",D,C)
        print("Stiffness matrix D  =\n",D)
        print("Compliance matrix C =\n",C)
        testch("Product DC = \n",DC,"unit matrix =\n", unit,"DC")
        print("and at specified stress...")
        C = -d2gdsds(sig, alp)
        D =  d2fdede(epst,alp)
        CD = np.einsum("ij,jk->ik",C,D)
        print("Compliance matrix D =\n",C)
        print("Stiffness matrix C  =\n",D)
        testch("Product CD = \n",CD,"unit matrix =\n", unit,"CD")
    
    if hasattr(hm, "f"):
        print("Checking derivatives of f...")
        testch1("dfde",    udfde(eps,alp),    adfde(eps,alp),    ndfde(eps,alp))
        testch1("dfda",    udfda(eps,alp),    adfda(eps,alp),    ndfda(eps,alp))
        testch1("d2fdede", ud2fdede(eps,alp), ad2fdede(eps,alp), nd2fdede(eps,alp))
        testch1("d2fdeda", ud2fdeda(eps,alp), ad2fdeda(eps,alp), nd2fdeda(eps,alp))
        testch1("d2fdade", ud2fdade(eps,alp), ad2fdade(eps,alp), nd2fdade(eps,alp))
        testch1("d2fdada", ud2fdada(eps,alp), ad2fdada(eps,alp), nd2fdada(eps,alp))
        print("Checking order of differentiation")
        reshape = np.einsum("ijk->kij",d2fdade(eps,alp))
        testch("d2fdeda",d2fdeda(eps,alp),"d2fdade(reshaped)", reshape,"order")
    if hasattr(hm, "g"):
        print("Checking derivatives of g...")
        testch1("dgds",    udgds(sig,alp),    adgds(sig,alp),    ndgds(sig,alp))
        testch1("dgda",    udgda(sig,alp),    adgda(sig,alp),    ndgda(sig,alp))
        testch1("d2gdsds", ud2gdsds(sig,alp), ad2gdsds(sig,alp), nd2gdsds(sig,alp))
        testch1("d2gdsda", ud2gdsda(sig,alp), ad2gdsda(sig,alp), nd2gdsda(sig,alp))
        testch1("d2gdads", ud2gdads(sig,alp), ad2gdads(sig,alp), nd2gdads(sig,alp))
        testch1("d2gdada", ud2gdada(sig,alp), ad2gdada(sig,alp), nd2gdada(sig,alp))       
        print("Checking order of differentiation")
        reshape = np.einsum("ijk->kij",d2gdads(sig,alp))
        testch("d2gdsda",d2gdsda(sig,alp),"d2gdads(reshaped)", reshape,"order")
    if hasattr(hm, "y"):
        print("Checking derivatives of y...")
        testch1("dyde", udyde(eps,sig,alp,chi), adyde(eps,sig,alp,chi), ndyde(eps,sig,alp,chi))
        testch1("dyds", udyds(eps,sig,alp,chi), adyds(eps,sig,alp,chi), ndyds(eps,sig,alp,chi))
        testch1("dyda", udyda(eps,sig,alp,chi), adyda(eps,sig,alp,chi), ndyda(eps,sig,alp,chi))
        testch1("dydc", udydc(eps,sig,alp,chi), adydc(eps,sig,alp,chi), ndydc(eps,sig,alp,chi))                
    if hasattr(hm, "w"):
        print("Checking derivative of w...")
        testch1("dwdc", udwdc(eps,sig,alp,chi), adwdc(eps,sig,alp,chi), ndwdc(eps,sig,alp,chi))
        
    print("Checks complete for:",hj.model+",",
          Check.npass,"passed,",
          Check.nfail,"failed,",
          Check.nmiss,"missed checks")
    if not hasattr(hm, "f"): print("hm.f not present")
    if not hasattr(hm, "g"): print("hm.g not present")
    if not hasattr(hm, "y"): print("hm.y not present")
    if not hasattr(hm, "w"): print("hm.w not present")
    if Check.nfail > 0:
        print("Summary of fails:")
        for text in Check.fails: print("  ",text)
    if Check.nzero > 0:
        print("Warning - zero values (may not have been rigorously checked):")
        for text in Check.zeros: print("  ",text)
    input("Checks complete, hit ENTER",)
            
def print_constants(hm):
    print("Constants for model:",hm.file)
    print(hm.const)
    print("Derived values:")
    for i in range(len(hm.const)):
        print(hm.name_const[i] + " =", hm.const[i])

def calcsum(vec):
    global nstep, nsub, deps, sigrec, const, hm, var
    sumerr = 0.0
    print(vec)
    hm.const = deepcopy(const)
    for i in range(hj.n_int):
       hm.const[2+2*i] = vec[i]
    if hasattr(hm, "deriv"): hm.deriv()
    s.eps = 0.0
    s.sig = 0.0
    s.alp = np.zeros(hj.n_int)
    s.chi = np.zeros(hj.n_int)
    if "_h" in var:
        s.alp = np.zeros(hj.n_int+1)
        s.chi = np.zeros(hj.n_int+1)
    if "_cbh" in var:
        s.alp = np.zeros(hj.n_int+1)
        s.chi = np.zeros(hj.n_int+1)
    for step in range(nstep):
        for i in range(nsub):
            strain_inc_f_spec(deps)
        error = s.sig - sigrec[step]
        sumerr += error**2
    return sumerr

def solve_L(yo, Lmatp, Lrhsp):
    Lmat = np.eye(hm.n_y)                    # initialise matrix and RHS for elastic solution
    Lrhs = np.zeros(hm.n_y)
    for N in range(hm.n_y):                  # loop over yield surfaces
        if yo[N] > hj.ytol:                  # if this surface is yielding:
            Lmat[N] = Lmatp[N]               # over-write line in matrix and RHS with plastic solution
            Lrhs[N] = Lrhsp[N]
    L = np.linalg.solve(Lmat, Lrhs)          # solve for plastic multipliers
    L = np.array([max(Lv, 0.0) for Lv in L]) # make plastic multipliers non-negative
    return L

def optim(base, variant):
    global sigrec, hm, nstep, nsub, const, deps, var
    var = variant
    sigrec = np.zeros(nstep)
    print("Calculate base curve from", base)
    hm = import_module(base)    
    #hm.setvals()
    ndim=1
    print(const)
    hm.const = deepcopy(const[:2+2*hj.n_int])
    if hasattr(hm, "deriv"): hm.deriv()
    print(hm.recip_k, hj.n_int)
    s.eps = np.zeros(ndim)
    s.sig = np.zeros(ndim)
    s.alp = np.zeros([hj.n_int,ndim])
    s.chi = np.zeros([hj.n_int,ndim])
    for step in range(nstep):
       for i in range(nsub): strain_inc_f_spec(deps)
       sigrec[step] = s.sig       
    print("optimise", variant)
    hm = import_module(variant)
    #hm.setvals()
    vec = np.zeros(hj.n_int)
    for i in range(hj.n_int): 
        vec[i] = const[2*i+2]
    print(vec,calcsum(vec))
    bnds = optimize.Bounds(0.0001,np.inf)
    resultop = optimize.minimize(calcsum, vec, method='L-BFGS-B', bounds=bnds)
    vec = resultop.x
    print(vec,calcsum(vec))
    for i in range(hj.n_int): 
        const[2+2*i] = resultop.x[i]
    return const
# = derive_from_points(modtype, epsd, sigd, Einf, epsmax, HARM_R)

def derive_from_points(modeltype, epsin, sigin, Einf, epsmax, HARM_R):
    global nstep, nsub, const, deps
    print("... deriving constants from points")
    eps = np.array(epsin)
    sig = np.array(sigin)
    hj.n_int = len(eps)
    E = np.zeros(hj.n_int+1)
    E[0] = sig[0] / eps[0]
    for i in range(1,hj.n_int):
        E[i] = (sig[i]-sig[i-1]) / (eps[i]-eps[i-1])
    E[hj.n_int] = Einf
    print("eps =",eps)
    print("sig =",sig)
    print("E   =",E)
    k = np.zeros(hj.n_int)
    H = np.zeros(hj.n_int)
    if "ser" in modeltype:
        print("Series parameters")
        E0 = E[0]
        for i in range(hj.n_int):
            k[i] = sig[i]
            H[i] = E[i+1]*E[i]/(E[i] - E[i+1])
        const = [0, E0, hj.n_int]
        for i in range(hj.n_int):
            const.append(round(k[i],6))
            const.append(round(H[i],6))
        base = "hnepmk_ser"
    elif "par" in modeltype:
        print("Parallel parameters")
        for i in range(hj.n_int):
            H[i] = E[i] - E[i+1]                
            k[i] = eps[i]*H[i]
        const = [0, Einf, hj.n_int]
        for i in range(hj.n_int):
            const.append(round(k[i],6))
            const.append(round(H[i],6))
        base = "hnepmk_par"
    elif "nest" in modeltype:
        print("Nested parameters")
        E0 = E[0]
        for i in range(hj.n_int):
            k[i] = sig[i] - sig[i-1]
            H[i] = E[i+1]*E[i]/(E[i] - E[i+1])                
        k[0] = sig[0]
        const = [0, E0, hj.n_int]
        for i in range(hj.n_int):
            const.append(round(k[i],6))
            const.append(round(H[i],6))
        base = "hnepmk_nest"
    if "_b" in modeltype: #now optimise for bounding surface model
        print("Optimise parameters for _b option")
        nstep = 100
        nsub = 10
        if epsmax == 0.0:
            epsmax = 1.5*eps[hj.n_int-1]
            print("setting epsmax =",epsmax)
        deps = epsmax / float(nstep*nsub)
        const = optim(base, modeltype)
        for i in range(2,2+2*hj.n_int):
            const[i] = round(const[i],6)
    if "_h" in modeltype: #now optimise for HARM model
        print("Optimise parameters for _h option")
        nstep = 100
        nsub = 10
        if epsmax == 0.0:
            epsmax = 1.5*eps[hj.n_int-1]
            print("setting epsmax =",epsmax)
        deps = epsmax / float(nstep*nsub)
        const.append(HARM_R)
        const = optim(base, modeltype)
        for i in range(2,2+2*hj.n_int):
            const[i] = round(const[i],6)
    if "_cbh" in modeltype: #now optimise for bounding HARM model
        print("Optimise parameters for _cbh option")
        nstep = 100
        nsub = 10
        if epsmax == 0.0:
            epsmax = 1.5*eps[hj.n_int-1]
            print("setting epsmax =",epsmax)
        deps = epsmax / float(nstep*nsub)
        const.append(HARM_R)
        const = optim(base, modeltype)
        for i in range(2,2+2*hj.n_int):
            const[i] = round(const[i],6)
    return const

def numdiff_1(fun, var, alp, vari):
    num = np.zeros([hj.n_dim])
    for i in range(hj.n_dim):
        var1 = deepcopy(var)
        var2 = deepcopy(var)
        var1[i] = var[i] - vari  
        var2[i] = var[i] + vari    
        f1 = fun(var1,alp)
        f2 = fun(var2,alp)
        num[i] = (f2 - f1) / (2.0*vari)
    return num
def numdiff_2(fun, var, alp, alpi):
    num = np.zeros([hj.n_int,hj.n_dim])
    for k in range(hj.n_int):
        for i in range(hj.n_dim):
            alp1 = deepcopy(alp)
            alp2 = deepcopy(alp)
            alp1[k,i] = alp[k,i] - alpi  
            alp2[k,i] = alp[k,i] + alpi
            f1 = fun(var,alp1)
            f2 = fun(var,alp2)
            num[k,i] = (f2 - f1) / (2.0*alpi)
    return num
def numdiff_3(fun, eps,sig,alp,chi, chii):
    num = np.zeros([hj.n_y,hj.n_int,hj.n_dim])
    for k in range(hj.n_int):
        for i in range(hj.n_dim):   
            chi1 = deepcopy(chi)
            chi2 = deepcopy(chi)
            chi1[k,i] = chi[k,i] - chii  
            chi2[k,i] = chi[k,i] + chii
            f1 = fun(eps,sig,alp,chi1)
            f2 = fun(eps,sig,alp,chi2)
            for l in range(hj.n_y):
                num[l,k,i] = (f2[l] - f1[l]) / (2.0*chii)
    return num
def numdiff_4e(fun, eps,sig,alp,chi, vari):
    num = np.zeros([hj.n_y,hj.n_dim])
    for i in range(hj.n_dim):   
        var1 = deepcopy(eps)
        var2 = deepcopy(eps)
        var1[i] = eps[i] - vari  
        var2[i] = eps[i] + vari
        f1 = fun(var1,sig,alp,chi)
        f2 = fun(var2,sig,alp,chi)
        for l in range(hj.n_y):
            num[l,i] = (f2[l] - f1[l]) / (2.0*vari)
    return num         
def numdiff_4s(fun, eps,sig,alp,chi, vari):
    num = np.zeros([hj.n_y,hj.n_dim])
    for i in range(hj.n_dim):   
        var1 = deepcopy(sig)
        var2 = deepcopy(sig)
        var1[i] = sig[i] - vari  
        var2[i] = sig[i] + vari
        f1 = fun(eps,var1,alp,chi)
        f2 = fun(eps,var2,alp,chi)
        for l in range(hj.n_y):
            num[l,i] = (f2[l] - f1[l]) / (2.0*vari)
    return num         
def numdiff_5(fun, eps,sig,alp,chi, alpi):
    num = np.zeros([hj.n_y,hj.n_int,hj.n_dim])
    for k in range(hj.n_int):
        for i in range(hj.n_dim):   
            alp1 = deepcopy(alp)
            alp2 = deepcopy(alp)
            alp1[k,i] = alp[k,i] - alpi  
            alp2[k,i] = alp[k,i] + alpi
            f1 = fun(eps,sig,alp1,chi)
            f2 = fun(eps,sig,alp2,chi)
            for l in range(hj.n_y):
                num[l,k,i] = (f2[l] - f1[l]) / (2.0*alpi)
    return num                
def numdiff_6(fun, eps,sig,alp,chi, chii):
    num = np.zeros([hj.n_int,hj.n_dim])
    for k in range(hj.n_int):
        for i in range(hj.n_dim):   
            chi1 = deepcopy(chi)
            chi2 = deepcopy(chi)
            chi1[k,i] = chi[k,i] - chii  
            chi2[k,i] = chi[k,i] + chii
            f1 = fun(eps,sig,alp,chi1)
            f2 = fun(eps,sig,alp,chi2)
            num[k,i] = (f2 - f1) / (2.0*chii)
    return num
def numdiff_6a(fun, eps,sig,alp,chi, chii):
    f0 = fun(eps,sig,alp,chi)
    num = np.zeros([hj.n_int,hj.n_dim])
    for k in range(hj.n_int):
        for i in range(hj.n_dim):   
            chi1 = deepcopy(chi)
            chi2 = deepcopy(chi)
            chi1[k,i] = chi[k,i] - chii  
            chi2[k,i] = chi[k,i] + chii
            f1 = fun(eps,sig,alp,chi1)
            f2 = fun(eps,sig,alp,chi2)
            if abs(f2-f0) > abs(f1-f0):
                num[k,i] = (f2 - f0) / chii
            else:
                num[k,i] = (f0 - f1) / chii
    return num
def numdiff2_1(fun, var, alp, vari):
    num = np.zeros([hj.n_dim,hj.n_dim])
    for i in range(hj.n_dim):
        for j in range(hj.n_dim):
            if i==j:
                var1 = deepcopy(var)
                var3 = deepcopy(var)
                var1[i] = var[i] - vari  
                var3[i] = var[i] + vari
                f1 = fun(var1,alp)
                f2 = fun(var, alp)
                f3 = fun(var3,alp)
                num[i,i] = (f1 - 2.0*f2 + f3) / (vari**2)
            else:
                var1 = deepcopy(var)
                var2 = deepcopy(var)
                var3 = deepcopy(var)
                var4 = deepcopy(var)
                var1[i] = var[i] - vari  
                var1[j] = var[j] - vari  
                var2[i] = var[i] - vari  
                var2[j] = var[j] + vari  
                var3[i] = var[i] + vari  
                var3[j] = var[j] - vari  
                var4[i] = var[i] + vari  
                var4[j] = var[j] + vari  
                f1 = fun(var1,alp)
                f2 = fun(var2,alp)
                f3 = fun(var3,alp)
                f4 = fun(var4,alp)
                num[i,j] = (f1 - f2 - f3 + f4) / (4.0*(vari**2))
    return num
def numdiff2_2(fun, var, alp, vari, alpi):
    num = np.zeros([hj.n_dim,hj.n_int,hj.n_dim])
    for N in range(hj.n_int):
        for i in range(hj.n_dim):
            for j in range(hj.n_dim):
                var1 = deepcopy(var)
                var2 = deepcopy(var)
                alp1 = deepcopy(alp)
                alp2 = deepcopy(alp)
                var1[i] = var[i] - vari  
                var2[i] = var[i] + vari
                alp1[N,j] = alp[N,j] - alpi  
                alp2[N,j] = alp[N,j] + alpi
                f1 = fun(var1,alp1)
                f2 = fun(var2,alp1)
                f3 = fun(var1,alp2)
                f4 = fun(var2,alp2)
                num[i,N,j] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
    return num
def numdiff2_3(fun, var, alp, vari, alpi):
    num = np.zeros([hj.n_int,hj.n_dim,hj.n_dim])
    for N in range(hj.n_int):
        for i in range(hj.n_dim):
            for j in range(hj.n_dim):
                var1 = deepcopy(var)
                var2 = deepcopy(var)
                alp1 = deepcopy(alp)
                alp2 = deepcopy(alp)
                var1[i] = var[i] - vari  
                var2[i] = var[i] + vari
                alp1[N,j] = alp[N,j] - alpi  
                alp2[N,j] = alp[N,j] + alpi
                f1 = fun(var1,alp1)
                f2 = fun(var2,alp1)
                f3 = fun(var1,alp2)
                f4 = fun(var2,alp2)
                num[N,j,i] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
    return num
def numdiff2_4(fun, var, alp, alpi):
    num = np.zeros([hj.n_int,hj.n_dim,hj.n_int,hj.n_dim])
    for N in range(hj.n_int):
        for M in range(hj.n_int):
            for i in range(hj.n_dim):
                for j in range(hj.n_dim):
                    if N==M and i==j:
                        alp1 = deepcopy(alp)
                        alp3 = deepcopy(alp)
                        alp1[N,i] = alp[N,i] - alpi  
                        alp3[N,i] = alp[N,i] + alpi
                        f1 = fun(var,alp1)
                        f2 = fun(var,alp)
                        f3 = fun(var,alp3)
                        num[N,i,N,i] = (f1 - 2.0*f2 + f3) / (alpi**2)
                    else:
                        alp1 = deepcopy(alp)
                        alp2 = deepcopy(alp)
                        alp3 = deepcopy(alp)
                        alp4 = deepcopy(alp)
                        alp1[N,i] = alp[N,i] - alpi  
                        alp1[M,j] = alp[M,j] - alpi  
                        alp2[N,i] = alp[N,i] - alpi  
                        alp2[M,j] = alp[M,j] + alpi  
                        alp3[N,i] = alp[N,i] + alpi  
                        alp3[M,j] = alp[M,j] - alpi  
                        alp4[N,i] = alp[N,i] + alpi  
                        alp4[M,j] = alp[M,j] + alpi  
                        f1 = fun(var,alp1)
                        f2 = fun(var,alp2)
                        f3 = fun(var,alp3)
                        f4 = fun(var,alp4)
                        num[N,i,M,j] = (f1 - f2 - f3 + f4) / (4.0*(alpi**2))
    return num
    
def printderivs():
    print("t      ", s.t)
    print("eps    ", s.eps)
    print("sig    ", s.sig)
    print("alp    ", s.alp)
    print("chi    ", s.chi)
    print("f      ", hm.f(s.eps,s.alp))
    print("dfde   ", hm.dfde(s.eps,s.alp))
    print("dfda   ", hm.dfda(s.eps,s.alp))
    print("d2fdede", hm.d2fdede(s.eps,s.alp))
    print("d2fdeda", hm.d2fdeda(s.eps,s.alp))
    print("d2fdade", hm.d2fdade(s.eps,s.alp))
    print("d2fdada", hm.d2fdada(s.eps,s.alp))
    print("g      ", hm.g(s.sig,s.alp))
    print("dgds   ", hm.dgds(s.sig,s.alp))
    print("dgda   ", hm.dgda(s.sig,s.alp))
    print("d2gdsds", hm.d2gdsds(s.sig,s.alp))
    print("d2gdsda", hm.d2gdsda(s.sig,s.alp))
    print("d2gdads", hm.d2gdads(s.sig,s.alp))
    print("d2gdada", hm.d2gdada(s.sig,s.alp))
    print("y      ", hm.y(s.eps,s.sig,s.alp,s.chi))
    print("dyde   ", hm.dyde(s.eps,s.sig,s.alp,s.chi))
    print("dyds   ", hm.dyds(s.eps,s.sig,s.alp,s.chi))
    print("dyda   ", hm.dyda(s.eps,s.sig,s.alp,s.chi))
    print("dydc   ", hm.dydc(s.eps,s.sig,s.alp,s.chi))
    print("w      ", hm.w(s.eps,s.sig,s.alp,s.chi))
    print("dwdc   ", hm.dwdc(s.eps,s.sig,s.alp,s.chi))

def run_general_inc(Smat, Emat, Tdt, dt, nprint, nsub):
    if hj.voigt:
        Smat[:,3:6] = Utils.rooth*Smat[:,3:6]
        Emat[:,3:6] = Utils.root2*Emat[:,3:6]
    dTdt = Tdt / float(nprint*nsub)
    ddt  = dt  / float(nprint*nsub)
    hj.start_inc = True
    for iprint in range(nprint):
        for isub in range(nsub): 
            apply_general_inc(Smat, Emat, dTdt, ddt)
        record(s.eps, s.sig)
    qprint("Increment complete\n")

def apply_large_strain_inc_f(Fnew, dt):
    inc_mess("apply_large_strain_inc_f")
    yo       = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_    = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_    = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_    = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2fdede_ = d2fdede(s.eps,s.alp)
    d2fdade_ = d2fdade(s.eps,s.alp)
    d2fdeda_ = d2fdeda(s.eps,s.alp)
    d2fdada_ = d2fdada(s.eps,s.alp)
    rho  = hm.rhoo / Utils.det(Fnew)
    if hl.large_strain_def == "Green":
        L = LL_G(Fnew)
        Hnew = Green(Fnew)
    elif hl.large_strain_def == "Hencky":
        L = LL_H(Fnew)
        Hnew = Hencky(Fnew)
    Hnew_m = Utils.t_to_m(Hnew)
    Hinc = Hnew_m - s.eps
    if yo >= 0.0:
        #dyde_minus = dyde_ + np.einsum(Ein.f,dyds_,d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
        dyde_minus = dyde_ + (dyds_ @ d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
        dyda_minus = dyda_ + np.einsum(Ein.g,dyds_,d2fdeda_) - np.einsum(Ein.i, rwtdydc_, d2fdada_)
        lam = (-hj.acc*yo[0] - np.einsum("Ni,i->",dyde_minus,Hinc)) / \
              (dt*np.einsum("Nij,Nij->",dyda_minus,rwtdydc_))
        if lam > 0.0: 
            s.alp += lam*rwtdydc_[0,:,:]*dt
    s.eps  = Hnew_m
    s.sig  =  dfde(s.eps, s.alp)
    s.chi  = -dfda(s.eps, s.alp)
    s.t   += dt
    hl.F   = Fnew
    hl.sig = Utils.t_to_m(rho*np.einsum("kl,klij->ij",Utils.m_to_t(s.sig), L))
def apply_large_strain_inc_g(Fnew, dt):
    inc_mess("apply_large_strain_inc_g")
    yo       = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_    = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_    = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_    = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2gdsds_ = d2gdsds(s.sig,s.alp)
    d2gdads_ = d2gdads(s.sig,s.alp)
    d2gdsda_ = d2gdsda(s.sig,s.alp)
    d2gdada_ = d2gdada(s.sig,s.alp)
    rho = hm.rhoo / Utils.det(Fnew)
    D   = -np.linalg.inv(d2gdsds_)
    if hl.large_strain_def == "Green":
        L = LL_G(Fnew)
        Hnew = Green(Fnew)
    elif hl.large_strain_def == "Hencky":
        L = LL_H(Fnew)
        Hnew = Hencky(Fnew)
    Hnew_m = Utils.t_to_m(Hnew)
    Hinc = Hnew_m - s.eps
    dalp = np.zeros([hj.n_int,hj.n_dim])
    if yo >= 0.0:
        temp = dyds_ - np.einsum(Ein.h,rwtdydc_,d2gdads_)
        #dyde_minus = dyde_ + np.einsum("Ni,ij->Nj",temp,D)
        dyde_minus = dyde_ + (temp @ D)
        dyda_minus = dyda_ + np.einsum("Ni,ij,jMk->NMk",temp,D,d2gdsda_) - np.einsum(Ein.i,rwtdydc_,d2gdada_)
        lam = (-hj.acc*yo[0] - np.einsum("Ni,i->",dyde_minus,Hinc)) / \
              (dt*np.einsum("Nij,Nij->",dyda_minus,rwtdydc_))
        if lam > 0.0:
            dalp = lam*rwtdydc_[0,:,:]*dt
    s.alp += dalp
    s.eps  = Hnew_m
    s.sig += np.einsum("ij,j->i",D,Hinc + np.einsum("iNj,Nj->i",d2gdsda_,dalp))
    s.chi  = -dgda(s.sig, s.alp)
    s.t   += dt
    hl.F   = Fnew
    hl.sig = Utils.t_to_m(rho*np.einsum("kl,klij->ij",Utils.m_to_t(s.sig), L))
def apply_strain_inc(deps, dt):
    if hj.rate:
        if hj.gform: 
            strain_inc_g_r(deps, dt)
        else: 
            strain_inc_f_r(deps, dt) #default is f-form for this case, even if not set explicitly
    else:
        if hj.gform: 
            strain_inc_g(deps, dt)
        else: 
            strain_inc_f(deps, dt) #default is f-form for this case, even if not set explicitly
def apply_stress_inc(dsig, dt):
    if hj.rate:
        if hj.fform: 
            stress_inc_f_r(dsig, dt)
        else: 
            stress_inc_g_r(dsig, dt) #default is g-form for this case, even if not set explicitly
    else:
        if hj.fform: 
            stress_inc_f(dsig, dt)
        else: 
            stress_inc_g(dsig, dt) #default is g-form for this case, even if not set explicitly
def apply_general_inc(Smat, Emat, dTdt, dt):
    if hj.fform:
        if hj.rate:
            general_inc_f_r(Smat, Emat, dTdt, dt)
        else:
            general_inc_f(Smat, Emat, dTdt, dt)
    elif hj.gform: 
        if hj.rate:
            general_inc_g_r(Smat, Emat, dTdt, dt)
        else:
            if RKDP.opt:
                general_inc_g_RKDP(Smat, Emat, dTdt, dt)
            else:
                general_inc_g(Smat, Emat, dTdt, dt)
    else: 
        error("Error in general_inc: f-form or g-form needs to be specified")

def update_f(dt, deps, dalp):
    s.t   += dt
    s.eps += deps
    s.alp += dalp
    sigold = deepcopy(s.sig)
    chiold = deepcopy(s.chi)
    s.sig  = sig_f(s.eps,s.alp)
    s.chi  = chi_f(s.eps,s.alp)
    dsig   = s.sig - sigold
    dchi   = s.chi - chiold
    if hasattr(hm,"update"): 
        hm.update(s.t,s.eps,s.sig,s.alp,s.chi, dt,deps,dsig,dalp,dchi)
def update_g(dt, dsig, dalp):
    s.t   += dt
    s.sig += dsig
    s.alp += dalp
    epsold = deepcopy(s.eps)
    chiold = deepcopy(s.chi)
    s.eps  = eps_g(s.sig,s.alp)
    s.chi  = chi_g(s.sig,s.alp)
    deps   = s.eps - epsold
    dchi   = s.chi - chiold
    if hasattr(hm,"update"): 
        hm.update(s.t,s.eps,s.sig,s.alp,s.chi, dt,deps,dsig,dalp,dchi)

def inc_mess(routine):
    if hj.start_inc: 
        qprint("Using "+routine+" for increment")
        hj.start_inc = False
    
def general_inc_f_r(Smat, Emat, dTdt, dt): # updated for weights
    inc_mess("general_inc_f_r")
    dwdc_ = dwdc(s.eps,s.sig,s.alp,s.chi)
    d2fdede_ = d2fdede(s.eps,s.alp)
    d2fdeda_ = d2fdeda(s.eps,s.alp)
    #P = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede_))
    P = np.linalg.inv(Emat + (Smat @ d2fdede_))
    dalp  = np.einsum(Ein.chi, hj.rwt, dwdc_)*dt
    #dalp  = (hj.rwt @ dwdc_)*dt
    #deps = np.einsum(Ein.b, P, (dTdt - np.einsum("ij,mjk,mk->i", Smat, d2fdeda_, dalp)))
    deps = P @ (dTdt - np.einsum("ij,mjk,mk->i", Smat, d2fdeda_, dalp))
    update_f(dt, deps, dalp)
def general_inc_g_r(Smat, Emat, dTdt, dt): # updated for weights
    inc_mess("general_inc_g_r")
    dwdc_ = dwdc(s.eps,s.sig,s.alp,s.chi)
    d2gdsds_ = d2gdsds(s.sig,s.alp)
    d2gdsda_ = d2gdsda(s.sig,s.alp)
    #Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds_))
    Q = np.linalg.inv(Smat - (Emat @ d2gdsds_))
    dalp  = np.einsum(Ein.chi, hj.rwt, dwdc_)*dt
    #dalp  = (hj.rwt @ dwdc_)*dt
    #dsig = np.einsum(Ein.b, Q, (dTdt + np.einsum("ij,mjk,mk->i", Emat, d2gdsda_, dalp)))
    dsig = Q @ (dTdt + np.einsum("ij,mjk,mk->i", Emat, d2gdsda_, dalp))
    update_g(dt, dsig, dalp)
    
def strain_inc_f_r(deps, dt): # updated for weights
    inc_mess("strain_inc_f_r")
    dwdc_ = dwdc(s.eps,s.sig,s.alp,s.chi)
    dalp  = np.einsum(Ein.chi, hj.rwt, dwdc_)*dt
    #dalp  = (hj.rwt @ dwdc_)*dt
    update_f(dt, deps, dalp)
def stress_inc_g_r(dsig, dt): # updated for weights
    inc_mess("stress_inc_g_r")
    dwdc_ = dwdc(s.eps,s.sig,s.alp,s.chi)
    dalp  = np.einsum(Ein.chi, hj.rwt, dwdc_)*dt
    #dalp  = (hj.rwt @ dwdc_)*dt
    update_g(dt, dsig, dalp)

def strain_inc_g_r(deps, dt): # updated for weights
    inc_mess("strain_inc_g_r")
    dwdc_    = dwdc(s.eps,s.sig,s.alp,s.chi)
    d2gdsds_ = d2gdsds(s.sig,s.alp)
    d2gdsda_ = d2gdsda(s.sig,s.alp)
    D = -np.linalg.inv(d2gdsds_)
    dalp = np.einsum(Ein.chi, hj.rwt, dwdc_)*dt
    #dalp = (hj.rwt @ dwdc_)*dt
    #dsig = np.einsum(Ein.b, D, (deps + np.einsum(Ein.c, d2gdsda_, dalp)))
    dsig = D @ (deps + np.einsum(Ein.c, d2gdsda_, dalp))
    update_g(dt, dsig, dalp)
def stress_inc_f_r(dsig, dt): # updated for weights
    inc_mess("stress_inc_f_r")
    dwdc_    = dwdc(s.eps,s.sig,s.alp,s.chi)
    d2fdede_ = d2fdede(s.eps,s.alp)
    d2fdeda_ = d2fdeda(s.eps,s.alp)
    C = np.linalg.inv(d2fdede_)
    dalp = np.einsum(Ein.chi, hj.rwt, dwdc_)*dt
    #dalp = (hj.rwt @ dwdc_)*dt
    #deps = np.einsum(Ein.b, C, (dsig - np.einsum(Ein.c, d2fdeda_, dalp)))
    deps = C @ (dsig - np.einsum(Ein.c, d2fdeda_, dalp))
    update_f(dt, deps, dalp)

def general_inc_f(Smat, Emat, dTdt, dt): # updated for weights
    inc_mess("general_inc_f")
    yo    = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_ = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_ = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_ = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2fdede_ = d2fdede(s.eps,s.alp)
    d2fdeda_ = d2fdeda(s.eps,s.alp)
    d2fdade_ = d2fdade(s.eps,s.alp)
    d2fdada_ = d2fdada(s.eps,s.alp)
    #dyde_minus = dyde_ + np.einsum(Ein.f,dyds_,d2fdede_) - np.einsum(Ein.h,rwtdydc_,d2fdade_)
    dyde_minus = dyde_ + (dyds_ @ d2fdede_) - np.einsum(Ein.h,rwtdydc_,d2fdade_)
    dyda_minus = dyda_ + np.einsum(Ein.g,dyds_,d2fdeda_) - np.einsum(Ein.i,rwtdydc_,d2fdada_)
    #P     = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede_))
    P     = np.linalg.inv(Emat + (Smat @ d2fdede_))
    #temp1 = np.einsum(Ein.f, dyde_minus, P)
    temp1 = dyde_minus @ P
    temp2 = np.einsum("pr,rl,lnk->pnk", temp1, Smat, d2fdeda_)
    Lmatp = np.einsum(Ein.j, (temp2 - dyda_minus), rwtdydc_)
    #Lrhsp = hj.acc*yo + np.einsum(Ein.e, temp1, dTdt)    
    Lrhsp = hj.acc*yo + (temp1 @ dTdt)    
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    temp3 = np.einsum("ij,jmk,mk->i", Smat, d2fdeda_, dalp)
    #deps  = np.einsum(Ein.b, P, (dTdt - temp3))
    deps  = P @ (dTdt - temp3)
    update_f(dt, deps, dalp)
def general_inc_g(Smat, Emat, dTdt, dt): # updated for weights
    inc_mess("general_inc_g")
    yo    = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_ = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_ = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_ = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2gdsds_ = d2gdsds(s.sig,s.alp)
    d2gdsda_ = d2gdsda(s.sig,s.alp)
    d2gdads_ = d2gdads(s.sig,s.alp)
    d2gdada_ = d2gdada(s.sig,s.alp)
    #dyds_minus = dyds_ - np.einsum(Ein.f,dyde_,d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
    dyds_minus = dyds_ - (dyde_ @ d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
    dyda_minus = dyda_ - np.einsum(Ein.g,dyde_,d2gdsda_) - np.einsum(Ein.i, rwtdydc_, d2gdada_)
    #Q     = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds_))
    Q     = np.linalg.inv(Smat - (Emat @ d2gdsds_))
    #temp1 = np.einsum(Ein.f, dyds_minus, Q)
    temp1 = dyds_minus @ Q
    temp2 = np.einsum("pr,rl,lnk->pnk",temp1, Emat, d2gdsda_)
    Lmatp = np.einsum(Ein.j, (-temp2 - dyda_minus), rwtdydc_)
    #Lrhsp = hj.acc*yo + np.einsum(Ein.e, temp1, dTdt)    
    Lrhsp = hj.acc*yo + (temp1 @ dTdt)    
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    temp3 = np.einsum("ij,jmk,mk->i", Emat, d2gdsda_, dalp)
    #dsig  = np.einsum(Ein.b, Q, (dTdt + temp3))
    dsig  =  Q @ (dTdt + temp3)
    update_g(dt, dsig, dalp)
def general_inc_g_RKDP(Smat, Emat, dTdt, dt): #experimental
    tf = RKDP.tfac
    te = RKDP.terr
    t5 = RKDP.t5th
    errtol = 0.01
    inc_mess("general_inc_g_RKDP")
    def dstep(siga, alpa, Smat, Emat, dTdt):
        epsa = eps_g(siga, alpa)
        chia = chi_g(siga, alpa)
        yo    = hm.y(epsa,siga,alpa,chia)
        dyde_ = dyde(epsa,siga,alpa,chia)
        dyds_ = dyds(epsa,siga,alpa,chia)
        dyda_ = dyda(epsa,siga,alpa,chia)
        #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(epsa,siga,alpa,chia))
        rwtdydc_ = weight_dydc(dydc(epsa,siga,alpa,chia))
        d2gdsds_ = d2gdsds(siga,alpa)
        d2gdsda_ = d2gdsda(siga,alpa)
        d2gdads_ = d2gdads(siga,alpa)
        d2gdada_ = d2gdada(siga,alpa)
        #dyds_minus = dyds_ - np.einsum(Ein.f,dyde_,d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
        dyds_minus = dyds_ - (dyde_ @ d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
        dyda_minus = dyda_ - np.einsum(Ein.g,dyde_,d2gdsda_) - np.einsum(Ein.i, rwtdydc_, d2gdada_)
        #Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds_))
        Q = np.linalg.inv(Smat - (Emat @ d2gdsds_))
        #temp1 = np.einsum(Ein.f, dyds_minus, Q)
        temp1 = dyds_minus @ Q
        temp2 = np.einsum("pr,rl,lnk->pnk",temp1, Emat, d2gdsda_)
        Lmatp = np.einsum(Ein.j, (-temp2 - dyda_minus), rwtdydc_)
        #Lrhsp = hj.acc*yo + np.einsum(Ein.e, temp1, dTdt)    
        Lrhsp = hj.acc*yo + (temp1 @ dTdt)    
        L = solve_L(yo, Lmatp, Lrhsp)
        dalp  = np.einsum(Ein.d, L, rwtdydc_)
        temp3 = np.einsum("ij,jmk,mk->i", Emat, d2gdsda_, dalp)
        #dsig  = np.einsum(Ein.b, Q, (dTdt + temp3))
        dsig  = Q @ (dTdt + temp3)
        if hasattr(hm,"update"): 
            print("RKDM routine cannot at present handle hm.update")
            error()
        return dsig, dalp
    err = 1.0
    substeps = 1
    sig0 = deepcopy(s.sig) # make copies in case this step is abandoned
    alp0 = deepcopy(s.alp)
    for i in range(10):
        s.sig = deepcopy(sig0)
        s.alp = deepcopy(alp0)
        dTdtsub = dTdt / float(substeps)
        sig_err_sum = 0.0
        sig_inc_sum = 0.0
        alp_err_sum = 0.0
        alp_inc_sum = 0.0
        for sub in range(substeps):
            sig1 = deepcopy(s.sig)
            alp1 = deepcopy(s.alp)
            dsig1, dalp1 = dstep(sig1, alp1, Smat, Emat, dTdtsub)          
            sig2 = sig1 + dsig1*tf[0,0]    
            alp2 = alp1 + dalp1*tf[0,0]    
            dsig2, dalp2 = dstep(sig2, alp2, Smat, Emat, dTdtsub)            
            sig3 = sig1 + dsig1*tf[1,0] + dsig2*tf[1,1]
            alp3 = alp1 + dalp1*tf[1,0] + dalp2*tf[1,1]
            dsig3, dalp3 = dstep(sig3, alp3, Smat, Emat, dTdtsub)            
            sig4 = sig1 + dsig1*tf[2,0] + dsig2*tf[2,1] + dsig3*tf[2,2]
            alp4 = alp1 + dalp1*tf[2,0] + dalp2*tf[2,1] + dalp3*tf[2,2]
            dsig4, dalp4 = dstep(sig4, alp4, Smat, Emat, dTdtsub)            
            sig5 = sig1 + dsig1*tf[3,0] + dsig2*tf[3,1] + dsig3*tf[3,2] + dsig4*tf[3,3]
            alp5 = alp1 + dalp1*tf[3,0] + dalp2*tf[3,1] + dalp3*tf[3,2] + dalp4*tf[3,3]
            dsig5, dalp5 = dstep(sig5, alp5, Smat, Emat, dTdtsub)            
            sig6 = sig1 + dsig1*tf[4,0] + dsig2*tf[4,1] + dsig3*tf[4,2] + dsig4*tf[4,3] + dsig5*tf[4,4]
            alp6 = alp1 + dalp1*tf[4,0] + dalp2*tf[4,1] + dalp3*tf[4,2] + dalp4*tf[4,3] + dalp5*tf[4,4]
            dsig6, dalp6 = dstep(sig6, alp6, Smat, Emat, dTdtsub)            
            sig_err = dsig1*te[0] + dsig2*te[1] + dsig3*te[2] + dsig4*te[3] + dsig5*te[4] + dsig6*te[5]    
            alp_err = dalp1*te[0] + dalp2*te[1] + dalp3*te[2] + dalp4*te[3] + dalp5*te[4] + dalp6*te[5]    
            sig_inc = dsig1*t5[0] + dsig2*t5[1] + dsig3*t5[2] + dsig4*t5[3] + dsig5*t5[4] + dsig6*t5[5]    
            alp_inc = dalp1*t5[0] + dalp2*t5[1] + dalp3*t5[2] + dalp4*t5[3] + dalp5*t5[4] + dalp6*t5[5]            
            sig_err_norm = np.sqrt(np.einsum("i,i->",   sig_err,sig_err)/float(hj.n_dim)) 
            alp_err_norm = np.sqrt(np.einsum("ij,ij->", alp_err,alp_err)/float(hj.n_dim*hj.n_int))    
            sig_inc_norm = np.sqrt(np.einsum("i,i->",   sig_inc,sig_inc)/float(hj.n_dim))    
            alp_inc_norm = np.sqrt(np.einsum("ij,ij->", alp_inc,alp_inc)/float(hj.n_dim*hj.n_int))
            sig_err_sum += sig_err_norm
            sig_inc_sum += sig_inc_norm
            alp_err_sum += alp_err_norm
            alp_inc_sum += alp_inc_norm
#            if sig_inc_norm == 0.0: sig_inc_norm = Utils.small
#            if alp_inc_norm == 0.0: alp_inc_norm = Utils.small
#            sig_err_rel = sig_err_norm / sig_inc_norm    
#            alp_err_rel = alp_err_norm / alp_inc_norm    
            print("{:10.6f}".format(sig_inc_norm), "{:10.6f}".format(sig_err_norm), 
                  "{:21.6f}".format(alp_inc_norm), "{:10.6f}".format(alp_err_norm))    
#            errstep = np.maximum(sig_err_rel, alp_err_rel)
#            err     = np.maximum(err, errstep)
            s.sig += sig_inc
            s.alp += alp_inc
        if sig_inc_sum == 0.0: sig_inc_sum = Utils.small
        if alp_inc_sum == 0.0: alp_inc_sum = Utils.small
        sig_err_rel = sig_err_sum / sig_inc_sum    
        alp_err_rel = alp_err_sum / alp_inc_sum    
        print("{:10.6f}".format(sig_inc_sum), "{:10.6f}".format(sig_err_sum), "{:10.6f}".format(sig_err_rel), 
              "{:10.6f}".format(alp_inc_sum), "{:10.6f}".format(alp_err_sum), "{:10.6f}".format(alp_err_rel))    
#        err = np.maximum(sig_err_rel, alp_err_rel)
        err = sig_err_rel
        if err < errtol:
            s.eps = eps_g(s.sig, s.alp)
            s.chi = chi_g(s.sig, s.alp)
            s.t  += dt
            return
        else:
#            substeps = int(1.25*(err/errtol)**0.2) + 1
            substeps = substeps*2
            print("Substeps:",substeps)
    print("Too many trials in general_inc_RKDP")
    error()
def strain_inc_f_spec(deps): # updated for weights
    acc = 0.5 # use a local acc factor
    yo = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_ = hm.dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_ = hm.dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_ = hm.dyda(s.eps,s.sig,s.alp,s.chi)
    rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    d2fdede_ = hm.d2fdede(s.eps,s.alp)
    d2fdade_ = hm.d2fdade(s.eps,s.alp)
    d2fdeda_ = hm.d2fdeda(s.eps,s.alp)
    d2fdada_ = hm.d2fdada(s.eps,s.alp)
    #dyde_minus = dyde_ + np.einsum(Ein.f,dyds_,d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
    dyde_minus = dyde_ + (dyds_ @ d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
    dyda_minus = dyda_ + np.einsum(Ein.g,dyds_,d2fdeda_) - np.einsum(Ein.i, rwtdydc_, d2fdada_)
    Lmatp = -np.einsum(Ein.j, dyda_minus, rwtdydc_)
    #Lrhsp = acc*yo + np.einsum(Ein.e, dyde_minus, deps)
    Lrhsp = acc*yo + (dyde_minus @ deps)
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    s.eps += deps
    s.alp += dalp
    s.sig = hm.dfde(s.eps,s.alp)
    s.chi = chi_f(s.eps,s.alp)
 
def strain_inc_f(deps, dt): # updated for weights
    inc_mess("strain_inc_f")
    yo = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_    = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_    = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_    = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2fdede_ = d2fdede(s.eps,s.alp)
    d2fdade_ = d2fdade(s.eps,s.alp)
    d2fdeda_ = d2fdeda(s.eps,s.alp)
    d2fdada_ = d2fdada(s.eps,s.alp)
    #dyde_minus = dyde_ + np.einsum(Ein.f,dyds_,d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
    dyde_minus = dyde_ + (dyds_ @ d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
    dyda_minus = dyda_ + np.einsum(Ein.g,dyds_,d2fdeda_) - np.einsum(Ein.i, rwtdydc_, d2fdada_)
    Lmatp = -np.einsum(Ein.j, dyda_minus, rwtdydc_)
    #Lrhsp = hj.acc*yo + np.einsum(Ein.e, dyde_minus, deps)
    Lrhsp = hj.acc*yo + (dyde_minus @ deps)
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    update_f(dt, deps, dalp)
def stress_inc_g(dsig, dt): # updated for weights
    inc_mess("stress_inc_g")
    yo       = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_    = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_    = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_    = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2gdsds_ = d2gdsds(s.sig,s.alp)
    d2gdads_ = d2gdads(s.sig,s.alp)
    d2gdsda_ = d2gdsda(s.sig,s.alp)
    d2gdada_ = d2gdada(s.sig,s.alp)
    dyds_minus = dyds_ - np.einsum(Ein.f,dyde_,d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
    #dyds_minus = dyds_ - (dyde_ @ d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
    dyda_minus = dyda_ - np.einsum(Ein.g,dyde_,d2gdsda_) - np.einsum(Ein.i, rwtdydc_, d2gdada_)
    Lmatp = -np.einsum(Ein.j, dyda_minus, rwtdydc_)
    #Lrhsp = hj.acc*yo + np.einsum(Ein.e, dyds_minus, dsig)
    Lrhsp = hj.acc*yo + (dyds_minus @ dsig)
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    update_g(dt, dsig, dalp)
def strain_inc_g(deps, dt): # updated for weights
    inc_mess("strain_inc_g")
    yo       = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_    = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_    = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_    = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2gdsds_ = d2gdsds(s.sig,s.alp)
    d2gdsda_ = d2gdsda(s.sig,s.alp)
    d2gdads_ = d2gdads(s.sig,s.alp)
    d2gdada_ = d2gdada(s.sig,s.alp)
    D = -np.linalg.inv(d2gdsds_)
    #dyds_minus = dyds_ - np.einsum(Ein.f,dyde_,d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
    dyds_minus = dyds_ - (dyde_ @ d2gdsds_) - np.einsum(Ein.h, rwtdydc_, d2gdads_)
    dyda_minus = dyda_ - np.einsum(Ein.g,dyde_,d2gdsda_) - np.einsum(Ein.i, rwtdydc_, d2gdada_)
    #temp  = np.einsum(Ein.f, dyds_minus, D)
    temp  = dyds_minus @ D
    Lmatp = np.einsum(Ein.j, (-dyda_minus - np.einsum(Ein.g, temp, d2gdsda_)), rwtdydc_)
    #Lrhsp = hj.acc*yo + np.einsum(Ein.e, temp, deps)
    Lrhsp = hj.acc*yo + (temp @ deps)
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    #dsig  = np.einsum(Ein.b, D, (deps + np.einsum(Ein.c, d2gdsda_, dalp)))
    dsig  = D @ (deps + np.einsum(Ein.c, d2gdsda_, dalp))
    update_g(dt, dsig, dalp)
def stress_inc_f(dsig, dt): # updated for weights
    inc_mess("stress_inc_f")
    yo       = hm.y(s.eps,s.sig,s.alp,s.chi)
    dyde_    = dyde(s.eps,s.sig,s.alp,s.chi)
    dyds_    = dyds(s.eps,s.sig,s.alp,s.chi)
    dyda_    = dyda(s.eps,s.sig,s.alp,s.chi)
    #rwtdydc_ = np.einsum(Ein.dydc, hj.rwt, dydc(s.eps,s.sig,s.alp,s.chi))
    rwtdydc_ = weight_dydc(dydc(s.eps,s.sig,s.alp,s.chi))
    d2fdede_ = d2fdede(s.eps,s.alp)
    d2fdeda_ = d2fdeda(s.eps,s.alp)
    d2fdade_ = d2fdade(s.eps,s.alp)
    d2fdada_ = d2fdada(s.eps,s.alp)
    C = np.linalg.inv(d2fdede_)
    #dyde_minus = dyde_ + np.einsum(Ein.f,dyds_,d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
    dyde_minus = dyde_ + (dyds_ @ d2fdede_) - np.einsum(Ein.h, rwtdydc_, d2fdade_)
    dyda_minus = dyda_ + np.einsum(Ein.g,dyds_,d2fdeda_) - np.einsum(Ein.i, rwtdydc_, d2fdada_)
    #temp  = np.einsum(Ein.f, dyde_minus, C)
    temp  = dyde_minus @ C
    Lmatp = np.einsum(Ein.j, (np.einsum(Ein.g, temp, d2fdeda_) - dyda_minus), rwtdydc_)
    #Lrhsp = hj.acc*yo + np.einsum(Ein.e, temp, dsig)
    Lrhsp = hj.acc*yo + (temp @ dsig)
    L     = solve_L(yo, Lmatp, Lrhsp)
    dalp  = np.einsum(Ein.d, L, rwtdydc_)
    #deps  = np.einsum(Ein.b, C, (dsig - np.einsum(Ein.c, d2fdeda_, dalp)))
    deps  = C @ (dsig - np.einsum(Ein.c, d2fdeda_, dalp))
    update_f(dt, deps, dalp)

def record(eps, sig):
    result = True
    epso = eps_to_voigt(eps - hj.epsref) # convert if using Voigt option
    sigo = sig_to_voigt(sig)
    if hasattr(hm,"step_print"): hm.step_print(s.t,epso,sigo,s.alp,s.chi)
    if hj.recording:
        if np.isnan(epso).any() or np.isnan(sigo).any(): 
            result = False
        else:
            if hj.large:
                sig_cauchyo = sig_to_voigt(hl.sig)
                hj.rec[hj.curr_test].append(np.concatenate(([s.t], epso, sigo, sig_cauchyo,
                                                            s.alp.flatten(),
                                                            s.chi.flatten())))
            else:
                hj.rec[hj.curr_test].append(np.concatenate(([s.t], epso, sigo,
                                                            s.alp.flatten(),
                                                            s.chi.flatten())))
    return result
def delrec(): #delete one record
    del hj.rec[hj.curr_test][-1]
def recordt(eps, sig): #record test data
    hj.test_rec.append(np.concatenate(([s.t],eps,sig)))

def results_print(oname):
    if oname[-4:] != ".csv": oname = oname + ".csv"
    out_file = open(oname, 'w')
    names = names_()
    units = units_()
    print("")
    for recline in hj.rec:
        print(("{:>10} "+"{:>14} "*12).format(*names))
        print(("{:>10} "+"{:>14} "*12).format(*units))
        out_file.write(",".join(names)+"\n")
        out_file.write(",".join(units)+"\n")
        for item in recline:
            print(("{:10.4f} "+"{:14.8f} "*12).format(*item[:1+2*hj.n_dim]))
            out_file.write(",".join([str(num) for num in item])+"\n")
    out_file.close()

def results_csv(oname):
    if oname[-4:] != ".csv": oname = oname + ".csv"
    out_file = open(oname, 'w')
    names = names_()
    units = units_()
    for recline in hj.rec:
        out_file.write(",".join(names)+"\n")
        out_file.write(",".join(units)+"\n")
        for item in recline:
#            out_file.write(",".join([str(num) for num in item])+"\n")
            out_file.write(",".join([("{:12.6f}").format(num) for num in item[:1+2*hj.n_dim]])+"\n")
    out_file.close()

def plothigh(plt, x, y, col, highl, ax1, ax2):
    plt.plot(x, y, col, linewidth=1)
    for item in highl: 
        plt.plot(x[item[0]:item[1]], y[item[0]:item[1]], 'r')
    plt.plot(0.0, 0.0)            
    plt.set_xlabel(greek(ax1))
    plt.set_ylabel(greek(ax2))

def greek(name):
    gnam = name.replace("gamma", r"$\gamma$")
    gnam = gnam.replace("eps",   r"$\epsilon$")
    gnam = gnam.replace("theta", r"$\theta$")
    gnam = gnam.replace("sig",   r"$\sigma$")
    gnam = gnam.replace("tau",   r"$\tau$")
    gnam = gnam.replace("Kap",   "K")
    gnam = gnam.replace("1",     r"$_1$")
    gnam = gnam.replace("2",     r"$_2$")
    gnam = gnam.replace("3",     r"$_3$")
    gnam = gnam.replace("4",     r"$_4$")
    gnam = gnam.replace("_p",    r"$_p$")
    gnam = gnam.replace("_q",    r"$_q$")
    gnam = gnam.replace("_s",    r"$_s$")
    gnam = gnam.replace("_v",    r"$_v$")
    gnam = gnam.replace("sq",    r"$^2$")
    return gnam
    
def results_graph(pname, axes, xsize, ysize):
    if pname[-4:] != ".png": pname = pname + ".png"
    names = names_()
    plt.rcParams["figure.figsize"] = (xsize,ysize)
    fig, ax = plt.subplots()
    plt.title(hj.title)
    for i in range(len(hj.rec)):
        recl = hj.rec[i]
        for j in range(len(names)):
            if axes[0] == names[j]: 
                ix = j
                x = [item[j] for item in recl]
                if hj.test:
                    xt = [item[j] for item in hj.test_rec]
            if axes[1] == names[j]: 
                iy = j
                y = [item[j] for item in recl]
                if hj.test:
                    yt = [item[j] for item in hj.test_rec]
        plothigh(ax, x, y, hj.test_col[i], hj.high[i], nunit(ix), nunit(iy))
        if hj.test: 
            plothigh(ax, xt, yt, "g", hj.test_high, "", "")
    print("Graph of",axes[1],"v.",axes[0])
    plt.title(hj.title)
    if pname != "null.png": plt.savefig(pname)
    plt.show()

def names_(): # default variable names
    if hasattr(hm,"names"): 
        return hm.names
    elif hj.n_dim == 1: 
        return ["t","eps","sig"]   
    elif hj.n_dim == 2: 
        return ["t","eps1","eps2","sig1","sig2"]
    elif hj.n_dim == 3: 
        return ["t","eps1","eps2","eps3","sig1","sig2","sig3"]
    elif hj.n_dim == 6: 
        if hj.large:
            return ["t","H11",  "H22",  "H33",  "H23",  "H31",  "H12",
                        "Kap11","Kap22","Kap33","Kap23","Kap31","Kap12",
                        "sig11","sig22","sig33","tau23","tau31","tau12"]
        else:
            return ["t","eps11","eps22","eps33","gam23","gam31","gam12",
                        "sig11","sig22","sig33","tau23","tau31","tau12"]
def units_():
    if hasattr(hm,"units"): # default variable units
        return hm.units
    elif hj.n_dim == 1: 
        return ["s","-","Pa"]   
    elif hj.n_dim == 2: 
        return ["s","-","-","Pa","Pa"]   
    elif hj.n_dim == 3: 
        return ["s","-","-","-","Pa","Pa","Pa"]
    elif hj.n_dim == 6: 
        if hj.large:
            return ["s","-","-","-","-","-","-",
                        "msq/ssq","msq/ssq","msq/ssq","msq/ssq","msq/ssq","msq/ssq",
                        "Pa","Pa","Pa","Pa","Pa","Pa"]
        else:
            return ["s","-", "-", "-", "-", "-", "-",
                        "Pa","Pa","Pa","Pa","Pa","Pa"]
def nunit(i):
    return names_()[i] + " (" + units_()[i] + ")"

def results_plot(pname="null.png"): # plots for ndim=2
    if pname[-4:] != ".png": pname = pname + ".png"
    if hj.n_dim == 1:
        plt.rcParams["figure.figsize"] = (8.2,8.0)
        ax = plt.subplot(1, 1, 1)
        plt.title(hj.title)
        for i in range(len(hj.rec)):
            recl = hj.rec[i]
            e1 = [item[1] for item in recl]
            s1 = [item[2] for item in recl]
            plothigh(ax, e1, s1, hj.test_col[i], hj.high[i], nunit(1), nunit(2))
            if hj.test:
                et1 = [item[1] for item in hj.test_rec]
                st1 = [item[2] for item in hj.test_rec]
                plothigh(ax, et1, st1, 'g', hj.test_high, nunit(1), nunit(2))
    elif hj.n_dim == 2:
        plt.rcParams["figure.figsize"]=(8.2,8.0)
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
        ax1 = plt.subplot(2, 2, 1)
        plt.title(hj.title)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        for i in range(len(hj.rec)):
            e1 = [item[1] for item in hj.rec[i]]
            e2 = [item[2] for item in hj.rec[i]]
            s1 = [item[3] for item in hj.rec[i]]
            s2 = [item[4] for item in hj.rec[i]]
            plothigh(ax1, s1, s2, hj.test_col[i], hj.high[i], nunit(3), nunit(4))
            plothigh(ax2, e2, s2, hj.test_col[i], hj.high[i], nunit(2), nunit(4))
            plothigh(ax3, s1, e1, hj.test_col[i], hj.high[i], nunit(3), nunit(1))
            plothigh(ax4, e2, e1, hj.test_col[i], hj.high[i], nunit(2), nunit(1))
            if hj.test:
                et1 = [item[1] for item in hj.test_rec]
                et2 = [item[2] for item in hj.test_rec]
                st1 = [item[3] for item in hj.test_rec]
                st2 = [item[4] for item in hj.test_rec]
                plothigh(ax1, st1, st2, 'g', hj.test_high, nunit(3), nunit(4))
                plothigh(ax2, et2, st2, 'g', hj.test_high, nunit(2), nunit(4))
                plothigh(ax3, st1, et1, 'g', hj.test_high, nunit(3), nunit(1))
                plothigh(ax4, et2, et1, 'g', hj.test_high, nunit(2), nunit(1))
    if pname != "null.png": plt.savefig(pname)
    plt.show()

def results_plotCS(pname="null.png"): # special plots for critical state models, variant of results_plot
    if pname[-4:] != ".png": pname = pname + ".png"
    plt.rcParams["figure.figsize"] = (13.0,8.0)
    ax2 = plt.subplot(2, 3, 2)
    plt.title(hj.title)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    plt.subplots_adjust(wspace=0.45, hspace=0.3)
    ax4.set_xscale("log")
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
    for i in range(len(hj.rec)):
        e1 = [item[1] for item in hj.rec[i]]
        e2 = [item[2] for item in hj.rec[i]]
        s1 = [item[3] for item in hj.rec[i]]
        s2 = [item[4] for item in hj.rec[i]]
        plothigh(ax2, s1, s2, hj.test_col[i], hj.high[i], nunit(3), nunit(4))
        plothigh(ax3, e2, s2, hj.test_col[i], hj.high[i], nunit(2), nunit(4))
        plothigh(ax4, s1, e1, hj.test_col[i], hj.high[i], nunit(3)+" (log scale)", nunit(1))
        plothigh(ax5, s1, e1, hj.test_col[i], hj.high[i], nunit(3), nunit(1))
        plothigh(ax6, e2, e1, hj.test_col[i], hj.high[i], nunit(2), nunit(1))
        if hj.test:
            et1 = [item[1] for item in hj.test_rec]
            et2 = [item[2] for item in hj.test_rec]
            st1 = [item[3] for item in hj.test_rec]
            st2 = [item[4] for item in hj.test_rec]
            plothigh(ax2, st1, st2, 'g', hj.test_high, nunit(3), nunit(4))
            plothigh(ax3, et2, st2, 'g', hj.test_high, nunit(2), nunit(4))
            plothigh(ax4, st1, et1, 'g', hj.test_high, nunit(3)+" (log scale)", nunit(1))
            plothigh(ax5, st1, et1, 'g', hj.test_high, nunit(3), nunit(1))
            plothigh(ax6, et2, et1, 'g', hj.test_high, nunit(2), nunit(1))
        if hasattr(hm,"M"):
            maxp = np.max(s1)
            maxq = np.max(s2)
            maxp = np.min([maxp,maxq*1.01/hm.M])
            minp = np.max(s1)
            minq = -np.min(s2)
            minp = np.min([minp,minq*1.01/hm.M])
            cslp = np.array([minp, 0.0, maxp])
            cslq = np.array([-minp*hm.M, 0.0, maxp*hm.M])
            ax2.plot(cslp,cslq,"red")
    if pname != "null.png": plt.savefig(pname)
    plt.show()

def calc_init(): # initialise the calculation
    hj.n_int = max(1,hm.n_int)
    hj.n_y   = max(1,hm.n_y)
    print("Initialising calculation: n_dim =", hj.n_dim, 
                                  ", n_int =", hj.n_int, 
                                  ", n_y =",   hj.n_y)
    s.sig = np.zeros(hj.n_dim)
    s.eps = np.zeros(hj.n_dim)
    s.alp = np.zeros([hj.n_int, hj.n_dim])
    s.chi = np.zeros([hj.n_int, hj.n_dim])
    hl.sig = np.zeros(hj.n_dim)
    hj.epsref = np.zeros(hj.n_dim)
    if hasattr(hm, "rwt"):
        hj.rwt = hm.rwt
        print("Setting reciprocal weights:", hj.rwt)
    else:
        hj.rwt = np.ones(hj.n_int)
    hl.F = np.eye(3)
    return s.sig, s.eps, s.chi, s.alp

print("+-----------------------------------------------------------------------------+")
print("| HyperDrive: driving routine for hyperplasticity models                      |")
print("| (c) G.T. Houlsby, 2018-2024                                                 |")
print("|                                                                             |")
print("| \x1b[1;31mThis program is provided in good faith, but with no warranty of correctness\x1b[0m |")
print("+-----------------------------------------------------------------------------+")
print("Current directory: " + os.getcwd())