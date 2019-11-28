#Utility routines for HyperDrive and HyperCheck
import numpy as np
import copy
import importlib
from scipy import optimize

big = 1.0e10
small = 1.0e-6

def mac(x):
    if x > 0.0: return x
    else: return 0.0
def S(x):
    if x > 0.0: return 1.0 
    else: return -1.0
def floor(x):
    if x < small: return small
    else: return x
def Ineg(x):
    if x <= 0.0: return 0.0 
    else: return big
def Nneg(x):
    if x <= 0.0: return 0.0 
    else: return big*x

def princon(hm):
    print("Constants for model:",hm.file)
    print(hm.const)
    print("Derived values:")
    for i in range(len(hm.const)):
        print(hm.name_const[i]+" =",hm.const[i])

def w_rate_lin(y,mu): return (mac(y)**2)/(2.0*mu)
def w_rate_lind(y,mu): return mac(y)/mu
def w_rate_rpt(y,mu,r): return mu*(r**2)*(np.cosh(mac(y)/(mu*r)) - 1.0)
def w_rate_rptd(y,mu,r): return r*np.sinh(mac(y)/(mu*r))

def calcsum(vec):
    global eps, sig, chi, alp, nstep, nsub, deps, sigrec, const, hm, n_int, var
    sumerr = 0.0
    print(vec)
    hm.const = copy.deepcopy(const)
    for i in range(n_int):
       hm.const[2+2*i] = vec[i]
    hm.deriv()
    eps = 0.0
    sig = 0.0
    alp = np.zeros(n_int)
    chi = np.zeros(n_int)
    if "_h" in var:
        alp = np.zeros(n_int+1)
        chi = np.zeros(n_int+1)
    if "_cbh" in var:
        alp = np.zeros(n_int+1)
        chi = np.zeros(n_int+1)
    for step in range(nstep):
        for i in range(nsub):
            strain_inc_f(deps)
        error = sig - sigrec[step]
        sumerr += error**2
    return sumerr

def solve_L(yo, Lmatp, Lrhsp):
    global Lmat, Lrhs, L
    Lmat = np.eye(hm.n_y)                    #initialise matrix and RHS for elastic solution
    Lrhs = np.zeros(hm.n_y)
    for N in range(hm.n_y):                  #loop over yield surfaces
        if yo[N] > -0.00001:                 #if this surface yielding ...
            Lmat[N] = Lmatp[N]               #over-write line in matrix with plastic solution
            Lrhs[N] = Lrhsp[N]
    L = np.linalg.solve(Lmat, Lrhs)          #solve for plastic multipliers
    L = np.array([max(Lv, 0.0) for Lv in L]) #make plastic multipliers non-negative
    return L

def strain_inc_f(deps):
    global eps, sig, alp, chi
    acc = 0.5
    yo = hm.y_f(chi,eps,alp)
    dyda_minus = hm.dyda_f(chi,eps,alp) - np.einsum("Nm,mn->Nn", hm.dydc_f(chi,eps,alp), hm.d2fdada(eps,alp))
    dyde_minus = hm.dyde_f(chi,eps,alp) - np.einsum("Nm,m->N", hm.dydc_f(chi,eps,alp), hm.d2fdade(eps,alp))
    Lmatp = -np.einsum("Nn,Mn->NM", dyda_minus, hm.dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum("N,->N", dyde_minus, deps)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum("N,Nm->m",L,hm.dydc_f(chi,eps,alp))
    eps = eps + deps
    alp = alp + dalp
    sig = hm.dfde(eps,alp)
    chi = -hm.dfda(eps,alp)

def optim(base, variant):
    global sigrec, hm, nstep, nsub, n_int, const, deps, var
    global eps, sig, alp, chi   
    var = variant
    sigrec = np.zeros(nstep)
    print("calculate base curve from", base)
    hm = importlib.import_module(base)    
    hm.setvals()
    hm.const = copy.deepcopy(const[:2+2*n_int])
    hm.deriv()
    eps = 0.0
    sig = 0.0
    alp = np.zeros(n_int)
    chi = np.zeros(n_int)
    for step in range(nstep):
       for i in range(nsub): strain_inc_f(deps)
       sigrec[step] = sig       
    print("optimise", variant)
    hm = importlib.import_module(variant)
    hm.setvals()
    vec = np.zeros(n_int)
    for i in range(n_int): vec[i] = const[2*i+2]
    print(vec,calcsum(vec))
    #optimize.Bounds(0.0,np.inf)
    bnds = optimize.Bounds(0.0001,np.inf)
    resultop = optimize.minimize(calcsum, vec, method='L-BFGS-B', bounds=bnds)
    vec = resultop.x
    print(vec,calcsum(vec))
    for i in range(n_int): const[2+2*i] = resultop.x[i]
    return const

def derive_from_points(modeltype, epsin, sigin, Einf=0.0, epsmax=0.0, HARM_R=0.0):
    global nstep, nsub, n_int, const, deps
    eps = np.array(epsin)
    sig = np.array(sigin)
    le = len(eps)
    ls = len(sig)
    if ls != le:
        print("Unequal numbers of values")
        le = min(le, ls)
    n_int = le
    E = np.zeros(n_int+1)
    E[0] = sig[0] / eps[0]
    for i in range(1,n_int):
        E[i] = (sig[i]-sig[i-1]) / (eps[i]-eps[i-1])
    E[n_int] = Einf
    print("eps =",eps)
    print("sig =",sig)
    print("E   =",E)
    k = np.zeros(n_int)
    H = np.zeros(n_int)
    if "ser" in modeltype:
        print("Series parameters")
        E0 = E[0]
        for i in range(n_int):
            k[i] = sig[i]
            H[i] = E[i+1]*E[i]/(E[i] - E[i+1])
        const = [E0, n_int]
        for i in range(n_int):
            const.append(round(k[i],6))
            const.append(round(H[i],6))
        base = "h1epmk_ser"
    elif "par" in modeltype:
        print("Parallel parameters")
        for i in range(n_int):
            H[i] = E[i] - E[i+1]                
            k[i] = eps[i]*H[i]
        const = [Einf, n_int]
        for i in range(n_int):
            const.append(round(k[i],6))
            const.append(round(H[i],6))
        base = "h1epmk_par"
    elif "nest" in modeltype:
        print("Nested parameters")
        E0 = E[0]
        for i in range(n_int):
            k[i] = sig[i] - sig[i-1]
            H[i] = E[i+1]*E[i]/(E[i] - E[i+1])                
        k[0] = sig[0]
        const = [E0, n_int]
        for i in range(n_int):
            const.append(round(k[i],6))
            const.append(round(H[i],6))
        base = "h1epmk_nest"
    if "_b" in modeltype: #now optimise for bounding surface model
        print("Optimise parameters for _b option")
        nstep = 100
        nsub = 10
        if epsmax == 0.0:
            epsmax = 1.5*eps[n_int-1]
            print("setting epsmax =",epsmax)
        deps = epsmax / float(nstep*nsub)
        const = optim(base, modeltype)
        for i in range(2,2+2*n_int):
            const[i] = round(const[i],6)
    if "_h" in modeltype: #now optimise for HARM model
        print("Optimise parameters for _h option")
        nstep = 100
        nsub = 10
        if epsmax == 0.0:
            epsmax = 1.5*eps[n_int-1]
            print("setting epsmax =",epsmax)
        deps = epsmax / float(nstep*nsub)
        const.append(HARM_R)
        const = optim(base, modeltype)
        for i in range(2,2+2*n_int):
            const[i] = round(const[i],6)
    if "_cbh" in modeltype: #now optimise for bounding HARM model
        print("Optimise parameters for _cbh option")
        nstep = 100
        nsub = 10
        if epsmax == 0.0:
            epsmax = 1.5*eps[n_int-1]
            print("setting epsmax =",epsmax)
        deps = epsmax / float(nstep*nsub)
        const.append(HARM_R)
        const = optim(base, modeltype)
        for i in range(2,2+2*n_int):
            const[i] = round(const[i],6)
    return const

def numdiff_1(mode,ndim,fun,var,alp,vari):
    if mode == 0: num = 0.0
    else: num = np.zeros([ndim])
    for i in range(ndim):
        var1 = copy.deepcopy(var)
        var2 = copy.deepcopy(var)
        if mode == 0:
            var1 = var - vari  
            var2 = var + vari
        else:
            var1[i] = var[i] - vari  
            var2[i] = var[i] + vari    
        f1 = fun(var1,alp)
        f2 = fun(var2,alp)
        if mode == 0: num = (f2 - f1) / (2.0*vari)
        else: num[i] = (f2 - f1) / (2.0*vari)
    return num
        
def numdiff_2(mode,ndim,n_int,fun,var,alp,alpi):
    if mode == 0: num = np.zeros([n_int])
    else: num = np.zeros([n_int,ndim])
    for k in range(n_int):
        for i in range(ndim):
            alp1 = copy.deepcopy(alp)
            alp2 = copy.deepcopy(alp)
            if mode == 0:
                alp1[k] = alp[k] - alpi  
                alp2[k] = alp[k] + alpi
            else:
                alp1[k,i] = alp[k,i] - alpi  
                alp2[k,i] = alp[k,i] + alpi
            f1 = fun(var,alp1)
            f2 = fun(var,alp2)
            if mode == 0: num[k] = (f2 - f1) / (2.0*alpi)
            else: num[k,i] = (f2 - f1) / (2.0*alpi)
    return num

def numdiff_3(mode,ndim,n_int,n_y,fun,chi,var,alp,chii):
    if mode == 0: num = np.zeros([n_y,n_int])
    else: num = np.zeros([n_y,n_int,ndim])
    for k in range(n_int):
        for i in range(ndim):   
            chi1 = copy.deepcopy(chi)
            chi2 = copy.deepcopy(chi)
            if mode == 0:
                chi1[k] = chi[k] - chii  
                chi2[k] = chi[k] + chii
            else:
                chi1[k,i] = chi[k,i] - chii  
                chi2[k,i] = chi[k,i] + chii
            f1 = fun(chi1,var,alp)
            f2 = fun(chi2,var,alp)
            for j in range(n_y):
                if mode == 0: num[j,k] = (f2[j] - f1[j]) / (2.0*chii)
                else: num[j,k,i] = (f2[j] - f1[j]) / (2.0*chii)
    return num

def numdiff_4(mode,ndim,n_int,n_y,fun,chi,var,alp,vari):
    if mode == 0: num = np.zeros([n_y])
    else: num = np.zeros([n_y,ndim])
    for i in range(ndim):   
        var1 = copy.deepcopy(var)
        var2 = copy.deepcopy(var)
        if mode == 0:
            var1 = var - vari  
            var2 = var + vari
        else:
            var1[i] = var[i] - vari  
            var2[i] = var[i] + vari
        f1 = fun(chi,var1,alp)
        f2 = fun(chi,var2,alp)
        for j in range(n_y):
            if mode == 0: num[j] = (f2[j] - f1[j]) / (2.0*vari)
            else: num[j,i] = (f2[j] - f1[j]) / (2.0*vari)
    return num
            
def numdiff_5(mode,ndim,n_int,n_y,fun,chi,var,alp,alpi):
    if mode == 0: num = np.zeros([n_y,n_int])
    else: num = np.zeros([n_y,n_int,ndim])
    for k in range(n_int):
        for i in range(ndim):   
            alp1 = copy.deepcopy(alp)
            alp2 = copy.deepcopy(alp)
            if mode == 0:
                alp1[k] = alp[k] - alpi  
                alp2[k] = alp[k] + alpi
            else:
                alp1[k,i] = alp[k,i] - alpi  
                alp2[k,i] = alp[k,i] + alpi
            f1 = fun(chi,var,alp1)
            f2 = fun(chi,var,alp2)
            for j in range(n_y):
                if mode == 0: num[j,k] = (f2[j] - f1[j]) / (2.0*alpi)
                else: num[j,k,i] = (f2[j] - f1[j]) / (2.0*alpi)
    return num
                
def numdiff_6(mode,ndim,n_int,fun,chi,var,alp,chii):
    if mode == 0: num = np.zeros([n_int])
    else: num = np.zeros([n_int,ndim])
    for k in range(n_int):
        for i in range(ndim):   
            chi1 = copy.deepcopy(chi)
            chi2 = copy.deepcopy(chi)
            if mode == 0:
                chi1[k] = chi[k] - chii  
                chi2[k] = chi[k] + chii
            else:
                chi1[k,i] = chi[k,i] - chii  
                chi2[k,i] = chi[k,i] + chii
            f1 = fun(chi1,var,alp)
            f2 = fun(chi2,var,alp)
            if mode == 0: num[k] = (f2 - f1) / (2.0*chii)
            else: num[k,i] = (f2 - f1) / (2.0*chii)
    return num

def numdiff_6a(mode,ndim,n_int,fun,chi,var,alp,chii):
    if mode == 0: num = np.zeros([n_int])
    else: num = np.zeros([n_int,ndim])
    for k in range(n_int):
        for i in range(ndim):   
            chi1 = copy.deepcopy(chi)
            chi2 = copy.deepcopy(chi)
            if mode == 0:
                chi1[k] = chi[k] - chii  
                chi2[k] = chi[k] + chii
            else:
                chi1[k,i] = chi[k,i] - chii  
                chi2[k,i] = chi[k,i] + chii
            f1 = fun(chi1,var,alp)
            f0 = fun(chi,var,alp)
            f2 = fun(chi2,var,alp)
            if abs(f2-f0) > abs(f1-f0) == 0.0:
                if mode == 0: num[k] = (f2 - f0) / chii
                else: num[k,i] = (f2 - f0) / chii
            else:
                if mode == 0: num[k] = (f0 - f1) / chii
                else: num[k,i] = (f0 - f1) / chii
    return num

def numdiff2_1(mode,ndim,fun,var,alp,vari):
    if mode == 0: num = 0.0
    else: num = np.zeros([ndim,ndim])
    for i in range(ndim):
        for j in range(ndim):
            if i==j:
                var1 = copy.deepcopy(var)
                var3 = copy.deepcopy(var)
                if mode == 0:
                    var1 = var - vari  
                    var3 = var + vari
                else:
                    var1[i] = var[i] - vari  
                    var3[i] = var[i] + vari
                f1 = fun(var1,alp)
                f2 = fun(var,alp)
                f3 = fun(var3,alp)
                if mode == 0: num = (f1 - 2.0*f2 + f3) / (vari**2)
                else: num[i,i] = (f1 - 2.0*f2 + f3) / (vari**2)
            else:
                var1 = copy.deepcopy(var)
                var2 = copy.deepcopy(var)
                var3 = copy.deepcopy(var)
                var4 = copy.deepcopy(var)
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

def numdiff2_2(mode,ndim,n_int,fun,var,alp,vari,alpi):
    if mode == 0: num = np.zeros(n_int)
    else: num = np.zeros([n_int,ndim,ndim])
    for k in range(n_int):
        for i in range(ndim):
            for j in range(ndim):
                var1 = copy.deepcopy(var)
                var2 = copy.deepcopy(var)
                alp1 = copy.deepcopy(alp)
                alp2 = copy.deepcopy(alp)
                if mode == 0:
                    var1 = var - vari  
                    var2 = var + vari
                    alp1[k] = alp[k] - alpi  
                    alp2[k] = alp[k] + alpi
                else:
                    var1[i] = var[i] - vari  
                    var2[i] = var[i] + vari
                    alp1[k,j] = alp[k,j] - alpi  
                    alp2[k,j] = alp[k,j] + alpi
                f1 = fun(var1,alp1)
                f2 = fun(var2,alp1)
                f3 = fun(var1,alp2)
                f4 = fun(var2,alp2)
                if mode == 0: num[k] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
                else: num[k,i,j] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
    return num

def numdiff2_3(mode,ndim,n_int,fun,var,alp,vari,alpi):
    if mode == 0: num = np.zeros(n_int)
    else: num = np.zeros([n_int,ndim,ndim])
    for k in range(n_int):
        for i in range(ndim):
            for j in range(ndim):
                var1 = copy.deepcopy(var)
                var2 = copy.deepcopy(var)
                alp1 = copy.deepcopy(alp)
                alp2 = copy.deepcopy(alp)
                if mode == 0:
                    var1 = var - vari  
                    var2 = var + vari
                    alp1[k] = alp[k] - alpi  
                    alp2[k] = alp[k] + alpi
                else:
                    var1[i] = var[i] - vari  
                    var2[i] = var[i] + vari
                    alp1[k,j] = alp[k,j] - alpi  
                    alp2[k,j] = alp[k,j] + alpi
                f1 = fun(var1,alp1)
                f2 = fun(var2,alp1)
                f3 = fun(var1,alp2)
                f4 = fun(var2,alp2)
                if mode == 0: num[k] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
                else: num[k,j,i] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
    return num

def numdiff2_4(mode,ndim,n_int,fun,var,alp,alpi):
    if mode == 0: num = np.zeros([n_int,n_int])
    else: num = np.zeros([n_int,n_int,ndim,ndim])
    for k in range(n_int):
        for l in range(n_int):
            for i in range(ndim):
                for j in range(ndim):
                    if k==l and i==j:
                        alp1 = copy.deepcopy(alp)
                        alp3 = copy.deepcopy(alp)
                        if mode == 0:
                            alp1[k] = alp[k] - alpi  
                            alp3[k] = alp[k] + alpi
                        else:
                            alp1[k,i] = alp[k,i] - alpi  
                            alp3[k,i] = alp[k,i] + alpi
                        f1 = fun(var,alp1)
                        f2 = fun(var,alp)
                        f3 = fun(var,alp3)
                        if mode == 0: num[k,k] = (f1 - 2.0*f2 + f3) / (alpi**2)
                        else: num[k,k,i,i] = (f1 - 2.0*f2 + f3) / (alpi**2)
                    else:
                        alp1 = copy.deepcopy(alp)
                        alp2 = copy.deepcopy(alp)
                        alp3 = copy.deepcopy(alp)
                        alp4 = copy.deepcopy(alp)
                        if mode == 0:
                            alp1[k] = alp[k] - alpi  
                            alp1[l] = alp[l] - alpi  
                            alp2[k] = alp[k] - alpi  
                            alp2[l] = alp[l] + alpi  
                            alp3[k] = alp[k] + alpi  
                            alp3[l] = alp[l] - alpi  
                            alp4[k] = alp[k] + alpi  
                            alp4[l] = alp[l] + alpi
                        else:
                            alp1[k,i] = alp[k,i] - alpi  
                            alp1[l,j] = alp[l,j] - alpi  
                            alp2[k,i] = alp[k,i] - alpi  
                            alp2[l,j] = alp[l,j] + alpi  
                            alp3[k,i] = alp[k,i] + alpi  
                            alp3[l,j] = alp[l,j] - alpi  
                            alp4[k,i] = alp[k,i] + alpi  
                            alp4[l,j] = alp[l,j] + alpi  
                        f1 = fun(var,alp1)
                        f2 = fun(var,alp2)
                        f3 = fun(var,alp3)
                        f4 = fun(var,alp4)
                        if mode == 0: num[k,l] = (f1 - f2 - f3 + f4) / (4.0*(alpi**2))
                        else: num[k,l,i,j] = (f1 - f2 - f3 + f4) / (4.0*(alpi**2))
    return num