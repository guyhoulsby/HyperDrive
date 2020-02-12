import copy
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os.path
import re
import sys
import time
import HyperUtils as hu

def startup():
    global title, model
    global mode, ndim
    global fform, gform, rate, numerical
    global epsi, sigi, alpi, chii
    global recording, test, history, history_rec
    global acc, colour
    global t, high, test_high
    # set up default values
    title = ""
    model = "undefined"
    mode = 0
    ndim = 1
    fform = False
    gform = False
    rate = False
    numerical = False
    epsi = 0.0001
    sigi = 0.001
    alpi = 0.0001
    chii = 0.001
    recording = True
    test = False
    history = False
    history_rec = []
    acc = 0.5 # rate acceleration factor
    colour = "b"
    # initialise some key variables
    t = 0.0
    high = [[]]
    test_high = []
def process(source):
    global line # necessary to allow recording of history using histrec
    for line in source:
        text = line.rstrip("\n\r")
        textsplit = re.split(r'[ ,;]',text)
        keyword = textsplit[0]
        if len(keyword) == 0:
            continue
        if keyword[:1] == "#":
            continue
        elif keyword[:1] == "*" and hasattr(Commands, keyword[1:]):
            getattr(Commands, keyword[1:])(textsplit[1:])
            if keyword == "*end": break
        else:
            print("\x1b[0;31mWARNING - keyword not recognised:",keyword,"\x1b[0m")
def histrec():
    global history, history_rec, line
    if history: history_rec.append(line)
           
def readv(args):
    global mode, ndim
    if mode == 0: return float(args[1])
    elif mode == 1: return np.array([float(i) for i in args[1:ndim+1]])
def readvs(args):
    global mode, ndim
    if mode == 0: return float(args[0])
    elif mode == 1: return np.array([float(i) for i in args[:ndim]])
def error(text = "Unspecified error"):
    print(text)
    sys.exit()
    
def sig_f(eps, alp): return  dfde(eps, alp)
def chi_f(eps, alp): return -dfda(eps, alp)
def eps_g(sig, alp): return -dgds(sig, alp)
def chi_g(sig, alp): return -dgda(sig, alp)
def dfde(eps,alp):
    if numerical or not hasattr(hm,"dfde"): return hu.numdiff_1(mode,ndim,hm.f,eps,alp,epsi)
    else: return hm.dfde(eps,alp)
def dfda(eps,alp):
    if numerical or not hasattr(hm,"dfda"): return hu.numdiff_2(mode,ndim,n_int,hm.f,eps,alp,alpi)
    else: return hm.dfda(eps,alp)
def dgds(sig,alp):
    if numerical or not hasattr(hm,"dgds"): return hu.numdiff_1(mode,ndim,hm.g,sig,alp,sigi)
    else: return hm.dgds(sig,alp)
def dgda(eps,alp):
    if numerical or not hasattr(hm,"dgda"): return hu.numdiff_2(mode,ndim,n_int,hm.g,sig,alp,alpi)
    else: return hm.dgda(sig,alp)
def dydc_f(chi,eps,alp):
    if numerical or not hasattr(hm,"dydc_f"): return hu.numdiff_3(mode,ndim,n_int,n_y,hm.y_f,chi,eps,alp,chii)
    else: return hm.dydc_f(chi,eps,alp)
def dyde_f(chi,eps,alp):
    if numerical or not hasattr(hm,"dyde_f"): return hu.numdiff_4(mode,ndim,n_int,n_y,hm.y_f,chi,eps,alp,epsi)
    else: return hm.dyde_f(chi,eps,alp)
def dyda_f(chi,eps,alp):
    if numerical or not hasattr(hm,"dyda_f"): return hu.numdiff_5(mode,ndim,n_int,n_y,hm.y_f,chi,eps,alp,alpi)
    else: return hm.dyda_f(chi,eps,alp)
def dydc_g(chi,sig,alp):
    if numerical or not hasattr(hm,"dydc_g"): return hu.numdiff_3(mode,ndim,n_int,n_y,hm.y_g,chi,sig,alp,chii)
    else: return hm.dydc_g(chi,sig,alp)
def dyds_g(chi,sig,alp):
    if numerical or not hasattr(hm,"dyds_g"): return hu.numdiff_4(mode,ndim,n_int,n_y,hm.y_g,chi,sig,alp,sigi)
    else: return hm.dyds_g(chi,sig,alp)
def dyda_g(chi,sig,alp):
    if numerical or not hasattr(hm,"dyda_g"): return hu.numdiff_5(mode,ndim,n_int,n_y,hm.y_g,chi,sig,alp,alpi)
    else: return hm.dyda_g(chi,sig,alp)
def dwdc_f(chi,eps,alp):
    if numerical or not hasattr(hm,"dwdc_f"): return hu.numdiff_6(mode,ndim,n_int,hm.w_f,chi,eps,alp,chii)
    else: return hm.dwdc_f(chi,eps,alp)
def dwdc_g(chi,sig,alp):
    if numerical or not hasattr(hm,"dwdc_g"): return hu.numdiff_6(mode,ndim,n_int,hm.w_g,chi,sig,alp,chii)
    else: return hm.dwdc_g(chi,eps,alp)
def d2fdede(eps,alp):
    if numerical or not hasattr(hm,"d2fdede"): return hu.numdiff2_1(mode,ndim,hm.f,eps,alp,epsi)
    else: return hm.d2fdede(eps,alp)
def d2fdeda(eps,alp):
    if numerical or not hasattr(hm,"d2fdeda"): return hu.numdiff2_2(mode,ndim,n_int,hm.f,eps,alp,epsi,alpi)
    else: return hm.d2fdeda(eps,alp)
def d2fdade(eps,alp):
    if numerical or not hasattr(hm,"d2fdade"): return hu.numdiff2_3(mode,ndim,n_int,hm.f,eps,alp,epsi,alpi)
    else: return hm.d2fdade(eps,alp)
def d2fdada(eps,alp):
    if numerical or not hasattr(hm,"d2fdada"): return hu.numdiff2_4(mode,ndim,n_int,hm.f,sig,alp,alpi)
    else: return hm.d2fdada(eps,alp)
def d2gdsds(sig,alp):
    if numerical or not hasattr(hm,"d2gdsds"): return hu.numdiff2_1(mode,ndim,hm.g,sig,alp,sigi)
    else: return hm.d2gdsds(sig,alp)
def d2gdsda(sig,alp):
    if numerical or not hasattr(hm,"d2gdsda"): return hu.numdiff2_2(mode,ndim,n_int,hm.g,sig,alp,sigi,alpi)
    else: return hm.d2gdsda(sig,alp)
def d2gdads(sig,alp):
    if numerical or not hasattr(hm,"d2gdads"): return hu.numdiff2_3(mode,ndim,n_int,hm.g,sig,alp,sigi,alpi)
    else: return hm.d2gdads(sig,alp)
def d2gdada(sig,alp):
    if numerical or not hasattr(hm,"d2gdada"): return hu.numdiff2_4(mode,ndim,n_int,hm.g,sig,alp,alpi)
    else: return hm.d2gdada(sig,alp)

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

def printderivs():
    print("eps    ",eps)
    print("sig    ",sig)
    print("alp    ",alp)
    print("chi    ",chi)
    print("f      ",hm.f(eps,alp))
    print("dfde   ",hm.dfde(eps,alp))
    print("dfda   ",hm.dfda(eps,alp))
    print("d2fdede",hm.d2fdede(eps,alp))
    print("d2fdeda",hm.d2fdeda(eps,alp))
    print("d2fdade",hm.d2fdade(eps,alp))
    print("d2fdada",hm.d2fdada(eps,alp))
    print("g      ",hm.g(sig,alp))
    print("dgds   ",hm.dgds(sig,alp))
    print("dgda   ",hm.dgda(sig,alp))
    print("d2gdsds",hm.d2gdsds(sig,alp))
    print("d2gdsda",hm.d2gdsda(sig,alp))
    print("d2gdads",hm.d2gdads(sig,alp))
    print("d2gdada",hm.d2gdada(sig,alp))
    print("y_f    ",hm.y_f(chi,eps,alp))
    print("dydc_f ",hm.dydc_f(chi,eps,alp))
    print("dyde_f ",hm.dyde_f(chi,eps,alp))
    print("dyda_f ",hm.dyda_f(chi,eps,alp))
    print("y_g    ",hm.y_g(chi,sig,alp))
    print("dydc_g ",hm.dydc_g(chi,sig,alp))
    print("dyds_g ",hm.dyds_g(chi,sig,alp))
    print("dyda_g ",hm.dyda_g(chi,sig,alp))
    print("w_f    ",hm.w_f(chi,eps,alp))
    print("dwdc_f ",hm.dwdc_f(chi,eps,alp))
    print("w_g    ",hm.w_g(chi,sig,alp))
    print("dwdc_g ",hm.dwdc_g(chi,sig,alp))

def strain_inc(deps, dt):
    if rate:
        if gform: strain_inc_g_r(deps, dt)
        else: strain_inc_f_r(deps, dt) #default is f-form for this case, even if not set explicitly
    else:
        if gform: strain_inc_g(deps, dt)
        else: strain_inc_f(deps, dt) #default is f-form for this case, even if not set explicitly
def stress_inc(dsig, dt):
    if rate:
        if fform: stress_inc_f_r(dsig, dt)
        else: stress_inc_g_r(dsig, dt) #default is g-form for this case, even if not set explicitly
    else:
        if fform: stress_inc_f(dsig, dt)
        else: stress_inc_g(dsig, dt) #default is g-form for this case, even if not set explicitly
def general_inc(Smat, Emat, dTdt, dt):
    if rate:
        if fform: general_inc_f_r(Smat, Emat, dTdt, dt)
        elif gform: general_inc_g_r(Smat, Emat, dTdt, dt)
        else: error("Error in general_inc - f-form or g-form needs to be specified")
    else:
        if fform: general_inc_f(Smat, Emat, dTdt, dt)
        elif gform: general_inc_g(Smat, Emat, dTdt, dt)
        else: error("Error in general_inc - f-form or g-form needs to be specified")

def update_f(dt, deps, dalp):
    global t, eps, sig, alp, chi
    t = t + dt
    eps = eps + deps
    alp = alp + dalp
    sigold = copy.deepcopy(sig)
    chiold = copy.deepcopy(chi)
    sig = sig_f(eps,alp)
    chi = chi_f(eps,alp)
    dsig = sig - sigold
    dchi = chi - chiold
    if hasattr(hm,"update"): hm.update(t,eps,sig,alp,chi,dt,deps,dsig,dalp,dchi)
def update_g(dt, dsig, dalp):
    global t, eps, sig, alp, chi
    t = t + dt
    sig = sig + dsig
    alp = alp + dalp
    epsold = copy.deepcopy(eps)
    chiold = copy.deepcopy(chi)
    eps = eps_g(sig,alp)
    chi = chi_g(sig,alp)
    deps = eps - epsold
    dchi = chi - chiold
    if hasattr(hm,"update"): hm.update(t,eps,sig,alp,chi,dt,deps,dsig,dalp,dchi)

def general_inc_f_r(Smat, Emat, dTdt, dt):
    global eps, sig, alp, chi
    #global P, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    if iprint==0 and isub==0: print("Using general_inc_f_r for increment")
    P = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede(eps,alp)))
    dalp = dwdc_f(chi,eps,alp)*dt
    deps = np.einsum(ein_b, P, (dTdt - np.einsum("ij,mjk,mk->i", Smat, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)
def general_inc_g_r(Smat, Emat, dTdt, dt):
    global eps, sig, alp, chi
    #global Q, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    if iprint==0 and isub==0: print("Using general_inc_g_r for increment")    
    Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds(sig,alp)))
    dalp = dwdc_g(chi,sig,alp)*dt
    dsig = np.einsum(ein_b, Q, (dTdt + np.einsum("ij,mjk,mk->i", Emat, d2gdsda(sig,alp), dalp)))
    update_g(dt, dsig, dalp)
    
def strain_inc_f_r(deps, dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_f_r for strain increment")
    dalp = dwdc_f(chi,eps,alp)*dt
    update_f(dt, deps, dalp)
def stress_inc_g_r(dsig,dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using stress_inc_g_r for stress increment")
    dalp = dwdc_g(chi,sig,alp)*dt
    update_g(dt, dsig, dalp)

def strain_inc_g_r(deps, dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_g_r for strain increment")
    if mode == 0: D = -1.0 / d2gdsds(sig,alp)
    else: D = -np.linalg.inv(d2gdsds(sig,alp))
    dalp = dwdc_g(chi,sig,alp)*dt
    dsig = np.einsum(ein_b, D, (deps + np.einsum(ein_c, d2gdsda(sig,alp), dalp)))
    update_g(dt, dsig, dalp)
def stress_inc_f_r(dsig, dt):
    global eps, sig, alp, chi
    #global C, dyde_minus, dyda_minus, Lmatp, Lrhsp
    if iprint==0 and isub==0: print("Using stress_inc_f_r for stress increment")
    if mode == 0: C = 1.0/d2fdede(eps,alp)
    else: C = np.linalg.inv(d2fdede(eps,alp))
    dalp = dwdc_f(chi,eps,alp)*dt
    deps = np.einsum(ein_b, C, (dsig - np.einsum(ein_c, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)

def general_inc_f(Smat, Emat, dTdt, dt):
    global eps, sig, alp, chi
    #global P, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    if iprint==0 and isub==0: print("Using general_inc_f for increment")
    yo = hm.y_f(chi,eps,alp)
    P = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede(eps,alp)))
    dyde_minus = dyde_f(chi,eps,alp) - np.einsum(ein_h, dydc_f(chi,eps,alp), d2fdade(eps,alp))
    dyda_minus = dyda_f(chi,eps,alp) - np.einsum(ein_i, dydc_f(chi,eps,alp), d2fdada(eps,alp))
    temp = np.einsum(ein_f, dyde_minus, P)
    Lmatp = np.einsum(ein_j,
                      (np.einsum("pr,rl,nlk->pnk",temp,Smat,d2fdeda(eps,alp)) - dyda_minus),
                      dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, temp, dTdt)    
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d, L, dydc_f(chi,eps,alp))
    deps = np.einsum(ein_b, P, (dTdt - np.einsum("ij,mjk,mk->i", Smat, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)
def general_inc_g(Smat, Emat, dTdt, dt):
    global eps, sig, alp, chi
    #global Q, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    if iprint==0 and isub==0: print("Using general_inc_g for increment")    
    yo = hm.y_g(chi,sig,alp)
    Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds(sig,alp)))
    dyds_minus = dyds_g(chi,sig,alp) - np.einsum(ein_h, dydc_g(chi,sig,alp), d2gdads(sig,alp))
    dyda_minus = dyda_g(chi,sig,alp) - np.einsum(ein_i, dydc_g(chi,sig,alp), d2gdada(sig,alp))
    temp = np.einsum(ein_f, dyds_minus, Q)
    Lmatp = np.einsum(ein_j,
                      (-np.einsum("pr,rl,nlk->pnk",temp,Emat,d2gdsda(sig,alp)) - dyda_minus),
                      dydc_g(chi,sig,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, temp, dTdt)    
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d, L, dydc_g(chi,sig,alp))
    dsig = np.einsum(ein_b, Q, (dTdt + np.einsum("ij,mjk,mk->i", Emat, d2gdsda(sig,alp), dalp)))
    update_g(dsig, dalp)
 
def strain_inc_f(deps, dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_f for strain increment")
    yo = hm.y_f(chi,eps,alp)
    dyda_minus = dyda_f(chi,eps,alp) - np.einsum(ein_i, dydc_f(chi,eps,alp), d2fdada(eps,alp))
    dyde_minus = dyde_f(chi,eps,alp) - np.einsum(ein_h, dydc_f(chi,eps,alp), d2fdade(eps,alp))
    Lmatp = -np.einsum(ein_j, dyda_minus, dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, dyde_minus, deps)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d,L,dydc_f(chi,eps,alp))
    update_f(dt, deps, dalp)
def stress_inc_g(dsig, dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using stress_inc_g for stress increment")
    yo = hm.y_g(chi,sig,alp)
    dyda_minus = dyda_g(chi,sig,alp) - np.einsum(ein_i, dydc_g(chi,sig,alp), d2gdada(sig,alp))
    dyds_minus = dyds_g(chi,sig,alp) - np.einsum(ein_h, dydc_g(chi,sig,alp), d2gdads(sig,alp))
    Lmatp = -np.einsum(ein_j, dyda_minus, dydc_g(chi,sig,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, dyds_minus, dsig)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d,L,dydc_g(chi,sig,alp))
    update_g(dt, dsig, dalp)
    
def strain_inc_g(deps, dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_g for strain increment")
    yo = hm.y_g(chi,sig,alp)
    if mode == 0: D = -1.0 / d2gdsds(sig,alp)
    else: D = -np.linalg.inv(d2gdsds(sig,alp))
    dyds_minus = dyds_g(chi,sig,alp) - np.einsum(ein_h, dydc_g(chi,sig,alp), d2gdads(sig,alp))
    dyda_minus = dyda_g(chi,sig,alp) - np.einsum(ein_i, dydc_g(chi,sig,alp), d2gdada(sig,alp))
    temp = np.einsum(ein_f, dyds_minus, D)
    Lmatp = np.einsum(ein_j,
                      (-dyda_minus - np.einsum(ein_g, temp, d2gdsda(sig,alp))),
                      dydc_g(chi,sig,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, temp, deps)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d, L, dydc_g(chi,sig,alp))
    dsig = np.einsum(ein_b, D, (deps + np.einsum(ein_c, d2gdsda(sig,alp), dalp)))
    update_g(dt, dsig, dalp)
def stress_inc_f(dsig, dt):
    global eps, sig, alp, chi
    global iprint, isub
    if iprint==0 and isub==0: print("Using stress_inc_f for stress increment")
    yo = hm.y_f(chi,eps,alp)
    if mode == 0: C = 1.0/d2fdede(eps,alp)
    else: C = np.linalg.inv(d2fdede(eps,alp))
    dyde_minus = dyde_f(chi,eps,alp) - np.einsum(ein_h, dydc_f(chi,eps,alp), d2fdade(eps,alp))
    dyda_minus = dyda_f(chi,eps,alp) - np.einsum(ein_i, dydc_f(chi,eps,alp), d2fdada(eps,alp))
    temp = np.einsum(ein_f, dyde_minus, C)
    Lmatp = np.einsum(ein_j,
                      (np.einsum(ein_g, temp, d2fdeda(eps,alp)) - dyda_minus),
                      dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, temp, dsig)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d, L, dydc_f(chi,eps,alp))
    deps = np.einsum(ein_b, C, (dsig - np.einsum(ein_c, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)

def record(eps, sig):
    result = True
    if recording:
#        print(t,eps,sig)
        if mode == 0:
            if np.isnan(eps) or np.isnan(sig): result = False
            else: rec[curr_test].append(np.concatenate(([t],[eps],[sig],alp,chi)))
        else:
            if np.isnan(sum(eps)) or np.isnan(sum(sig)): result = False
            else: rec[curr_test].append(np.concatenate(([t],eps,sig,alp.flatten(),chi.flatten())))
    return result
def delrec():
    del rec[curr_test][-1]
    
def recordt(eps, sig): #record test data
    if mode == 0: test_rec.append(np.concatenate(([t],[eps],[sig])))
    else: test_rec.append(np.concatenate(([t],eps,sig)))

def results_print(oname):
    print("")
    if oname[-4:] != ".csv": oname = oname + ".csv"
    out_file = open(oname, 'w')
    names = names_()
    units = units_()
    for recline in rec:
        if mode == 0:
            print("{:>10} {:>14} {:>14}".format(*names))
            print("{:>10} {:>14} {:>14}".format(*units))
            out_file.write(",".join(names)+"\n")
            out_file.write(",".join(units)+"\n")
            for item in recline:
                print("{:10.4f} {:14.8f} {:14.8f}".format(*item[:3]))
                out_file.write(",".join([str(num) for num in item])+"\n")
        elif mode == 1:
            print("{:>10} {:>14} {:>14} {:>14} {:>14}".format(*names))
            print("{:>10} {:>14} {:>14} {:>14} {:>14}".format(*units))
            out_file.write(",".join(names)+"\n")
            out_file.write(",".join(units)+"\n")
            for item in recline:
                print("{:10.4f} {:14.8f} {:14.8f} {:14.8f} {:14.8f}".format(*item[:1+2*ndim]))
                out_file.write(",".join([str(num) for num in item])+"\n")
    out_file.close()

def plothigh(x,y,col,highl,ax1,ax2):
    plt.plot(x, y, col, linewidth=1)
    for item in highl: 
        plt.plot(x[item[0]:item[1]], y[item[0]:item[1]], 'r')
    plt.plot(0.0,0.0)            
    plt.xlabel(greek(ax1))
    plt.ylabel(greek(ax2))

def greek(name):
    gnam = name.replace("eps","$\epsilon$")
    gnam = gnam.replace("sig","$\sigma$")
    gnam = gnam.replace("1","$_1$")
    gnam = gnam.replace("2","$_2$")
    gnam = gnam.replace("3","$_3$")
    return gnam
    
def results_graph(pname, axes):
    global test_col, high
    #global x, y
    if pname[-4:] != ".png": pname = pname + ".png"
    names = names_()
    plt.rcParams["figure.figsize"]=(6,6)
    for i in range(len(rec)):
        recl = rec[i]
        for j in range(len(names)):
            if axes[0] == names[j]: 
                ix =j
                x = [item[j] for item in recl]
            if axes[1] == names[j]: 
                iy = j
                y = [item[j] for item in recl]
        plothigh(x, y, test_col[i], high[i], nu(ix), nu(iy))
    print("Graph of",axes[1],"v.",axes[0])
    plt.title(title)
    if pname != "null.png": plt.savefig(pname)
    plt.show()

def names_():
    if hasattr(hm,"names"): 
        return hm.names
    elif mode == 0: 
        return ["t","eps","sig"]
    else:
        if ndim == 1: return ["t","eps","sig"]   
        elif ndim == 2: return ["t","eps1","eps2","sig1","sig2"]   
        elif ndim == 3: return ["t","eps1","eps2","eps3","sig1","sig2","sig3"]
        
def units_():
    if hasattr(hm,"units"): 
        return hm.units
    elif mode == 0: 
        return ["t","-","Pa"]
    else: 
        if ndim == 1: return ["t","-","Pa"]   
        elif ndim == 2: return ["t","-","-","Pa","Pa"]   
        elif ndim == 3: return ["t","-","-","-","Pa","Pa","Pa"]
    
def nu(i):
    return names_()[i] + " (" + units_()[i] + ")"

def results_plot(pname):
    global test_col, high
    if pname[-4:] != ".png": pname = pname + ".png"
    if mode == 0:
        plt.rcParams["figure.figsize"]=(6,6)
        for i in range(len(rec)):
            recl = rec[i]
            e = [item[1] for item in recl]
            s = [item[2] for item in recl]
            if test:
                et = [item[1] for item in test_rec]
                st = [item[2] for item in test_rec]
            plothigh(e, s, test_col[i], high[i], nu(1), nu(2))
            if test: plothigh(et, st, 'g', test_high, "", "")
        plt.title(title)
        if pname != "null.png": plt.savefig(pname)
        plt.show()
    elif mode == 1:
        plt.rcParams["figure.figsize"]=(8.2,8)
        for i in range(len(rec)):
            recl = rec[i]
            e1 = [item[1] for item in recl]
            e2 = [item[2] for item in recl]
            s1 = [item[3] for item in recl]
            s2 = [item[4] for item in recl]
            if test:
                et1 = [item[1] for item in test_rec]
                et2 = [item[2] for item in test_rec]
                st1 = [item[3] for item in test_rec]
                st2 = [item[4] for item in test_rec]
            plt.subplots_adjust(wspace=0.5,hspace=0.3)
            plt.subplot(2,2,1)
            plt.title(title)
            plothigh(s1, s2, test_col[i], high[i], nu(3), nu(4))
            if test: plothigh(st1, st2, 'g', test_high, nu(3), nu(4))
            plt.subplot(2,2,2)
            plothigh(e1, e2, test_col[i], high[i], nu(1), nu(2))
            if test: plothigh(et1, et2, 'g', test_high, nu(1), nu(2))
            plt.subplot(2,2,3)
            plothigh(e1, s1, test_col[i], high[i], nu(1), nu(3))
            if test: plothigh(et1, st1, 'g', test_high, nu(1), nu(3))
            plt.subplot(2,2,4)
            plothigh(e2, s2, test_col[i], high[i], nu(2), nu(4))
            if test: plothigh(et2, st2, 'g', test_high, nu(2), nu(4))
        if pname != "null.png": plt.savefig(pname)
        plt.show()

def modestart():
    global ein_a, ein_b, ein_c, ein_d, ein_e, ein_f, ein_g, ein_h, ein_i, ein_j
    global sig, sig_inc, sig_targ, sig_cyc, dsig
    global eps, eps_inc, eps_targ, eps_cyc, deps
    global chi, alp, dalp, yo
    global n_int, n_y

    #print(hm.n_int)
    n_int = max(1,hm.n_int)
    n_y = max(1,hm.n_y)
    if mode == 0:
        ein_a = ",->"       #sig sig
        ein_b = ",->"
        ein_c = "m,m->"
        ein_d = "N,Nm->m"   #L dydc
        ein_e = "N,->N"     #dyde deps
        ein_f = "N,->N"
        ein_g = "N,n->Nn"
        ein_h = "Nm,m->N"   #dydc d2gdads -> dyds
        ein_i = "Nm,mn->Nn" #dydc d2ydada -> dyda
        ein_j = "Nn,Mn->NM" #dyda dydc
        sig = 0.0
        eps = 0.0
        alp  = np.zeros(n_int)
        chi  = np.zeros(n_int)
        sig_inc = 0.0
        sig_targ = 0.0
        sig_cyc = 0.0
        dsig = 0.0
        eps_inc = 0.0
        eps_targ = 0.0
        eps_cyc = 0.0
        deps = 0.0
        dalp = np.zeros(n_int)
        yo = np.zeros(n_y)
    elif mode == 1:
        ein_a = "i,i->"
        ein_b = "ki,i->k"
        ein_c = "mij,mj->i"
        ein_d = "N,Nmi->mi"
        ein_e = "Ni,i->N"
        ein_f = "Nj,jk->Nk"
        ein_g = "Nk,nkl->Nnl"
        ein_h = "Nmi,mij->Nj"
        ein_i = "Nmi,mnij->Nnj"
        ein_j = "Nni,Mni->NM"
        sig = np.zeros(ndim)
        eps = np.zeros(ndim)
        alp  = np.zeros([n_int,ndim])
        chi  = np.zeros([n_int,ndim])
        sig_inc = np.zeros(ndim)
        sig_targ = np.zeros(ndim)
        sig_cyc = np.zeros(ndim)
        dsig = np.zeros(ndim)
        eps_inc = np.zeros(ndim)
        eps_targ = np.zeros(ndim)
        eps_cyc = np.zeros(ndim)
        deps = np.zeros(ndim)
        dalp = np.zeros(n_int)
        yo = np.zeros(n_y)
    elif mode == 2:
        ein_a = "ij,ij->"
        ein_b = "klij,ij->kl"
        ein_c = "mijkl,mkl->ij"
        ein_d = "N,Nmij->mij"
        ein_e = "Nij,ij->N"
        ein_f = "Nij,ijkl->Nkl"
        ein_g = "Nkl,nklij->Nnij"
        ein_h = "Nmij,mijkl->Nkl"
        ein_i = "Nmij,mnijkl->Nnkl"
        ein_j = "Nnij,Mnij->NM"
        sig = np.zeros([ndim,ndim])
        eps = np.zeros([ndim,ndim])
        alp  = np.zeros([n_int,ndim,ndim])
        chi  = np.zeros([n_int,ndim,ndim])
        sig_inc = np.zeros([ndim,ndim])
        sig_targ = np.zeros([ndim,ndim])
        sig_cyc = np.zeros([ndim,ndim])
        dsig = np.zeros([ndim,ndim])
        eps_inc = np.zeros([ndim,ndim])
        eps_targ = np.zeros([ndim,ndim])
        eps_cyc = np.zeros([ndim,ndim])
        deps = np.zeros([ndim,ndim])
        dalp = np.zeros([n_int,ndim,ndim])
        yo = np.zeros([n_y,ndim,ndim])
    else:
        error("Mode not recognised:" + mode)        

class Commands:
#    def process(self, source):
#        global line # necessary to allow recording of history using histrec
#        for line in source:
#            text = line.rstrip("\n\r")
#            textsplit = re.split(r'[ ,;]',text)
#            keyword = textsplit[0]
#            if len(keyword) == 0:
#                continue
#            if keyword[:1] == "#":
#                continue
#            elif keyword[:1] == "*" and hasattr(self, keyword[1:]):
#                getattr(self, keyword[1:])(textsplit[1:])
#                if keyword == "*end": break
#            else:
#                print("\x1b[0;31mWARNING - keyword not recognised:",keyword,"\x1b[0m")

    def title(args):
        global title
        title = " ".join(args)
        print("Title: ", title)
    def mode(args):
        global mode, ndim
        mode = int(args[0])
        if mode == 0: ndim = 1
        else: ndim = int(args[1])
        print("Mode:", mode, "ndim:", ndim)
    def model(args):
        global mode, ndim
        global hm
        model_temp = args[0]
        if os.path.isfile(model_temp + ".py"):
            model = model_temp + ""
            print("Importing hyperplasticity model: ", model)
            hm = importlib.import_module(model)
            hu.princon(hm)
            print("Description: ", hm.name)
            if hm.mode != mode:
                print("Model mode mismatch: hm.mode:",hm.mode,", mode:", mode)
                error()
            if hm.ndim != ndim:
                print("Model ndim mismatch: hm.ndim:",hm.ndim,", ndim:", ndim)
                error()
        else:
            error("Model not found:" + model_temp)
        for fun in ["f","g","y_f","y_g","w_f","w_g"]:
            if not hasattr(hm, fun): print(model+"."+fun+" not present")
        for fun in ["dfde","dfda","dgds","dgda",
                    "dydc_f","dydc_g","dyde_f","dyds_g","dyda_f","dyda_g",
                    "dwdc_f","dwdc_g",
                    "d2fdede","d2fdeda","d2fdade","d2fdada",
                    "d2gdsds","d2gdsda","d2gdads","d2gdada"]:
            if not hasattr(hm, fun): 
                print(model+"."+fun+" not present - will use numerical differentation for this function if required")

# commands for setting options
    def f_form(args=[]):
        global fform
        print("Setting f-form")
        fform = True
    def g_form(args=[]):
        global gform
        print("Setting g-form")
        gform = True
    def rate(args=[]):
        global rate, hm
        print("Setting rate dependent analysis")
        rate = True
        if len(args) > 0: hm.mu = float(args[0])
    def rateind(args=[]):
        global rate
        print("Setting rate independent analysis")
        rate = False
    def numerical(args=[]):
        global numerical
        numerical = True
        print("Setting numerical differentiation for all functions")
    def analytical(args=[]):
        global numerical 
        numerical = False
        print("Unsetting numerical differentiation")
    def acc(args):
        global acc
        acc = float(args[0])
        print("Acceleration factor:", acc)
    def colour(args):
        global colour
        colour = args[0]

# commands for setting constants
    def const(args):
        global model, hm
        if model != "undefined":
            hm.const = [float(tex) for tex in args]
            print("Constants:", hm.const)
            hm.deriv()
            hu.princon(hm)
        else:
            print("Cannot read constants: model undefined")
    def tweak(args):
        global model, hm
        if model != "undefined":
            ctweak = args[0]
            vtweak = float(args[1])
            for i in range(len(hm.const)):
                if ctweak == hm.name_const[i]: 
                    hm.const[i] = vtweak 
                    print("Tweaked constant value:",hm.name_const[i],"set to", hm.const[i])
            hm.deriv()
        else:
            print("Cannot tweak constant: model undefined")
    def const_from_points(args):
        global model, hm
        if model != "undefined":
            npt = int(args[0])
            epsd = np.zeros(npt)
            sigd = np.zeros(npt)
            for ipt in range(npt):
                epsd[ipt] = float(args[1+2*ipt])
                sigd[ipt] = float(args[2+2*ipt])
            Einf = float(args[1+2*npt])
            epsmax = float(args[2+2*npt])
            HARM_R = float(args[3+2*npt])
            hm.const = hu.derive_from_points(model,epsd,sigd,Einf,epsmax,HARM_R)
            print("Constants from points:", hm.const)
            hm.deriv()
            hu.princon(hm)
        else:
            print("Model undefined")
    def const_from_curve(args):
        global model, hm
        modtype = args[0]
        curve = args[1]
        npt = int(args[2])
        sigmax = float(args[3])
        HARM_R = float(args[4])
        param = np.zeros(3)
        param[0:3] = [float(item) for item in args[5:8]]
        epsd = np.zeros(npt)
        sigd = np.zeros(npt)
        print("Calculated points from curve:")
        for ipt in range(npt):
            sigd[ipt] = sigmax*float(ipt+1)/float(npt)
            if curve == "power":
                Ei = param[0]
                epsmax = param[1]
                power = param[2]
                epsd[ipt] = sigd[ipt]/Ei + (epsmax-sigmax/Ei)*(sigd[ipt]/sigmax)**power
            if curve == "jeanjean":                    
                Ei = param[0]
                epsmax = param[1]
                A = param[2]
                epsd[ipt] = sigd[ipt]/Ei + (epsmax-sigmax/Ei)*((np.atanh(np.tanh(A)*sigd[ipt]/sigmax)/A)**2)
            print(epsd[ipt],sigd[ipt])
            if curve == "PISA":
                Ei = param[0]
                epsmax = param[1]
                n = param[2]
                epspmax = epsmax - sigmax/Ei
                A = n*sigmax/(Ei*epspmax)
                B = -2.0*A*sigd[ipt]/sigmax + (1.0-n)*((1.0 + sigmax/(Ei*epspmax))**2)*(sigd[ipt]/sigmax - 1.0)
                C = A*(sigd[ipt]/sigmax)**2
                D = np.max(B**2-4.0*A*C, 0.0)
                epsd[ipt] = sigd[ipt]/Ei + 2.0*epspmax*C/(-B + np.sqrt(D))
        Einf = 0.5*(sigd[npt-1]-sigd[npt-2])/(epsd[npt-1]-epsd[npt-2])
        epsmax = epsd[npt-1]
        hm.const = hu.derive_from_points(modtype,epsd,sigd,Einf,epsmax,HARM_R)
        print("Constants from curve:", hm.const)
        hm.deriv()
        hu.princon(hm)
    def const_from_data(args): #not yet implemented
        global model, hm
        if model != "undefined":
            dataname = args[0]
            if dataname[-4:] != ".csv": dataname = dataname + ".csv"
            print("Reading data from", dataname)
            data_file = open(dataname,"r")
            data_text = data_file.readlines()
            data_file.close()
            for i in range(len(data_text)):
                data_split = re.split(r'[ ,;]',data_text[i])
#                ttest = float(data_split[0])
                epstest = float(data_split[1])
                sigtest = float(data_split[2])
            npt = int(args[1])
            maxsig = float(args[2])
            HARM_R = float(args[3])
            epsd = np.zeros(npt)
            sigd = np.zeros(npt)
            print("Calculating const from data")
            for ipt in range(npt):
                sigd[ipt] = maxsig*float(ipt+1)/float(npt)
                for i in range(len(data_text)-1):
                    if sigd[ipt] >= sigtest[i] and sigd[ipt] <= sigtest[i+1]:
                        epsd[ipt] = epstest[i] + (sigtest[i+1]-sigd[ipt]) * \
                                    (epstest[i+1]-epstest[i])/(sigtest[i+1]-sigtest[i])
                print(epsd[ipt],sigd[ipt])
            Einf = 0.5*(sigd[npt-1]-sigd[npt-2])/(epsd[npt-1]-epsd[npt-2])
            epsmax = epsd[npt-1]
            hm.const = hu.derive_from_points(model,epsd,sigd,Einf,epsmax,HARM_R)
            print("Constants from data:", hm.const)
            hm.deriv()
            hu.princon(hm)
        else:
            print("Model undefined")

# commands for starting and stopping processing
    def start(args=[]):
        global hm
        global n_stage, n_cyc, n_print
        global rec, curr_test, test_rec, test_col, colour
        global start, last
        modestart()
        if hasattr(hm, "setalp"): hm.setalp(alp)
        n_stage = 0
        n_cyc = 0
        n_print = 0
        rec=[[]]
        curr_test = 0
        test_rec=[]
        test_col=[]
        record(eps, sig)
        test_col.append(colour)
        start = time.process_time()
        last = start + 0.0
    def restart(args=[]):
        global t, rec, high, curr_test, hm
        global n_stage, n_cyc, n_print
        global eps, sig
        global test_col, colour
        global start, last
        t = 0.0
        rec.append([])
        high.append([])
        curr_test += 1
        modestart()
        if hasattr(hm, "setalp"): hm.setalp(alp)
        n_stage = 0
        n_cyc = 0
        n_print = 0 
        record(eps, sig)
        test_col.append(colour)
        now = time.process_time()
        print("Time:",now-start,now-last)
        last = now
    def rec(args=[]):
        global recording
        print("Recording data")
        recording = True
    def stoprec(args=[]):
        global mode, recording
        if mode == 0: temp = np.nan
        else: temp = np.full(ndim, np.nan)
        record(temp, temp) # write a line of nan to split plots
        print("Stop recording data")
        recording = False
    def end(args=[]):
        global start, last
        now = time.process_time()
        print("Time:",now-start,now-last)
        print("End of test")

# commands for initialisation
    def init_stress(args):
        global sig, eps
        histrec()
        sig = readvs(args)
        eps = eps_g(sig,alp)
        print("Initial stress:", sig)
        print("Initial strain:", eps)
        delrec()
        record(eps, sig)
    def init_strain(args):
        global sig, eps
        histrec()
        eps = readvs(args)
        sig = sig_f(eps,alp)
        print("Initial strain:", eps)
        print("Initial stress:", sig)
        delrec()
        record(eps, sig)
        
# commands for applying stress and strain increments
    def general_inc(args):
        global sig, eps
        global iprint, isub
        histrec()
        Smat = np.reshape(np.array([float(i) for i in args[:ndim*ndim]]), (ndim, ndim))
        Emat = np.reshape(np.array([float(i) for i in args[ndim*ndim:2*ndim*ndim]]), (ndim, ndim))
        Tdt = np.array([float(i) for i in args[2*ndim*ndim:2*ndim*ndim+ndim]])
        dt = float(args[2*ndim*ndim+ndim])
        nprint = int(args[2*ndim*ndim+ndim+1])
        nsub = int(args[2*ndim*ndim+ndim+2])
        dTdt = Tdt / float(nprint*nsub)
        print("General control increment:")
        print("S   =", Smat)
        print("E   =", Emat)
        print("Tdt =", Tdt)
        for iprint in range(nprint):
            for isub in range(nsub): general_inc(Smat, Emat, dTdt, dt)
            record(eps, sig)
        print("Increment complete")
    def general_cyc(args):
        global sig, eps
        global iprint, isub
        histrec()
        Smat = np.reshape(np.array([float(i) for i in args[:ndim*ndim]]), (ndim, ndim))
        Emat = np.reshape(np.array([float(i) for i in args[ndim*ndim:2*ndim*ndim]]), (ndim, ndim))
        Tdt = np.array([float(i) for i in args[2*ndim*ndim:2*ndim*ndim+ndim]])
        tper = float(args[2*ndim*ndim+ndim])
        ctype = args[2*ndim*ndim+ndim+1]
        ncyc = int(args[2*ndim*ndim+ndim+2])
        nprint = int(args[2*ndim*ndim+ndim+3])
        if nprint%2 == 1: nprint += 1
        nsub = int(args[2*ndim*ndim+ndim+4])
        print("General control cycles:")
        print("S   =", Smat)
        print("E   =", Emat)
        print("Tdt =", Tdt)
        if ctype == "saw":
            dTdt = Tdt / float(nprint/2) / float(nsub)
            print("General cycles (saw): tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(int(nprint/2)):
                    for isub in range(nsub): general_inc(Smat, Emat, dTdt, tper)
                    record(eps, sig)
                dTdt = -dTdt
        if ctype == "sine":
            print("General cycles (sine): tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dTdt = Tdt * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): general_inc(Smat, Emat, dTdt, tper)
                    record(eps, sig)
        if ctype == "haversine":
            print("General cycles (haversine): tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dTdt = Tdt * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): general_inc(Smat, Emat, dTdt, tper)
                    record(eps, sig)
        print("Cycles complete")
    def strain_inc(args):
        global sig, eps
        global iprint, isub
        histrec()
        t_inc = float(args[0])
        eps_inc = readv(args)
        nprint = int(args[ndim+1])
        nsub = int(args[ndim+2])
        deps = eps_inc / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Strain increment:", eps_inc, "deps =", deps)
        for iprint in range(nprint):
            for isub in range(nsub): strain_inc(deps,dt)
            record(eps, sig)
        print("Increment complete")
    def strain_targ(args):
        global sig, eps
        global iprint, isub
        histrec()
        t_inc = float(args[0])
        eps_targ = readv(args)
        nprint = int(args[ndim+1])
        nsub = int(args[ndim+2])
        deps = (eps_targ - eps) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Strain target:", eps_targ, "deps =", deps)
        for iprint in range(nprint):
            for isub in range(nsub): strain_inc(deps,dt)
            record(eps, sig)
        print("Increment complete")
    def strain_cyc(args):
        global sig, eps
        global iprint, isub
        histrec()
        tper = float(args[0])
        eps_cyc = readv(args)
        ctype = args[ndim+1]
        ncyc = int(args[ndim+2])
        nprint = int(args[ndim+3])
        if nprint%2 == 1: nprint += 1
        nsub = int(args[ndim+4])
        dt = tper / float(nprint*nsub)
        if ctype == "saw":
            deps = eps_cyc / float(nprint/2) / float(nsub)
            print("Strain cycle (saw):", eps_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(int(nprint/2)):
                    for isub in range(nsub): strain_inc(deps,dt)
                    record(eps, sig)
                deps = -deps
        if ctype == "sine":
            print("Strain cycle (sine):", eps_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    deps = eps_cyc * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): strain_inc(deps,dt)
                    record(eps, sig)
        if ctype == "haversine":
            print("Strain cycle (haversine):", eps_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    deps = eps_cyc * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): strain_inc(deps,dt)
                    record(eps, sig)
        print("Cycles complete")
    def stress_inc(args):
        global sig, eps
        global iprint, isub
        histrec()
        sig_inc = readv(args)
        nprint = int(args[ndim+1])
        nsub = int(args[ndim+2])
        t_inc = float(args[0])
        dsig = sig_inc / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Stress increment:", sig_inc,"dsig =", dsig,dt)
        for iprint in range(nprint):
            for isub in range(nsub): stress_inc(dsig,dt)
            record(eps, sig)
        print("Increment complete")
    def stress_targ(args):
        global sig, eps
        global iprint, isub
        histrec()
        sig_targ = readv(args)
        nprint = int(args[ndim+1])
        nsub = int(args[ndim+2])
        t_inc = float(args[0])
        dsig = (sig_targ - sig) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Stress target:", sig_targ, "dsig =", dsig)
        for iprint in range(nprint):
            for isub in range(nsub): stress_inc(dsig,dt)
            record(eps, sig)
        print("Increment complete")
    def stress_cyc(args):
        global sig, eps
        global iprint, isub
        histrec()
        sig_cyc = readv(args)
        tper = float(args[0])
        ctype = args[ndim+1]
        ncyc = int(args[ndim+2])
        nprint = int(args[ndim+3])
        if nprint%2 == 1: nprint += 1
        nsub = int(args[ndim+4])
        dt = tper / float(nprint*nsub)
        if ctype == "saw":
            dsig = sig_cyc / float(nprint/2) / float(nsub)
            print("Stress cycle (saw):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(nprint/2):
                    for isub in range(nsub): stress_inc(dsig,dt)
                    record(eps, sig)
                dsig = -dsig
        if ctype == "sine":
            print("Stress cycle (sine):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dsig = sig_cyc * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): stress_inc(dsig,dt)
                    record(eps, sig)
        if ctype == "haversine":
            print("Stress cycle (haversine):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dsig = sig_cyc * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): stress_inc(dsig,dt)
                    record(eps, sig)
        print("Cycles complete")
    def test(args,ptype):
        global mode, ndim, t, test, test_rec
        global sig, eps
        global iprint, isub
        histrec()
        test = True
        testname = args[0]
        if testname[-4:] != ".csv": testname = testname + ".csv"
        print("Reading from", testname)
        test_file = open(testname,"r")
        test_text = test_file.readlines()
        test_file.close()
        nsub = int(args[1])
        test_rec = []
        for iprint in range(len(test_text)):
            test_split = re.split(r'[ ,;]',test_text[iprint])
            ttest = float(test_split[0])
            if mode==0:
                epstest = float(test_split[1])
                sigtest = float(test_split[2])
            else:                
                epstest = np.array([float(tex) for tex in test_split[1:ndim+1]])
                sigtest = np.array([float(tex) for tex in test_split[ndim+1:2*ndim+1]])
            if iprint == 0:
                eps = copy.deepcopy(epstest)
                sig = copy.deepcopy(sigtest)
                delrec()
                record(eps, sig)
                recordt(epstest, sigtest)
            else:
                dt = (ttest - t) / float(nsub)
                if ptype == "strain":
                    deps = (epstest - eps) / float(nsub)
                    for isub in range(nsub): strain_inc(deps,dt)
                elif ptype == "*stress":
                    dsig = (sigtest - sig) / float(nsub)
                    for isub in range(nsub): stress_inc(dsig,dt)                    
                recordt(epstest, sigtest)
                record(eps, sig)
    def strain_test(args):
        Commands.test(args, "strain")
    def stress_test(args):
        Commands.test(args, "stress")
    def path(args, ptype):
        global mode, ndim, t
        global eps, sig
        global iprint, isub
        histrec()
        testname = args[0]
        if testname[-4:] != ".csv": testname = testname + ".csv"
        print("Reading from", testname)
        test_file = open(testname,"r")
        test_text = test_file.readlines()
        test_file.close()
        nsub = int(args[1])
        for iprint in range(len(test_text)):
            test_split = re.split(r'[ ,;]',test_text[iprint])
            ttest = float(test_split[0])
            if mode == 0:
                val = float(test_split[1])
            else:                
                val = np.array([float(tex) for tex in test_split[1:ndim+1]])
            if iprint == 0:
                if ptype == "strain":
                    eps = copy.deepcopy(val)
                elif ptype == "stress":
                    sig = copy.deepcopy(val)
                delrec()
                record(eps, sig)
            else:
                dt = (ttest - t) / float(nsub)
                if ptype == "strain":
                    deps = (val - eps) / float(nsub)
                    for isub in range(nsub): strain_inc(deps,dt)
                elif ptype == "stress":
                    dsig = (val - sig) / float(nsub)
                    for isub in range(nsub): stress_inc(dsig,dt)
                record(eps, sig)
    def strain_path(args):
        Commands.path(args, "strain")
    def stress_path(args):
        Commands.path(args, "stress")
        
# commands for plotting and printing
    def printrec(args=[]):
        oname = "hout_" + model
        if len(args) > 0: oname = args[0]
        results_print(oname)
    def specialprint(args=[]):
        oname = "hout_" + model
        if len(args) > 0: oname = args[0]
        hm.results_print(oname)
    def plot(args=[]):
        pname = "hplot_" + model
        if len(args) > 0: pname = args[0]
        results_plot(pname)
    def graph(args):
        pname = "hplot_" + model
        axes = args[0:2]
        if len(args) > 2: pname = args[2]
        results_graph(pname, axes)
    def specialplot(args=[]):
        pname = "hplot_" + model
        if len(args) > 0: pname = args[0]
        hm.results_plot(pname, rec)
    def high(args=[]):
        global hstart
        histrec()
        hstart = len(rec[curr_test])-1
    def unhigh(args=[]):
        global hend, high
        histrec()
        hend = len(rec[curr_test])
        high[curr_test].append([hstart,hend])
    def pause(args=[]):
        temp = input("Paused...")

# commands for recording and using history
    def start_history(args=[]):
        global history, history_rec
        history = True
        history_rec = []
    def end_history(args=[]):
        global history
        history = False
    def run_history(args=[]):
        global history
        history = False
        if len(args) > 0: runs = int(args[0])
        else: runs = 1
        for irun in range(runs):
            process(history_rec)
