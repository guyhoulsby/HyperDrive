# HyperDrive
# (c) G.T. Houlsby, 2018-2020
#
# Routines to run the Hyperdrive system for implementing hyperplasticity
# models. Load this software then run "drive()" or "check()".
#

# set following line to auto = False if autograd not available
auto = True

if auto:
    import autograd as ag
    import autograd.numpy as np
else:
    import numpy as np
import copy
import importlib
import matplotlib.pyplot as plt
import os.path
import re
from scipy import optimize
import sys
import time

udef = "undefined"

#Some utility routines
class Utils:
    big = 1.0e12
    small = 1.0e-12

    def mac(x): # Macaulay bracket
        if x > 0.0: return x
        else: return 0.0
    def S(x): # modified Signum function
        if abs(x) < Utils.small: return x / Utils.small
        elif x > 0.0: return 1.0  
        else: return -1.0
    def non_zero(x): # return a finite small value if argument close to zero
        if np.abs(x) < Utils.small: return Utils.small
        else: return x
    def Ineg(x): # Indicator function for set of negative reals
        if x <= 0.0: return 0.0 
        else: return Utils.big * 0.5*(x**2)
    def Nneg(x): # Normal Cone for set of negative reals
        if x <= 0.0: return 0.0 
        else: return Utils.big * x
    def w_rate_lin(y, mu): 
        return (Utils.mac(y)**2) / (2.0*mu) # y must be canonical
    def w_rate_lind(y, mu): 
        return Utils.mac(y) / mu
    def w_rate_rpt(y, mu,r): 
        return mu*(r**2) * (np.cosh(Utils.mac(y) / (mu*r)) - 1.0)
    def w_rate_rptd(y, mu,r): 
        return r * np.sinh(Utils.mac(y) / (mu*r))
    # Utilities for mode 2
    delta = np.eye(3) # Kronecker delta, unit tensor
    def pijkl(t1,t2): return np.einsum("ij,kl->ijkl",t1,t2)
    def piljk(t1,t2): return np.einsum("ij,kl->iljk",t1,t2)
    def pikjl(t1,t2): return np.einsum("ij,kl->ikjl",t1,t2)
    def pijjk(t1,t2): return np.einsum("ij,jk->ik",t1,t2)
    def dev(t): return t - (Utils.tr1(t)/3.0)*Utils.delta
    def tr1(t): return np.einsum("ii->",t)           # trace
    def tr2(t): return np.einsum("ij,ji->",t,t)      # trace of square
    def tr3(t): return np.einsum("ij,jk,ki->",t,t,t) # trace of cube
    def i1(t): # 1st invariant
        return Utils.tr1(t) 
    def i2(t): # 2nd invariant
        return (Utils.tr2(t) - Utils.tr1(t)**2)/2.0 
    def i3(t): # 3rd invariant
        return (2.0*Utils.tr3(t) - 3.0*Utils.tr2(t)*Utils.tr1(t) + Utils.tr1(t)**3)/6.0 
    def j2(t): # 2nd invariant of deviator
        return (3.0*Utils.tr2(t) - Utils.tr1(t)**2)/6.0 
    def j3(t): # 3rd invariant of deviator
        return (9.0*Utils.tr3(t) - 9.0*Utils.tr2(t)*Utils.tr1(t) + 2.0*Utils.tr1(t)**3)/27.0 
    def det(t): # determinant
        return t[0,0]*(t[1,1]*t[2,2]-t[1,2]*t[2,1]) + \
                       t[0,1]*(t[1,2]*t[2,0]-t[1,0]*t[2,2]) + \
                       t[0,2]*(t[1,0]*t[2,1]-t[1,1]*t[2,0])
    def trace(t): # trace (alternative)
        return t[1] + t[2] + t[3]
#    def res33to9(t): return t.reshape((9))
#    def res9to33(t): return t.reshape((3,3))
#    def trace9(t): return t[0] + t[4] + t[8] 
#    def trace33(t): return np.einsum("ii->",t)
#    def iprod9(t1,t2): return Utils.res33to9(Utils.pijjk(Utils.res9to33(t1),Utils.res9to33(t2)))
    II    = pikjl(delta, delta)  # 4th order unit tensor
    IIb   = piljk(delta, delta)  # 4th order unit tensor
    IIbb  = pijkl(delta, delta)  # 4th order unit tensor
    IIsym = (II + IIb) / 2.0     # 4th order unit tensor
    PP    = II - (IIbb / 3.0)    # 4th order projection tensor
    PPsym = IIsym - (IIbb / 3.0) # 4th order projection tensor

def process(source): # processes the Hyperdrive commands
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

class Commands:
# each of these functions corresponds to a command in Hyperdrive
    def title(args): # read a job title
        global title
        title = " ".join(args)
        print("Title: ", title)
    def mode(args): # set mode (0 = scalar, 1 = vector, 3 = tensor)
        global mode, n_dim
        mode = int(args[0])
        if mode == 0: 
            n_dim = 1
        else: 
            n_dim = int(args[1])
        print("Mode:", mode, "n_dim:", n_dim)
    def model(args): # specify model
        global mode, n_dim
        global hm, model
        model_temp = args[0]
        if os.path.isfile(model_temp + ".py"):
            model = model_temp + ""
            print("Importing hyperplasticity model: ", model)
            hm = importlib.import_module(model)
            princon(hm)
            print("Description: ", hm.name)
            if hm.mode != mode:
                print("Model mode mismatch: hm.mode:",hm.mode,", mode:", mode)
                error()
            if mode == 0:
                n_dim = 1
            else:
                if hm.ndim != n_dim:
                    print("Model n_dim mismatch: hm.ndim:",hm.ndim,", n_dim:", n_dim)
                    error()
        else:
            error("Model not found:" + model_temp)
        for fun in ["f", "g", "y_f", "y_g", "w_f", "w_g"]:
            if not hasattr(hm, fun): print("Function",fun,"not present in",model)
        set_up_auto()
        set_up_num()
        choose_diffs()
        
# commands for setting options
    def prefs(args=[]): # set preferences for method of differentiation
        global pref
        print("Setting differential preferences:")
        pref[0:3] = args[0:3]
        for i in range(3):
            print("Preference", i+1, pref[i])
        choose_diffs()
    def f_form(args=[]): # use f-functions for preference
        global fform, gform
        print("Setting f-form")
        fform = True
        gform = False
    def g_form(args=[]): # use g-functions for preference
        global gform, fform
        print("Setting g-form")
        gform = True
        fform = False
    def rate(args=[]): # use rate-dependent algorithm
        global rate, hm
        print("Setting rate dependent analysis")
        rate = True
        if len(args) > 0: hm.mu = float(args[0])
    def rateind(args=[]): # use rate-independent algorithm
        global rate
        print("Setting rate independent analysis")
        rate = False
    def acc(args): # set acceleration facto for rate-independent algorithm
        global acc
        acc = float(args[0])
        print("Setting acceleration factor:", acc)
    def colour(args): # select plot colour
        global colour
        colour = args[0]
        print("Setting plot colour:", colour)

# commands for setting constants
    def const(args): # read model constants
        global model, hm
        if model != "undefined":
            hm.const = [float(tex) for tex in args]
            print("Constants:", hm.const)
            hm.deriv()
            princon(hm)
        else:
            print("Cannot read constants: model undefined")
    def tweak(args): # tweak a single model constant
        global model, hm
        if model != "undefined":
            const_tweak = args[0]
            val_tweak = float(args[1])
            for i in range(len(hm.const)):
                if const_tweak == hm.name_const[i]: 
                    hm.const[i] = val_tweak 
                    print("Tweaked constant: "+hm.name_const[i]+", value set to", hm.const[i])
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
            hm.const = derive_from_points(model,epsd,sigd,Einf,epsmax,HARM_R)
            print("Constants from points:", hm.const)
            hm.deriv()
            princon(hm)
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
            print(epsd[ipt],sigd[ipt])
        Einf = 0.5*(sigd[npt-1]-sigd[npt-2]) / (epsd[npt-1]-epsd[npt-2])
        epsmax = epsd[npt-1]
        hm.const = derive_from_points(modtype, epsd, sigd, Einf, epsmax, HARM_R)
        print("Constants from curve:", hm.const)
        hm.deriv()
        princon(hm)
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
            hm.const = derive_from_points(model,epsd,sigd,Einf,epsmax,HARM_R)
            print("Constants from data:", hm.const)
            hm.deriv()
            princon(hm)
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
        global hm
        global n_stage, n_cyc, n_print
        global rec, curr_test, test_col, colour, high
        global start, last
        global t
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
        if mode == 0: 
            temp = np.nan
        else: 
            temp = np.full(n_dim, np.nan)
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
        global sig, eps, chi
        histrec()
        sig = readvs(args)
        eps = eps_g(sig,alp)
        chi = chi_g(sig,alp)
        print("Initial stress:", sig)
        print("Initial strain:", eps)
        delrec()
        record(eps, sig)
    def init_strain(args):
        global sig, eps, chi
        histrec()
        eps = readvs(args)
        sig = sig_f(eps,alp)
        chi = chi_f(eps,alp)
        print("Initial strain:", eps)
        print("Initial stress:", sig)
        delrec()
        record(eps, sig)
    def save_state(args=[]):
        global eps, sig, alp, chi, save_eps, save_sig, save_alp, save_chi
        save_eps = copy.deepcopy(eps)
        save_sig = copy.deepcopy(sig)
        save_alp = copy.deepcopy(alp)
        save_chi = copy.deepcopy(chi)
        print("State saved")
    def restore_state(args=[]):
        global eps, sig, alp, chi, save_eps, save_sig, save_alp, save_chi
        eps = copy.deepcopy(save_eps)
        sig = copy.deepcopy(save_sig)
        alp = copy.deepcopy(save_alp)
        chi = copy.deepcopy(save_chi)
        delrec()
        record(eps, sig)
        print("State restored")
        
# commands for applying stress and strain increments
    def general_inc(args):
        global sig, eps
        global start_inc
        histrec()
        Smat = np.reshape(np.array([float(i) for i in args[:n_dim*n_dim]]), (n_dim, n_dim))
        Emat = np.reshape(np.array([float(i) for i in args[n_dim*n_dim:2*n_dim*n_dim]]), (n_dim, n_dim))
        Tdt = np.array([float(i) for i in args[2*n_dim*n_dim:2*n_dim*n_dim+n_dim]])
        dt     = float(args[2*n_dim*n_dim+n_dim])
        nprint =   int(args[2*n_dim*n_dim+n_dim+1])
        nsub   =   int(args[2*n_dim*n_dim+n_dim+2])
        dTdt = Tdt / float(nprint*nsub)
        print("General control increment:")
        print("S   =", Smat)
        print("E   =", Emat)
        print("Tdt =", Tdt)
        start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): 
                general_inc(Smat, Emat, dTdt, dt)
            record(eps, sig)
        print("Increment complete")
    def general_cyc(args):
        global sig, eps
        global start_inc
        histrec()
        Smat = np.reshape(np.array([float(i) for i in args[:n_dim*n_dim]]), (n_dim, n_dim))
        Emat = np.reshape(np.array([float(i) for i in args[n_dim*n_dim:2*n_dim*n_dim]]), (n_dim, n_dim))
        Tdt = np.array([float(i) for i in args[2*n_dim*n_dim:2*n_dim*n_dim+n_dim]])
        tper   = float(args[2*n_dim*n_dim+n_dim])
        ctype  =       args[2*n_dim*n_dim+n_dim+1]
        ncyc   =   int(args[2*n_dim*n_dim+n_dim+2])
        nprint =   int(args[2*n_dim*n_dim+n_dim+3])
        if nprint%2 == 1: nprint += 1
        nsub = int(args[2*n_dim*n_dim+n_dim+4])
        print("General control cycles:")
        print("S   =", Smat)
        print("E   =", Emat)
        print("Tdt =", Tdt)
        start_inc = True
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
        global start_inc
        histrec()
        t_inc = float(args[0])
        eps_val = readv(args)
        nprint = int(args[n_dim+1])
        nsub = int(args[n_dim+2])
        deps = eps_val / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Strain increment:", eps_val, "deps =", deps, "dt =", dt)
        start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): 
                strain_inc(deps, dt)
            record(eps, sig)
        print("Increment complete")
    def strain_targ(args):
        global sig, eps
        global start_inc
        histrec()
        t_inc = float(args[0])
        eps_val = readv(args)
        nprint = int(args[n_dim+1])
        nsub = int(args[n_dim+2])
        deps = (eps_val - eps) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Strain target:", eps_val, "deps =", deps, "dt =", dt)
        start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): 
                strain_inc(deps, dt)
            record(eps, sig)
        print("Increment complete")
    def strain_cyc(args):
        global sig, eps
        global start_inc
        histrec()
        tper = float(args[0])
        eps_cyc = readv(args)
        ctype = args[n_dim+1]
        ncyc = int(args[n_dim+2])
        nprint = int(args[n_dim+3])
        if nprint%2 == 1: nprint += 1
        nsub = int(args[n_dim+4])
        dt = tper / float(nprint*nsub)
        start_inc = True
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
        global start_inc
        histrec()
        sig_val = readv(args)
        nprint = int(args[n_dim+1])
        nsub = int(args[n_dim+2])
        t_inc = float(args[0])
        dsig = sig_val / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Stress increment:", sig_val,"dsig =", dsig, "dt =", dt)
        start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): 
                stress_inc(dsig, dt)
            record(eps, sig)
        print("Increment complete")
    def stress_targ(args):
        global sig, eps
        global start_inc
        histrec()
        sig_val = readv(args)
        nprint = int(args[n_dim+1])
        nsub = int(args[n_dim+2])
        t_inc = float(args[0])
        dsig = (sig_val - sig) / float(nprint*nsub)
        dt = t_inc / float(nprint*nsub)
        print("Stress target:", sig_val, "dsig =", dsig, "dt =", dt)
        start_inc = True
        for iprint in range(nprint):
            for isub in range(nsub): 
                stress_inc(dsig, dt)
            record(eps, sig)
        print("Increment complete")
    def stress_cyc(args):
        global sig, eps
        global start_inc
        histrec()
        sig_cyc = readv(args)
        tper = float(args[0])
        ctype = args[n_dim+1]
        ncyc = int(args[n_dim+2])
        nprint = int(args[n_dim+3])
        if nprint%2 == 1: nprint += 1
        nsub = int(args[n_dim+4])
        dt = tper / float(nprint*nsub)
        start_inc = True
        if ctype == "saw":
            dsig = sig_cyc / float(nprint/2) / float(nsub)
            print("Stress cycle (saw):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(2*ncyc):
                for iprint in range(nprint/2):
                    for isub in range(nsub): 
                        stress_inc(dsig,dt)
                    record(eps, sig)
                dsig = -dsig
        if ctype == "sine":
            print("Stress cycle (sine):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dsig = sig_cyc * (np.sin(th2) - np.sin(th1)) / float(nsub)
                    for isub in range(nsub): 
                        stress_inc(dsig,dt)
                    record(eps, sig)
        if ctype == "haversine":
            print("Stress cycle (haversine):", sig_cyc, "tper=", tper, "ncyc =", ncyc)
            for icyc in range(ncyc):
                for iprint in range(int(nprint)):
                    th1 = 2.0*np.pi*float(iprint)/float(nprint)
                    th2 = 2.0*np.pi*float(iprint+1)/float(nprint)
                    dsig = sig_cyc * (np.sin(th2/2.0)**2 - np.sin(th1/2.0)**2) / float(nsub)
                    for isub in range(nsub): 
                        stress_inc(dsig,dt)
                    record(eps, sig)
        print("Cycles complete")
    def test(args, ptype):
        global mode, n_dim, t, test, test_rec
        global sig, eps
        global start_inc
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
        start_inc = True
        for iprint in range(len(test_text)):
            test_split = re.split(r'[ ,;]',test_text[iprint])
            ttest = float(test_split[0])
            if mode==0:
                epstest = float(test_split[1])
                sigtest = float(test_split[2])
            else:                
                epstest = np.array([float(tex) for tex in test_split[1:n_dim+1]])
                sigtest = np.array([float(tex) for tex in test_split[n_dim+1:2*n_dim+1]])
            if start_inc:
                eps = copy.deepcopy(epstest)
                sig = copy.deepcopy(sigtest)
                delrec()
                record(eps, sig)
                recordt(epstest, sigtest)
                start_inc= False
            else:
                dt = (ttest - t) / float(nsub)
                if ptype == "strain":
                    deps = (epstest - eps) / float(nsub)
                    for isub in range(nsub): 
                        strain_inc(deps,dt)
                elif ptype == "stress":
                    dsig = (sigtest - sig) / float(nsub)
                    for isub in range(nsub): 
                        stress_inc(dsig,dt)                    
                recordt(epstest, sigtest)
                record(eps, sig)
    def strain_test(args):
        Commands.test(args, "strain")
    def stress_test(args):
        Commands.test(args, "stress")
    def path(args, ptype):
        global mode, n_dim, t
        global eps, sig
        global start_inc
        histrec()
        testname = args[0]
        if testname[-4:] != ".csv": testname = testname + ".csv"
        print("Reading from", testname)
        test_file = open(testname,"r")
        test_text = test_file.readlines()
        test_file.close()
        nsub = int(args[1])
        start_inc = True
        for iprint in range(len(test_text)):
            test_split = re.split(r'[ ,;]',test_text[iprint])
            ttest = float(test_split[0])
            if mode == 0:
                val = float(test_split[1])
            else:                
                val = np.array([float(tex) for tex in test_split[1:n_dim+1]])
            if start_inc:
                if ptype == "strain":
                    eps = copy.deepcopy(val)
                elif ptype == "stress":
                    sig = copy.deepcopy(val)
                delrec()
                record(eps, sig)
                start_inc= False
            else:
                dt = (ttest - t) / float(nsub)
                if ptype == "strain":
                    deps = (val - eps) / float(nsub)
                    for isub in range(nsub): 
                        strain_inc(deps,dt)
                elif ptype == "stress":
                    dsig = (val - sig) / float(nsub)
                    for isub in range(nsub): 
                        stress_inc(dsig,dt)
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
        if hasattr(hm,"specialprint"): hm.specialprint(oname, eps, sig, alp, chi)
    def printstate(args=[]):
        print("eps =", eps)
        print("sig =", sig)
        print("alp =", alp)
        print("chi =", chi)
    def plot(args=[]):
        pname = "hplot_" + model
        if len(args) > 0: pname = args[0]
        results_plot(pname)
    def plotCS(args=[]):
        pname = "hplot_" + model
        if len(args) > 0: pname = args[0]
        results_plotCS(pname)
    def graph(args):
        pname = "hplot_" + model
        axes = args[0:2]
        if len(args) > 2: pname = args[2]
        results_graph(pname, axes)
    def specialplot(args=[]):
        pname = "hplot_" + model
        if len(args) > 0: pname = args[0]
        if hasattr(hm,"specialplot"): hm.specialplot(pname, title, rec, eps, sig, alp, chi)
    def high(args=[]):
        global hstart
        print("Start highlighting plot")
        histrec()
        hstart = len(rec[curr_test])-1
    def unhigh(args=[]):
        global hend, high
        print("Stop highlighting plot")
        histrec()
        hend = len(rec[curr_test])
        high[curr_test].append([hstart,hend])
    def pause(args=[]):
        if len(args) > 0:
            message = " ".join(args)
            print(message)
        input("Processing paused: hit ENTER to continue...")

# commands for recording and running history
    def start_history(args=[]):
        global history, history_rec
        print("Start recording history")
        history = True
        history_rec = []
    def end_history(args=[]):
        global history
        print("Stop recording history")
        history = False
    def run_history(args=[]):
        global history
        history = False
        if len(args) > 0: runs = int(args[0])
        else: runs = 1
        for irun in range(runs):
            print("Running history, cycle",irun+1,"of",runs)
            process(history_rec)

def error(text = "Unspecified error"):
    print(text)
    sys.exit()
def undef(): return udef

def set_up_auto():
    global adfde, adfda, ad2fdede, ad2fdeda, ad2fdade, ad2fdada
    global adgds, adgda, ad2gdsds, ad2gdsda, ad2gdads, ad2gdada
    global adydc_f, adyde_f, adyda_f
    global adydc_g, adyds_g, adyda_g
    global adwdc_f
    global adwdc_g
    adfde =    udef
    adfda =    udef
    ad2fdede = udef
    ad2fdeda = udef
    ad2fdade = udef
    ad2fdada = udef
    adgds =    udef
    adgda =    udef
    ad2gdsds = udef
    ad2gdsda = udef
    ad2gdads = udef
    ad2gdada = udef
    adydc_f =  udef
    adyde_f =  udef
    adyda_f =  udef
    adydc_g =  udef
    adyds_g =  udef
    adyda_g =  udef
    adwdc_f =  udef
    adwdc_g =  udef
    if not auto:
        print("\nAutomatic differentials not available")
        return
    print("\nSetting up automatic differentials")
    if hasattr(hm,"f"):
        print("Setting up auto-differentials of f")
        print("  dfde...")
        adfde = ag.jacobian(hm.f,0)
        print("  dfda...")
        adfda = ag.jacobian(hm.f,1)
        print("  d2fdede...")
        ad2fdede = ag.jacobian(adfde,0)
        print("  d2fdeda...")
        ad2fdeda = ag.jacobian(adfde,1)
        print("  d2fdade...")
        ad2fdade = ag.jacobian(adfda,0)
        print("  d2fdada...")
        ad2fdada = ag.jacobian(adfda,1)
    else:
        print("f not specified in", hm.file)
    if hasattr(hm,"g"):
        print("Setting up auto-differentials of g")
        print("  dgds...")
        adgds = ag.jacobian(hm.g,0)
        print("  dgda...")
        adgda = ag.jacobian(hm.g,1)
        print("  d2gdsds...")
        ad2gdsds = ag.jacobian(adgds,0)
        print("  d2gdsda...")
        ad2gdsda = ag.jacobian(adgds,1)
        print("  d2gdads...")
        ad2gdads = ag.jacobian(adgda,0)
        print("  d2gdada...")
        ad2gdada = ag.jacobian(adgda,1)
    else:
        print("g not specified in", hm.file)
    
    if hasattr(hm,"y_f"):
        print("Setting up auto-differentials of y_f")
        print("  dydc_f...")
        adydc_f = ag.jacobian(hm.y_f,0)
        print("  dyde_f...")
        adyde_f = ag.jacobian(hm.y_f,1)
        print("  dyda_f...")
        adyda_f = ag.jacobian(hm.y_f,2)
    else:
        print("y_f not specified in", hm.file)
    
    if hasattr(hm,"y_g"):
        print("Setting up auto-differentials of y_g")
        print("  dydc_g...")
        adydc_g = ag.jacobian(hm.y_g,0)
        print("  dyds_g...")
        adyds_g = ag.jacobian(hm.y_g,1)
        print("  dyda_g...")
        adyda_g = ag.jacobian(hm.y_g,2)
    else:
        print("y_g not specified in", hm.file)
    
    if hasattr(hm,"w_f"):
        print("Setting up auto-differential of w_f")
        print("  dwdc_f...")
        adwdc_f = ag.jacobian(hm.w_f,0)
    else:
        print("w_f not specified in", hm.file)
    
    if hasattr(hm,"w_g"):
        print("Setting up auto-differential of w_g")
        print("  dwdc_g...")
        adwdc_g = ag.jacobian(hm.w_g,0)
    else:
        print("w_g not specified in", hm.file)

def set_up_num():
    global ndfde, ndfda, nd2fdede, nd2fdeda, nd2fdade, nd2fdada
    global ndgds, ndgda, nd2gdsds, nd2gdsda, nd2gdads, nd2gdada
    global ndydc_f, ndyde_f, ndyda_f
    global ndydc_g, ndyds_g, ndyda_g
    global ndwdc_f
    global ndwdc_g
    ndfde =    udef
    ndfda =    udef
    nd2fdede = udef
    nd2fdeda = udef
    nd2fdade = udef
    nd2fdada = udef
    ndgds =    udef
    ndgda =    udef
    nd2gdsds = udef
    nd2gdsda = udef
    nd2gdads = udef
    nd2gdada = udef
    ndydc_f =  udef
    ndyde_f =  udef
    ndyda_f =  udef
    ndydc_g =  udef
    ndyds_g =  udef
    ndyda_g =  udef
    ndwdc_f =  udef
    ndwdc_g =  udef
    print("\nSetting up numerical differentials")
    if hasattr(hm,"f"):
        print("Setting up numerical differentials of f")
        print("  dfde...")
        def ndfde(eps,alp): return numdiff_1(mode, n_dim, hm.f, eps, alp, epsi)
        print("  dfda...")
        def ndfda(eps,alp): return numdiff_2(mode, n_dim, n_int, hm.f, eps, alp, alpi)
        print("  d2fdede...")
        def nd2fdede(eps,alp): return numdiff2_1(mode, n_dim, hm.f, eps, alp, epsi)
        print("  d2fdeda...")
        def nd2fdeda(eps,alp): return numdiff2_2(mode, n_dim, n_int, hm.f, eps, alp, epsi, alpi)
        print("  d2fdade...")
        def nd2fdade(eps,alp): return numdiff2_3(mode, n_dim, n_int, hm.f, eps, alp, epsi, alpi)
        print("  d2fdada...")
        def nd2fdada(eps,alp): return numdiff2_4(mode, n_dim, n_int, hm.f, eps, alp, alpi)
    else:
        print("f not specified in", hm.file)
    if hasattr(hm,"g"):
        print("Setting up numerical differentials of g")
        print("  dgds...")
        def ndgds(sig,alp): return numdiff_1(mode, n_dim, hm.g, sig, alp, sigi)
        print("  dgda...")
        def ndgda(sig,alp): return numdiff_2(mode, n_dim, n_int, hm.g, sig, alp, alpi)
        print("  d2gdsds...")
        def nd2gdsds(sig,alp): return numdiff2_1(mode, n_dim, hm.g, sig, alp, sigi)
        print("  d2gdsda...")
        def nd2gdsda(sig,alp): return numdiff2_2(mode, n_dim, n_int, hm.g, sig, alp, sigi, alpi)
        print("  d2gdads...")
        def nd2gdads(sig,alp): return numdiff2_3(mode, n_dim, n_int, hm.g, sig, alp, sigi, alpi)
        print("  d2gdada...")
        def nd2gdada(sig,alp): return numdiff2_4(mode, n_dim, n_int, hm.g, sig, alp, alpi)
    else:
        print("g not specified in", hm.file)
    if hasattr(hm,"y_f"):
        print("Setting up numerical differentials of y_f")
        print("  dydc_f...")
        def ndydc_f(chi,eps,alp): return numdiff_3(mode, n_dim, n_int, n_y, hm.y_f, chi, eps, alp, chii)
        print("  dyde_f...")
        def ndyde_f(chi,eps,alp): return numdiff_4(mode, n_dim, n_int, n_y, hm.y_f, chi, eps, alp, epsi)
        print("  dyda_f...")
        def ndyda_f(chi,eps,alp): return numdiff_5(mode, n_dim, n_int, n_y, hm.y_f, chi, eps, alp, alpi)
    else:
        print("y_f not specified in", hm.file)
    if hasattr(hm,"y_g"):
        print("Setting up numerical differentials of y_g")
        print("  dydc_g...")
        def ndydc_g(chi,sig,alp): return numdiff_3(mode, n_dim, n_int, n_y, hm.y_g, chi, sig, alp, chii)
        print("  dyds_g...")
        def ndyds_g(chi,sig,alp): return numdiff_4(mode, n_dim, n_int, n_y, hm.y_g, chi, sig, alp, sigi)
        print("  dyda_g...")
        def ndyda_g(chi,sig,alp): return numdiff_5(mode, n_dim, n_int, n_y, hm.y_g, chi, sig, alp, alpi)
    else:
        print("y_g not specified in", hm.file)
    if hasattr(hm,"w_f"):
        print("Setting up numerical differential of w_f")
        print("  dwdc_f...")
        def ndwdc_f(chi,eps,alp): return numdiff_6(mode, n_dim, n_int, hm.w_f, chi, eps, alp, chii)
    else:
        print("w_f not specified in", hm.file)
    if hasattr(hm,"w_g"):
        print("Setting up numerical differential of w_g")
        print("  dwdc_g...")
        def ndwdc_g(chi,sig,alp): return numdiff_6(mode, n_dim, n_int, hm.w_g, chi, sig, alp, chii)
    else:
        print("w_g not specified in", hm.file)

def choose_diffs():
    global dfde, dfda, d2fdede, d2fdeda, d2fdade, d2fdada
    global dgds, dgda, d2gdsds, d2gdsda, d2gdads, d2gdada
    global dydc_f, dyde_f, dyda_f
    global dydc_g, dyds_g, dyda_g
    global dwdc_f
    global dwdc_g
    def choose(name, nd, ad):
        d = udef
        for i in range(3):
            if pref[i] == "analytical" and hasattr(hm,name): d = getattr(hm,name)
            if pref[i] == "automatic"  and ad != udef: d = ad 
            if pref[i] == "numerical"  and nd != udef: d = nd
            if d != udef:
                print(name+":", pref[i])
                return d
        d = undef
        print(name+": undefined - will not run if this is required")
        return d        
    print("\nChoosing preferred differential methods")
    dfde    = choose("dfde",    ndfde,    adfde)
    dfda    = choose("dfda",    ndfda,    adfda)
    d2fdede = choose("d2fdede", nd2fdede, ad2fdede)
    d2fdeda = choose("d2fdeda", nd2fdeda, ad2fdeda)
    d2fdade = choose("d2fdade", nd2fdade, ad2fdade)
    d2fdada = choose("d2fdada", nd2fdada, ad2fdada)
    dgds    = choose("dgds",    ndgds,    adgds)
    dgda    = choose("dgda",    ndgda,    adgda)
    d2gdsds = choose("d2gdsds", nd2gdsds, ad2gdsds)
    d2gdsda = choose("d2gdsda", nd2gdsda, ad2gdsda)
    d2gdads = choose("d2gdads", nd2gdads, ad2gdads)
    d2gdada = choose("d2gdada", nd2gdada, ad2gdada)
    dydc_f  = choose("dydc_f",  ndydc_f,  adydc_f)
    dyde_f  = choose("dyde_f",  ndyde_f,  adyde_f)
    dyda_f  = choose("dyda_f",  ndyda_f,  adyda_f)
    dydc_g  = choose("dydc_g",  ndydc_g,  adydc_g)
    dyds_g  = choose("dyds_g",  ndyds_g,  adyds_g)
    dyda_g  = choose("dyda_g",  ndyda_g,  adyda_g)
    dwdc_f  = choose("dwdc_f",  ndwdc_f,  adwdc_f)
    dwdc_g  = choose("dwdc_g",  ndwdc_g,  adwdc_g)
    print("")

def sig_f(e, a): 
#    print("eps in sig_f",e)
    return  dfde(e, a)
def chi_f(e, a): return -dfda(e, a)
def eps_g(s, a): return -dgds(s, a)
def chi_g(s, a): return -dgda(s, a)

def testch(text1, val1, text2, val2):
    global npass, nfail
    print(text1, val1)
    print(text2, val2)
    if hasattr(val1, "shape") and hasattr(val2, "shape"):
        if val1.shape != val2.shape:
            print("\x1b[0;31mArrays different dimensions:",val1.shape,"and",val2.shape,"\x1b[0m")
            nfail += 1
            input("Hit ENTER to continue",)
            return
    if all(np.isclose(val1, val2, rtol=0.0001, atol=0.000001).reshape(-1)): 
        print("\x1b[1;32m***PASSED***\n\x1b[0m")
        npass += 1
    else:
        print("\x1b[1;31m",np.isclose(val1, val2, rtol=0.0001, atol=0.000001),"\x1b[0m")
        print("\x1b[1;31m***FAILED***\n\x1b[0m")
        nfail += 1
        input("Hit ENTER to continue",)

def testch2(ch, v1, v2):
    global nfail, npass, nmiss
    
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
            nmiss += 1
            return
    if not v2_iterable:
        if v2 == "undefined":
            print(ch,"second variable undefined, comparison not possible")
            nmiss += 1
            return
    
    if v1_iterable != v2_iterable:
        print(ch,"variables of different types, comparison not possible")
        nfail += 1
        return
    if v1_iterable and v2_iterable:
        if hasattr(v1,"shape") and hasattr(v2,"shape"):
            if v1.shape != v2.shape:
                print("\x1b[0;31mArrays different dimensions:",v1.shape,"and",v2.shape,"\x1b[0m")
                nfail += 1
                input("Hit ENTER to continue",)
                return
    if all(np.isclose(v1, v2, rtol=0.0001, atol=0.000001).reshape(-1)): 
        print(ch,"\x1b[1;32m***PASSED***\x1b[0m")
        npass += 1
    else:
        print("\x1b[1;31m",np.isclose(v1, v2, rtol=0.0001, atol=0.000001),"\x1b[0m")
        print(v1)
        print(v2)
        print("\x1b[1;31m***FAILED***\n\x1b[0m")
        nfail += 1
        input("Hit ENTER to continue",)
    
def testch1(text, hval, aval, nval):
    global npass, nfail
    print("\nAnalytical", text, hval)
    print("Automatic ",   text, aval)
    print("Numerical ",   text, nval)
    testch2("analytical v. automatic: ", hval, aval)
    testch2("automatic  v. numerical: ", aval, nval)
    testch2("numerical  v. analytical:", nval, hval)

def modestart_short():
    print("Modestart: n_int =",n_int,", n_dim =",n_dim)
    an_int = max(1,n_int)
    if mode == 0:
        sig = 0
        eps = 0
        alp  = np.zeros(an_int)
        chi  = np.zeros(an_int)
        ein1 = ",->"
        ein2 = ",->"
    elif mode == 1:
        sig = np.zeros(n_dim)
        eps = np.zeros(n_dim)
        alp  = np.zeros([an_int,n_dim])
        chi  = np.zeros([an_int,n_dim])
        ein1 = "ij,jl->il"
        ein2 = "i,i->"
    elif mode == 2:
        sig = np.zeros([n_dim,n_dim])
        eps = np.zeros([n_dim,n_dim])
        alp  = np.zeros([an_int,n_dim,n_dim])
        chi  = np.zeros([an_int,n_dim,n_dim])
        ein1 = "iIjJ,jJkK->iIkK"
        ein2 = "iI,iI->"
    else:
        print("Mode not recognised:", mode)
        error()
    return ein1, ein2, sig, eps, chi, alp

def check(arg="unknown"):
    print("")
    print("+---------------------------------------------------------+")
    print("| HyperCheck: checking routine for hyperplasticity models |")
    print("| (c) G.T. Houlsby, 2019-2020                             |")
    print("+---------------------------------------------------------+")
    global epsi, sigi, alpi, chii
    global n_int, n_dim, n_y, mode, hm, npass, nfail, nmiss
    global hm
    global pref
    
    print("Current directory: " + os.getcwd())
    if os.path.isfile("HyperDrive.cfg"):
        input_file = open("HyperDrive.cfg", 'r')
        last = input_file.readline().split(",")
        drive_file_name = last[0]
        check_file_name = last[1]
        input_file.close()
    else:
        drive_file_name = "hyper.dat"
        check_file_name = "unknown"
    if arg != "unknown": 
        check_file_name = arg
    input_found = False
    count = 0
    while not input_found:
        count += 1
        if count > 5: error("Too many tries")
        check_file_temp = input("Input model for checking [" + check_file_name + "]: ",)
        if len(check_file_temp) == 0:
            check_file_temp = check_file_name
        if os.path.isfile(check_file_temp + ".py"):
            check_file_name = check_file_temp
            print("Importing hyperplasticity model:", check_file_name)
            hm = importlib.import_module(check_file_name)
            #hm.setvals()
            print("Description:", hm.name)
            input_found = True
        else:
            print("File not found:", check_file_temp)
    last_file = open("HyperDrive.cfg", 'w')
    last_file.writelines([drive_file_name+","+check_file_name])
    last_file.close()

    npass = 0
    nfail = 0
    nmiss = 0
    model = hm.file
    mode  = hm.mode
    if mode == 0:
        n_dim = 1
    else:
        n_dim = hm.ndim
    n_int = hm.n_int
    n_y   = hm.n_y
    
    epsi = 0.000005
    sigi = 0.0001
    alpi = 0.000055
    chii = 0.0011
    
    pref = ["analytical", "automatic", "numerical"]
    set_up_auto()
    set_up_num()
    choose_diffs()
    
    ein1, ein2, sig, eps, chi, alp = modestart_short()
    
    if hasattr(hm, "check_eps"): 
        eps = hm.check_eps
    else:
        text = input("Input strain: ",)
        if mode == 0: 
            if len(text) == 0.0: text = "0.01"
            eps = float(text)
        else: 
            eps = np.array([float(item) for item in re.split(r'[ ,;]',text)])
    print("eps =", eps)
    
    if hasattr(hm, "check_sig"): 
        sig = hm.check_sig
    else:
        text = input("Input stress: ",)
        if mode == 0: 
            if len(text) == 0.0: text = "1.0"
            sig = float(text)
        else: 
            sig = np.array([float(item) for item in re.split(r'[ ,;]',text)])
    print("sig =", sig)
    
    if hasattr(hm, "check_alp"): 
        alp = hm.check_alp
    print("alp =", alp)
    
    if hasattr(hm, "check_chi"): 
        chi = hm.check_chi
    print("chi =", chi)
    
    princon(hm)   
    input("Hit ENTER to start checks",)
    
    if hasattr(hm, "f") and hasattr(hm, "g"):
        print("Checking consistency of f and g formulations ...\n")
        
        print("eps in check",eps)
        print("Checking inverse relations...")
        sigt = sig_f(eps, alp)
        epst = eps_g(sig, alp)
        print("Stress from strain:", sigt)
        print("Strain from stress:", epst)
        testch("Strain:             ", eps, "-> stress -> strain:", eps_g(sigt, alp))
        testch("Stress:             ", sig, "-> strain -> stress:", sig_f(epst, alp))
    
        print("Checking chi from different routes...")
        testch("from eps and f:", chi_f(eps, alp), "from g:        ", chi_g(sigt, alp))
        testch("from sig and g:", chi_g(sig, alp), "from f:        ", chi_f(epst, alp))
    
        print("Checking Legendre transform...")
        W = np.einsum(ein2, sigt, eps)
        testch("from eps: f + (-g) =", hm.f(eps,alp) - hm.g(sigt,alp), "sig.eps =          ", W)
        W = np.einsum(ein2, sig, epst)
        testch("from sig: f + (-g) =", hm.f(epst,alp) - hm.g(sig,alp), "sig.eps =          ", W)
    
        print("Checking elastic stiffness and compliance at specified strain...")
        if mode == 0:
            unit = 1.0
        else:
            unit = np.diag(np.ones(n_dim))
        D = d2fdede(eps,alp)
        C = -d2gdsds(sigt,alp)
        DC = np.einsum(ein1,D,C)
        print("Stiffness matrix D  =\n",D)
        print("Compliance matrix C =\n",C)
        testch("Product DC = \n",DC,"unit matrix =\n", unit)
        print("and at specified stress...")
        C = -d2gdsds(sig,alp)
        D = d2fdede(epst,alp)
        CD = np.einsum(ein1,C,D)
        print("Compliance matrix D =\n",C)
        print("Stiffness matrix C  =\n",D)
        testch("Product CD = \n",CD,"unit matrix =\n", unit)
    
    if hasattr(hm, "f"):
        print("Checking derivatives of f...")
        if hasattr(hm, "dfde"): 
            hval = hm.dfde(eps,alp)
        else:
            hval = undef()
        testch1("dfde", hval, adfde(eps,alp), ndfde(eps,alp))
        if hasattr(hm, "dfda"): 
            hval = hm.dfda(eps,alp)
        else:
            hval = undef()
        testch1("dfda", hval, adfda(eps,alp), ndfda(eps,alp))
        if hasattr(hm, "d2fdede"): 
            hval = hm.d2fdede(eps,alp)
        else:
            hval = undef()
        testch1("d2fdede", hval, ad2fdede(eps,alp), nd2fdede(eps,alp))
        if hasattr(hm, "d2fdeda"): 
            hval = hm.d2fdeda(eps,alp)
        else:
            hval = undef()
        testch1("d2fdeda", hval, ad2fdeda(eps,alp), nd2fdeda(eps,alp))
        if hasattr(hm, "d2fdade"): 
            hval = hm.d2fdade(eps,alp)
        else:
            hval = undef()
        testch1("d2fdade", hval, ad2fdade(eps,alp), nd2fdade(eps,alp))
        if hasattr(hm, "d2fdada"): 
            hval = hm.d2fdada(eps,alp)
        else:
            hval = undef()
        testch1("d2fdada", hval, ad2fdada(eps,alp), nd2fdada(eps,alp))

    if hasattr(hm, "g"):
        print("Checking derivatives of g...")
        if hasattr(hm, "dgds"): 
            hval = hm.dgds(sig,alp)
        else:
            hval = undef()
        testch1("dgds", hval, adgds(sig,alp), ndgds(sig,alp))
        if hasattr(hm, "dgda"): 
            hval = hm.dgda(sig,alp)
        else:
            hval = undef()
        testch1("dgda", hval, adgda(sig,alp), ndgda(sig,alp))
        if hasattr(hm, "d2gdsds"): 
            hval = hm.d2gdsds(sig,alp)
        else:
            hval = undef()
        testch1("d2gdsds", hval, ad2gdsds(sig,alp), nd2gdsds(sig,alp))
        if hasattr(hm, "d2gdsda"): 
            hval = hm.d2gdsda(sig,alp)
        else:
            hval = undef()
        testch1("d2gdsda", hval, ad2gdsda(sig,alp), nd2gdsda(sig,alp))
        if hasattr(hm, "d2gdads"): 
            hval = hm.d2gdads(sig,alp)
        else:
            hval = undef()
        testch1("d2gdads", hval, ad2gdads(sig,alp), nd2gdads(sig,alp))
        if hasattr(hm, "d2gdada"): 
            hval = hm.d2gdada(sig,alp)
        else:
            hval = undef()
        testch1("d2gdada", hval, ad2gdada(sig,alp), nd2gdada(sig,alp))
        
    if hasattr(hm, "y_f") and hasattr(hm, "y_g"): 
        print("Checking consistency of y from different routines (if possible)...")
        if hasattr(hm, "f"): 
            testch("from eps and alp and y_f:", hm.y_f(chi_f(eps,alp),eps,alp), 
    	          "...from y_g", hm.y_g(chi_f(eps,alp),sig_f(eps,alp),alp))
        else: 
            print("f not present\n")
            nmiss += 1
        if hasattr(hm, "g"): 
            testch("from sig and alp and y_g:", hm.y_g(chi_g(sig,alp),sig,alp), 
    	         "...from y_f", hm.y_f(chi_g(sig,alp),eps_g(sig,alp),alp))
        else: 
            print("g not present\n")
            nmiss += 1
    
    if hasattr(hm, "y_f"):
        print("Checking derivatives of y_f...")
        if hasattr(hm, "dydc_f"): 
            hval = hm.dydc_f(chi,eps,alp)
        else:
            hval = undef()
        testch1("dydc_f", hval, adydc_f(chi,eps,alp), ndydc_f(chi,eps,alp))
        if hasattr(hm, "dyde_f"): 
            hval = hm.dyde_f(chi,eps,alp)
        else:
            hval = undef()
        testch1("dyde_f", hval, adyde_f(chi,eps,alp), ndyde_f(chi,eps,alp))
        if hasattr(hm, "dyda_f"): 
            hval = hm.dyda_f(chi,eps,alp)
        else:
            hval = undef()
        testch1("dyda_f", hval, adyda_f(chi,eps,alp), ndyda_f(chi,eps,alp))

    if hasattr(hm, "y_g"):
        print("Checking derivatives of y_g...")
        if hasattr(hm, "dydc_g"): 
            hval = hm.dydc_g(chi,sig,alp)
        else:
            hval = undef()
        testch1("dydc_g", hval, adydc_g(chi,sig,alp), ndydc_g(chi,sig,alp))
        if hasattr(hm, "dyds_g"): 
            hval = hm.dyds_g(chi,sig,alp)
        else:
            hval = undef()
        testch1("dyds_g", hval, adyds_g(chi,sig,alp), ndyds_g(chi,sig,alp))
        if hasattr(hm, "dyda_g"): 
            hval = hm.dyda_g(chi,sig,alp)
        else:
            hval = undef()
        testch1("dyda_g", hval, adyda_g(chi,sig,alp), ndyda_g(chi,sig,alp))
                
    if hasattr(hm, "f") and hasattr(hm, "g") and hasattr(hm, "w_f") and hasattr(hm, "w_g"): 
        print("Checking consistency of w from different routines...")
        testch("from eps and alp and w_f:", hm.w_f(chi_f(eps,alp),eps,alp), 
              "...from w_g:", hm.w_g(chi_f(eps,alp),sig_f(eps,alp),alp))
        testch("from sig and alp and w_g:", hm.w_g(chi_g(sig,alp),sig,alp), 
              "...from w_f:", hm.w_f(chi_g(sig,alp),eps_g(sig,alp),alp))
    
    if hasattr(hm, "w_f"):
        print("Checking derivative of w_f...")
        if hasattr(hm, "dwdc_f"): 
            hval = hm.dwdc_f(chi,eps,alp)
        else:
            hval = undef()
        testch1("dwdc_f", hval, adwdc_f(chi,eps,alp), ndwdc_f(chi,eps,alp))

    if hasattr(hm, "w_g"):
        print("Checking derivative of w_g...")
        if hasattr(hm, "dwdc_g"): 
            hval = hm.dwdc_g(chi,sig,alp)
        else:
            hval = undef()
        testch1("dwdc_g", hval, adwdc_g(chi,sig,alp), ndwdc_g(chi,sig,alp))
        
    print("Checks complete for:",model+",",npass,"passed,",nfail,"failed,",nmiss,"missed checks")
    if not hasattr(hm, "f"): print("hm.f not present")
    if not hasattr(hm, "g"): print("hm.g not present")
    if not hasattr(hm, "y_f"): print("hm.y_f not present")
    if not hasattr(hm, "y_g"): print("hm.y_g not present")
    if not hasattr(hm, "w_f"): print("hm.w_f not present")
    if not hasattr(hm, "w_g"): print("hm.w_g not present")

def princon(hm):
    print("Constants for model:",hm.file)
    print(hm.const)
    print("Derived values:")
    for i in range(len(hm.const)):
        print(hm.name_const[i] + " =", hm.const[i])

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
            strain_inc_f_spec(deps)
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
       for i in range(nsub): strain_inc_f_spec(deps)
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

def numdiff_1(mode,n_dim,fun,var,alp,vari):
    if mode == 0: num = 0.0
    else: num = np.zeros([n_dim])
    for i in range(n_dim):
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
def numdiff_2(mode,n_dim,n_int,fun,var,alp,alpi):
    if mode == 0: num = np.zeros([n_int])
    else: num = np.zeros([n_int,n_dim])
    for k in range(n_int):
        for i in range(n_dim):
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
def numdiff_3(mode,n_dim,n_int,n_y,fun,chi,var,alp,chii):
    if mode == 0: num = np.zeros([n_y,n_int])
    else: num = np.zeros([n_y,n_int,n_dim])
    for k in range(n_int):
        for i in range(n_dim):   
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
def numdiff_4(mode,n_dim,n_int,n_y,fun,chi,var,alp,vari):
    if mode == 0: num = np.zeros([n_y])
    else: num = np.zeros([n_y,n_dim])
    for i in range(n_dim):   
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
def numdiff_5(mode,n_dim,n_int,n_y,fun,chi,var,alp,alpi):
    if mode == 0: num = np.zeros([n_y,n_int])
    else: num = np.zeros([n_y,n_int,n_dim])
    for k in range(n_int):
        for i in range(n_dim):   
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
def numdiff_6(mode,n_dim,n_int,fun,chi,var,alp,chii):
    if mode == 0: num = np.zeros([n_int])
    else: num = np.zeros([n_int,n_dim])
    for k in range(n_int):
        for i in range(n_dim):   
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
def numdiff_6a(mode,n_dim,n_int,fun,chi,var,alp,chii):
    if mode == 0: num = np.zeros([n_int])
    else: num = np.zeros([n_int,n_dim])
    for k in range(n_int):
        for i in range(n_dim):   
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
            if abs(f2-f0) > abs(f1-f0):
                if mode == 0: num[k] = (f2 - f0) / chii
                else: num[k,i] = (f2 - f0) / chii
            else:
                if mode == 0: num[k] = (f0 - f1) / chii
                else: num[k,i] = (f0 - f1) / chii
    return num
def numdiff2_1(mode,n_dim,fun,var,alp,vari):
    if mode == 0: 
        num = 0.0
    else: 
        num = np.zeros([n_dim,n_dim])
    for i in range(n_dim):
        for j in range(n_dim):
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
                if mode == 0: 
                    num = (f1 - 2.0*f2 + f3) / (vari**2)
                else: 
                    num[i,i] = (f1 - 2.0*f2 + f3) / (vari**2)
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
def numdiff2_2(mode,n_dim,n_int,fun,var,alp,vari,alpi):
    if mode == 0: 
        num = np.zeros(n_int)
    else: 
        num = np.zeros([n_dim,n_int,n_dim])
    for k in range(n_int):
        for i in range(n_dim):
            for j in range(n_dim):
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
                if mode == 0: 
                    num[k] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
                else: 
                    num[i,k,j] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
    return num
def numdiff2_3(mode,n_dim,n_int,fun,var,alp,vari,alpi):
    if mode == 0: 
        num = np.zeros(n_int)
    else: 
        num = np.zeros([n_int,n_dim,n_dim])
    for k in range(n_int):
        for i in range(n_dim):
            for j in range(n_dim):
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
                if mode == 0: 
                    num[k] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
                else: 
                    num[k,j,i] = (f1 - f2 - f3 + f4) / (4.0*vari*alpi)
    return num
def numdiff2_4(mode, n_dim, n_int, fun, var, alp, alpi):
    if mode == 0: 
        num = np.zeros([n_int,n_int])
    else: 
        num = np.zeros([n_int,n_dim,n_int,n_dim])
    for k in range(n_int):
        for l in range(n_int):
            for i in range(n_dim):
                for j in range(n_dim):
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
                        else: num[k,i,k,i] = (f1 - 2.0*f2 + f3) / (alpi**2)
                        #print(f1)
                        #print(f2)
                        #print(f3)
                        #print(f1-f2)
                        #print(f2-f3)
                        #print(f1-2.0*f2+f3)
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
                        if mode == 0: 
                            num[k,l] = (f1 - f2 - f3 + f4) / (4.0*(alpi**2))
                        else: 
                            num[k,i,l,j] = (f1 - f2 - f3 + f4) / (4.0*(alpi**2))
    return num

def startup():
    global title, model
    global mode, n_dim
    global fform, gform, rate
    global epsi, sigi, alpi, chii
    global recording, test, history, history_rec
    global acc, colour
    global t, high, test_high
    global pref
    # set up default values
    title = ""
    model = "undefined"
    mode = 0
    n_dim = 1
    fform = False
    gform = False
    rate = False
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
    pref = ["analytical", "automatic", "numerical"]
def histrec():
    global history, history_rec, line
    if history: history_rec.append(line)
           
def readv(args):
    global mode, n_dim
    if mode == 0: return float(args[1])
    elif mode == 1: return np.array([float(i) for i in args[1:n_dim+1]])
def readvs(args):
    global mode, n_dim
    if mode == 0: return float(args[0])
    elif mode == 1: return np.array([float(i) for i in args[:n_dim]])

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

def general_inc_f_r(Smat, Emat, dTdt, dt): #for mode 1 only at present
    global eps, sig, alp, chi
    #global P, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    #if iprint==0 and isub==0: print("Using general_inc_f_r for increment")
    global start_inc
    if start_inc: 
        print("Using general_inc_f_r for increment")
        start_inc = False
    P = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede(eps,alp)))
    dalp = dwdc_f(chi,eps,alp)*dt
    deps = np.einsum(ein_b, P, (dTdt - np.einsum("ij,mjk,mk->i", Smat, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)
def general_inc_g_r(Smat, Emat, dTdt, dt): #for mode 1 only at present
    global eps, sig, alp, chi
    #global Q, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
#    if iprint==0 and isub==0: print("Using general_inc_g_r for increment")    
    global start_inc
    if start_inc: 
        print("Using general_inc_g_r for increment")
        start_inc = False
    Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds(sig,alp)))
    dalp = dwdc_g(chi,sig,alp)*dt
    dsig = np.einsum(ein_b, Q, (dTdt + np.einsum("ij,mjk,mk->i", Emat, d2gdsda(sig,alp), dalp)))
    update_g(dt, dsig, dalp)
    
def strain_inc_f_r(deps, dt):
    global eps, sig, alp, chi
#    if iprint==0 and isub==0: print("Using strain_inc_f_r for strain increment")
    global start_inc
    if start_inc: 
        print("Using strain_inc_f_r for increment")
        start_inc = False
    dalp = dwdc_f(chi,eps,alp)*dt
    update_f(dt, deps, dalp)
def stress_inc_g_r(dsig,dt):
    global eps, sig, alp, chi
#    if iprint==0 and isub==0: print("Using stress_inc_g_r for stress increment")
    global start_inc
    if start_inc: 
        print("Using stress_inc_g_r for increment")
        start_inc = False
    dalp = dwdc_g(chi,sig,alp)*dt
    update_g(dt, dsig, dalp)

def strain_inc_g_r(deps, dt):
    global eps, sig, alp, chi
#    if iprint==0 and isub==0: print("Using strain_inc_g_r for strain increment")
    global start_inc
    if start_inc: 
        print("Using strain_inc_f_r for increment")
        start_inc = False
    if mode == 0: 
        D = -1.0 / d2gdsds(sig,alp)
    else: 
        D = -np.linalg.inv(d2gdsds(sig,alp))
    dalp = dwdc_g(chi,sig,alp)*dt
    dsig = np.einsum(ein_b, D, (deps + np.einsum(ein_c, d2gdsda(sig,alp), dalp)))
    update_g(dt, dsig, dalp)
def stress_inc_f_r(dsig, dt):
    global eps, sig, alp, chi
#    if iprint==0 and isub==0: print("Using stress_inc_f_r for stress increment")
    global start_inc
    if start_inc: 
        print("Using stress_inc_f_r for increment")
        start_inc = False
    if mode == 0: 
        C = 1.0/d2fdede(eps,alp)
    else: 
        C = np.linalg.inv(d2fdede(eps,alp))
    dalp = dwdc_f(chi,eps,alp)*dt
    deps = np.einsum(ein_b, C, (dsig - np.einsum(ein_c, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)

def general_inc_f(Smat, Emat, dTdt, dt): #for mode 1 only at present
    global eps, sig, alp, chi
    #global P, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
#    if iprint==0 and isub==0: print("Using general_inc_f for increment")
    global start_inc
    if start_inc: 
        print("Using general_inc_f for increment")
        start_inc = False
    yo = hm.y_f(chi,eps,alp)
    P = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede(eps,alp)))
    dyde_minus = dyde_f(chi,eps,alp) - np.einsum(ein_h, dydc_f(chi,eps,alp), d2fdade(eps,alp))
    dyda_minus = dyda_f(chi,eps,alp) - np.einsum(ein_i, dydc_f(chi,eps,alp), d2fdada(eps,alp))
    temp = np.einsum(ein_f, dyde_minus, P)
    Lmatp = np.einsum(ein_j,
                      (np.einsum("pr,rl,lnk->pnk",temp,Smat,d2fdeda(eps,alp)) - dyda_minus),
                      dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, temp, dTdt)    
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d, L, dydc_f(chi,eps,alp))
    deps = np.einsum(ein_b, P, (dTdt - np.einsum("ij,jmk,mk->i", Smat, d2fdeda(eps,alp), dalp)))
    update_f(dt, deps, dalp)
def general_inc_g(Smat, Emat, dTdt, dt): #for mode 1 only at present
    global eps, sig, alp, chi
    #global Q, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
#    if iprint==0 and isub==0: print("Using general_inc_g for increment")    
    global start_inc
    if start_inc: 
        print("Using general_inc_g for increment")
        start_inc = False
    yo = hm.y_g(chi,sig,alp)
    Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds(sig,alp)))
    dyds_minus = dyds_g(chi,sig,alp) - np.einsum(ein_h, dydc_g(chi,sig,alp), d2gdads(sig,alp))
    dyda_minus = dyda_g(chi,sig,alp) - np.einsum(ein_i, dydc_g(chi,sig,alp), d2gdada(sig,alp))
    temp = np.einsum(ein_f, dyds_minus, Q)
    Lmatp = np.einsum(ein_j,
                      (-np.einsum("pr,rl,lnk->pnk",temp,Emat,d2gdsda(sig,alp)) - dyda_minus),
                      dydc_g(chi,sig,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, temp, dTdt)    
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d, L, dydc_g(chi,sig,alp))
    dsig = np.einsum(ein_b, Q, (dTdt + np.einsum("ij,jmk,mk->i", Emat, d2gdsda(sig,alp), dalp)))
    update_g(dt, dsig, dalp)

def strain_inc_f_spec(deps):
    global eps, sig, alp, chi
    acc = 0.5
    yo = hm.y_f(chi,eps,alp)
    dyde_minus = hm.dyde_f(chi,eps,alp) - np.einsum(ein_h, hm.dydc_f(chi,eps,alp), hm.d2fdade(eps,alp))
    dyda_minus = hm.dyda_f(chi,eps,alp) - np.einsum(ein_i, hm.dydc_f(chi,eps,alp), hm.d2fdada(eps,alp))
    Lmatp = -np.einsum(ein_j, dyda_minus, hm.dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, dyde_minus, deps)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d,L,hm.dydc_f(chi,eps,alp))
    eps = eps + deps
    alp = alp + dalp
    sig = hm.dfde(eps,alp)
    chi = -hm.dfda(eps,alp)
 
def strain_inc_f(deps, dt):
    global eps, sig, alp, chi
#    if iprint==0 and isub==0: print("Using strain_inc_f for strain increment")
    global start_inc
    if start_inc: 
        print("Using strain_inc_f for increment")
        start_inc = False
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
#    if iprint==0 and isub==0: print("Using stress_inc_g for stress increment")
    global start_inc
    if start_inc: 
        print("Using stress_inc_g for increment")
        start_inc = False
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
#    if iprint==0 and isub==0: print("Using strain_inc_g for strain increment")
    global start_inc
    if start_inc: 
        print("Using strain_inc_g for increment")
        start_inc = False
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
#    if iprint==0 and isub==0: print("Using stress_inc_f for stress increment")
    global start_inc
    if start_inc: 
        print("Using stress_inc_f for increment")
        start_inc = False
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
    if hasattr(hm,"step_print"): hm.step_print(t,eps,sig,alp,chi)
    if recording:
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
                print("{:10.4f} {:14.8f} {:14.8f} {:14.8f} {:14.8f}".format(*item[:1+2*n_dim]))
                out_file.write(",".join([str(num) for num in item])+"\n")
    out_file.close()

def plothigh(plt,x,y,col,highl,ax1,ax2):
    plt.plot(x, y, col, linewidth=1)
    for item in highl: 
        plt.plot(x[item[0]:item[1]], y[item[0]:item[1]], 'r')
    plt.plot(0.0,0.0)            
    plt.set_xlabel(greek(ax1))
    plt.set_ylabel(greek(ax2))

def greek(name):
    gnam = name.replace("eps","$\epsilon$")
    gnam = gnam.replace("sig","$\sigma$")
    gnam = gnam.replace("1","$_1$")
    gnam = gnam.replace("2","$_2$")
    gnam = gnam.replace("3","$_3$")
    return gnam
    
def results_graph(pname, axes):
    global test_col, high
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
        plothigh(plt, x, y, test_col[i], high[i], nunit(ix), nunit(iy))
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
        if n_dim == 1: return ["t","eps","sig"]   
        elif n_dim == 2: return ["t","eps1","eps2","sig1","sig2"]   
        elif n_dim == 3: return ["t","eps1","eps2","eps3","sig1","sig2","sig3"] 
def units_():
    if hasattr(hm,"units"): 
        return hm.units
    elif mode == 0: 
        return ["s","-","Pa"]
    else: 
        if n_dim == 1: return ["s","-","Pa"]   
        elif n_dim == 2: return ["s","-","-","Pa","Pa"]   
        elif n_dim == 3: return ["s","-","-","-","Pa","Pa","Pa"]
def nunit(i):
    return names_()[i] + " (" + units_()[i] + ")"

def results_plot(pname):
    global test_col, high
    if pname[-4:] != ".png": pname = pname + ".png"
    if mode == 0:
        plt.rcParams["figure.figsize"]=(6,6)
        plt.title(title)
        fig, ax = plt.subplots()
        for i in range(len(rec)):
            recl = rec[i]
            e = [item[1] for item in recl]
            s = [item[2] for item in recl]
            if test:
                et = [item[1] for item in test_rec]
                st = [item[2] for item in test_rec]
            plothigh(ax, e, s, test_col[i], high[i], nunit(1), nunit(2))
            if test: plothigh(ax, et, st, 'g', test_high, "", "")
        plt.title(title)
        if pname != "null.png": plt.savefig(pname)
        plt.show()
    elif mode == 1:
        plt.rcParams["figure.figsize"]=(8.2,8)
        plt.title(title)
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
        plt.subplots_adjust(wspace=0.5,hspace=0.3)
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
            plothigh(ax1, s1, s2, test_col[i], high[i], nunit(3), nunit(4))
            if test: plothigh(ax1, st1, st2, 'g', test_high, nunit(3), nunit(4))
            plothigh(ax2, e2, s2, test_col[i], high[i], nunit(2), nunit(4))
            if test: plothigh(ax2, et2, st2, 'g', test_high, nunit(2), nunit(4))
            plothigh(ax3, s1, e1, test_col[i], high[i], nunit(3), nunit(1))
            if test: plothigh(ax3, st1, et1, 'g', test_high, nunit(3), nunit(1))
            plothigh(ax4, e2, e1, test_col[i], high[i], nunit(2), nunit(1))
            if test: plothigh(ax4, et2, et1, 'g', test_high, nunit(2), nunit(1))
        if pname != "null.png": plt.savefig(pname)
        plt.show()

def results_plotCS(pname):
    global test_col, high
    if pname[-4:] != ".png": pname = pname + ".png"
    plt.rcParams["figure.figsize"]=(13.0,8.0)
    #fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3)
    ax2 = plt.subplot(2, 3, 2)
    plt.title(title)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 3, 6)
    plt.subplots_adjust(wspace=0.45,hspace=0.3)
    ax4.set_xscale("log")
    ax4.invert_yaxis()
    ax5.invert_yaxis()
    ax6.invert_yaxis()
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
        maxp = np.max(s1)
        maxq = np.max(s2)
        maxp = np.min([maxp,maxq*1.3/hm.M])
        minp = np.max(s1)
        minq = -np.min(s2)
        minp = np.min([minp,minq*1.3/hm.M])
        cslp = np.array([minp, 0.0, maxp])
        cslq = np.array([-minp*hm.M, 0.0, maxp*hm.M])
        plothigh(ax2, s1, s2, test_col[i], high[i], nunit(3), nunit(4))
        if test: plothigh(ax2, st1, st2, 'g', test_high, nunit(3), nunit(4))
        ax2.plot(cslp,cslq,"red")
        plothigh(ax3, e2, s2, test_col[i], high[i], nunit(2), nunit(4))
        if test: plothigh(ax3, et2, st2, 'g', test_high, nunit(2), nunit(4))
        plothigh(ax4, s1, e1, test_col[i], high[i], nunit(3)+" (log scale)", nunit(1))
        if test: plothigh(ax4, st1, et1, 'g', test_high, nunit(3), nunit(1))
        plothigh(ax5, s1, e1, test_col[i], high[i], nunit(3), nunit(1))
        if test: plothigh(ax5, st1, et1, 'g', test_high, nunit(3), nunit(1))
        plothigh(ax6, e2, e1, test_col[i], high[i], nunit(2), nunit(1))
        if test: plothigh(ax6, et2, et1, 'g', test_high, nunit(2), nunit(1))
    if pname != "null.png": plt.savefig(pname)
    plt.show()

def modestart(): # initialise the calculation for the specified mode
    global ein_a, ein_b, ein_c, ein_d, ein_e, ein_f, ein_g, ein_h, ein_i, ein_j
    global sig, sig_inc, sig_targ, sig_cyc, dsig
    global eps, eps_inc, eps_targ, eps_cyc, deps
    global chi, alp, dalp
    global yo
    global n_int, n_y

    #print(hm.n_int)
    n_int = max(1,hm.n_int)
    n_y = max(1,hm.n_y)
    if mode == 0:           # typical usage below
        ein_a = ",->"       # sig eps
        ein_b = ",->"       # D deps -> dsig
        ein_c = "m,m->"     # d2fdeda dalp
        ein_d = "N,Nm->m"   # L dydc
        ein_e = "N,->N"     # dyde deps
        ein_f = "N,->N"     # dyde C
        ein_g = "N,n->Nn"   # temp d2fdeda
        ein_h = "Nm,m->N"   # dydc d2gdade -> dyde
        ein_i = "Nm,mn->Nn" # dydc d2ydada -> dyda
        ein_j = "Nn,Mn->NM" # dyda dydc
        sig = 0.0
        eps = 0.0
        alp  = np.zeros(n_int)
        chi  = np.zeros(n_int)
        dsig = 0.0
        deps = 0.0
        dalp = np.zeros(n_int)
        sig_inc  = 0.0
        sig_targ = 0.0
        sig_cyc  = 0.0
        eps_inc  = 0.0
        eps_targ = 0.0
        eps_cyc  = 0.0
        yo = np.zeros(n_y)
    elif mode == 1:
        ein_a = "i,i->"
        ein_b = "ki,i->k"
        ein_c = "imj,mj->i"
        ein_d = "N,Nmi->mi"
        ein_e = "Ni,i->N"
        ein_f = "Nj,jk->Nk"
        ein_g = "Nk,knl->Nnl"
        ein_h = "Nmi,mij->Nj"
        ein_i = "Nmi,minj->Nnj"
        ein_j = "Nni,Mni->NM"
        sig = np.zeros(n_dim)
        eps = np.zeros(n_dim)
        alp  = np.zeros([n_int,n_dim])
        chi  = np.zeros([n_int,n_dim])
        dsig = np.zeros(n_dim)
        deps = np.zeros(n_dim)
        dalp = np.zeros([n_int,n_dim])
        sig_inc  = np.zeros(n_dim)
        sig_targ = np.zeros(n_dim)
        sig_cyc  = np.zeros(n_dim)
        eps_inc  = np.zeros(n_dim)
        eps_targ = np.zeros(n_dim)
        eps_cyc  = np.zeros(n_dim)
        yo = np.zeros(n_y)
    elif mode == 2:
        ein_a = "ij,ij->"
        ein_b = "klij,ij->kl"
        ein_c = "ijmkl,mkl->ij"
        ein_d = "N,Nmij->mij"
        ein_e = "Nij,ij->N"
        ein_f = "Nij,ijkl->Nkl"
        ein_g = "Nkl,klnij->Nnij"
        ein_h = "Nmij,mijkl->Nkl"
        ein_i = "Nmij,mijnkl->Nnkl"
        ein_j = "Nnij,Mnij->NM"
        sig = np.zeros([n_dim,n_dim])
        eps = np.zeros([n_dim,n_dim])
        alp  = np.zeros([n_int,n_dim,n_dim])
        chi  = np.zeros([n_int,n_dim,n_dim])
        dsig = np.zeros([n_dim,n_dim])
        deps = np.zeros([n_dim,n_dim])
        dalp = np.zeros([n_int,n_dim,n_dim])
        sig_inc  = np.zeros([n_dim,n_dim])
        sig_targ = np.zeros([n_dim,n_dim])
        sig_cyc  = np.zeros([n_dim,n_dim])
        eps_inc  = np.zeros([n_dim,n_dim])
        eps_targ = np.zeros([n_dim,n_dim])
        eps_cyc  = np.zeros([n_dim,n_dim])
        yo = np.zeros(n_y)
    else:
        error("Mode not recognised:" + mode)        

# Drive - driving routine for hyperplasticity models
def drive(arg="hyper.dat"): 
    print("")
    print("+-----------------------------------------------------------------------------+")
    print("| HyperDrive: driving routine for hyperplasticity models                      |")
    print("| (c) G.T. Houlsby, 2018-2020                                                 |")
    print("|                                                                             |")
    print("| \x1b[1;31mThis program is provided in good faith, but with no warranty of correctness\x1b[0m |")
    print("+-----------------------------------------------------------------------------+")
    
    print("Current directory: " + os.getcwd())
    if os.path.isfile("HyperDrive.cfg"):
        input_file = open("HyperDrive.cfg", 'r')
        last = input_file.readline().split(",")
        drive_file_name = last[0]
        check_file_name = last[1]
        input_file.close()
    else:
        drive_file_name = "hyper.dat"
        check_file_name = "unknown"
    if arg != "hyper.dat":
        drive_file_name = arg
    input_found = False
    count = 0
    while not input_found:
        count += 1
        if count > 5: error("Too many tries")
        drive_file_temp = input("Enter input filename [" + drive_file_name + "]: ",)
        if len(drive_file_temp) > 0: 
            drive_file_name = drive_file_temp
        if drive_file_name[-4:] != ".dat": 
            drive_file_name = drive_file_name + ".dat"
        if os.path.isfile(drive_file_name):
            print("Reading from file: " + drive_file_name)
            print("")
            input_file = open(drive_file_name, 'r')
            input_found = True
        else:
            print("File not found:", drive_file_name)
    last_file = open("HyperDrive.cfg", 'w')
    last_file.writelines([drive_file_name+","+check_file_name])
    last_file.close()
    
    startup()
    process(input_file)
    
    print("End of (processing from) file: " + drive_file_name)
    input_file.close()
    
if __name__ == "__main__":
    print("\nHyperDrive routines loaded")
    print("(c) G.T. Houlsby 2018-2020\n")
    print("Usage:")
    print("  drive('datafile') - run HyperDrive taking input from 'datafile.dat'")
    print("  drive()           - run HyperDrive taking input from last datafile used")
    print("  check('model')    - run HyperCheck on code in 'model.py'")
    print("  check()           - run HyperCheck on last code tested")