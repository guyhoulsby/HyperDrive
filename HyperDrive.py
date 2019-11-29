#HyperDrive - driving routine for hyperplasticity models
import sys
import re
import importlib
import os.path
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

#JUST A LITTLE COMMENT TO TRAIN 

#Another comment to try out the branch function

hu = importlib.import_module("HyperUtils")

# set up default values
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
model = "undefined"
title = ""
acc = 0.5 # rate acceleration factor
colour = "b"

# initialise some key variables
t = 0.0
high = [[]]
test_high = []
history_rec = []

def process(source):
    global textsplit, hm
    global mode, numerical, ndim, fform, gform, rate, recording, test, test_rec
    global history, history_rec, model, title, acc, t, colour, high, test_high
    global n_stage, n_cyc, n_print, iprint, isub, rec, curr_test, test_col
    global eps, sig, alp, chi
    
    for line in source:
        text = line.rstrip("\n\r")
        #print("Read: ", text)
        textsplit = re.split(r'[ ,;]',text)
        #print("Split:", textsplit)
        keyword = textsplit[0]
        if len(keyword) == 0:
            continue
        if keyword[:1] == "#":
            continue
        elif keyword == "*model":
            model_temp = textsplit[1]
            if os.path.isfile(model_temp + ".py"):
                model = model_temp + ""
                print("Importing hyperplasticity model: ", model)
                hm = importlib.import_module(model)
                #hm.setvals()
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
            for fun in ["dfde","dfda","dgds","dgda","dydc_f","dydc_g","dyde_f","dyds_g","dyda_f","dyda_g","dwdc_f","dwdc_g",
                        "d2fdede","d2fdeda","d2fdade","d2fdada","d2gdsds","d2gdsda","d2gdads","d2gdada"]:
                if not hasattr(hm, fun): print(model+"."+fun+" not present - will use numerical differentation for this function if required")
        elif keyword == "*title":
            title = text.replace("*title ", "")
            print("Title: ", title)
        elif keyword == "*mode":
            mode = int(textsplit[1])
            if mode == 0: ndim = 1
            else: ndim = int(textsplit[2])
            print("Mode:", mode, "ndim:", ndim)
        elif keyword == "*f-form":
            print("Setting f-form")
            fform = True
        elif keyword == "*g-form":
            print("Setting g-form")
            gform = True
        elif keyword == "*colour":
            colour = textsplit[1]
        elif keyword == "*rate":
            print("Setting rate dependent analysis")
            rate = True
            if len(textsplit) > 1: hm.mu = float(textsplit[1])
        elif keyword == "*rateind":
            print("Setting rate independent analysis")
            rate = False
        elif keyword == "*numerical":
            numerical = True
            print("Setting numerical differentiation for all functions")
        elif keyword == "*analytical":
            numerical = False
            print("Unsetting numerical differentiation")
        elif keyword == "*const":
            if model != "undefined":
                #print("Constants text:", textsplit[1:])
                hm.const = [float(tex) for tex in textsplit[1:]]
                print("Constants:", hm.const)
                hm.deriv()
                hu.princon(hm)
            else:
                print("Cannot read constants: model undefined")
        elif keyword == "*tweak":
            if model != "undefined":
                #print("Constants text:", textsplit[1:])
                ctweak = textsplit[1]
                vtweak = float(textsplit[2])
                for i in range(len(hm.const)):
                    if ctweak == hm.name_const[i]: 
                        hm.const[i] = vtweak 
                        print("Tweaked constant value:",hm.name_const[i],"set to", hm.const[i])
                hm.deriv()
                #hu.princon(hm)
            else:
                print("Cannot tweak constants: model undefined")
        elif keyword == "*const_from_points":
            if model != "undefined":
                npt = int(textsplit[1])
                epsd = np.zeros(npt)
                sigd = np.zeros(npt)
                for ipt in range(npt):
                    temp = float(textsplit[2+2*ipt])
                    epsd[ipt] = float(textsplit[2+2*ipt])
                    sigd[ipt] = float(textsplit[3+2*ipt])
                Einf = float(textsplit[2+2*npt])
                epsmax = float(textsplit[3+2*npt])
                HARM_R = float(textsplit[4+2*npt])
                hm.const = hu.derive_from_points(model,epsd,sigd,Einf,epsmax,HARM_R)
                print("Constants from points:", hm.const)
                hm.deriv()
                hu.princon(hm)
            else:
                print("Model undefined")
        elif keyword == "*const_from_curve":
            if model != "undefined":
                curve = textsplit[1]
                npt = int(textsplit[2])
                maxsig = float(textsplit[3])
                HARM_R = float(textsplit[4])
                param = np.zeros(3)
                param[0:3] = [float(item) for item in textsplit[5:8]]
                epsd = np.zeros(npt)
                sigd = np.zeros(npt)
                print("Calculated points from curve:")
                for ipt in range(npt):
                    sigd[ipt] = maxsig*float(ipt+1)/float(npt)
                    if curve == "power":                    
                        epsd[ipt] = sigd[ipt]/param[0] + param[1]*(sigd[ipt]/maxsig)**param[2]
                    if curve == "jeanjean":                    
                        epsd[ipt] = sigd[ipt]/param[0] + param[1]*(sigd[ipt]/maxsig)**param[2]
                    print(epsd[ipt],sigd[ipt])
                Einf = 0.5*(sigd[npt-1]-sigd[npt-2])/(epsd[npt-1]-epsd[npt-2])
                epsmax = epsd[npt-1]
                hm.const = hu.derive_from_points(model,epsd,sigd,Einf,epsmax,HARM_R)
                print("Constants from curve:", hm.const)
                hm.deriv()
                hu.princon(hm)
            else:
                print("Model undefined")
        elif keyword == "*const_from_data":
            if model != "undefined":
                dataname = textsplit[1]
                if dataname[-4:] != ".csv": dataname = dataname + ".csv"
                print("Reading data from", dataname)
                data_file = open(dataname,"r")
                data_text = data_file.readlines()
                data_file.close()
                for i in range(len(data_text)):
                    data_split = re.split(r'[ ,;]',data_text[i])
                    ttest = float(data_split[0])
                    epstest = float(data_split[1])
                    sigtest = float(data_split[2])
                npt = int(textsplit[2])
                HARM_R = float(textsplit[3])
                print("Calculating const from data")
                print("Constants from data:", hm.const)
                hm.deriv()
                hu.princon(hm)
            else:
                print("Model undefined")
        elif keyword == "*acc":
            acc = float(textsplit[1])
            print("Acceleration factor:", acc)
        elif keyword == "*start":
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
            last = start
            #printderivs()
        elif keyword == "*restart":
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
        elif keyword == "*rec":
            print("Recording data")
            recording = True
        elif keyword == "*stoprec":
            if mode == 0: temp = np.nan
            else: temp = np.full(ndim, np.nan)
            record(temp, temp) # write a line of nan to split plots
            print("Stop recording data")
            recording = False
        elif keyword == "*init_stress":
            if history: history_rec.append(line)
            sig = readvs()
            eps = eps_g(sig,alp)
            print("Initial stress:", sig)
            print("Initial strain:", eps)
            delrec()
            record(eps, sig)
        elif keyword == "*init_strain":
            if history: history_rec.append(line)
            eps = readvs()
            sig = sig_f(eps,alp)
            print("Initial strain:", eps)
            print("Initial stress:", sig)
            delrec()
            record(eps, sig)
        elif keyword == "*general_inc":
            if history: history_rec.append(line)
            Smat = np.reshape(np.array([float(i) for i in textsplit[1:ndim*ndim+1]]), (ndim, ndim))
            Emat = np.reshape(np.array([float(i) for i in textsplit[ndim*ndim+1:2*ndim*ndim+1]]), (ndim, ndim))
            Tdt = np.array([float(i) for i in textsplit[2*ndim*ndim+1:2*ndim*ndim+ndim+1]])
            dt = float(textsplit[2*ndim*ndim+ndim+1])
            nprint = int(textsplit[2*ndim*ndim+ndim+2])
            nsub = int(textsplit[2*ndim*ndim+ndim+3])
            dTdt = Tdt / float(nprint*nsub)
            print("General control increment:")
            print("S   =", Smat)
            print("E   =", Emat)
            print("Tdt =", Tdt)
            for iprint in range(nprint):
                for isub in range(nsub): general_inc(Smat, Emat, dTdt, dt)
                record(eps, sig)
            print("Increment complete")
        elif keyword == "*general_cyc":
            if history: history_rec.append(line)
            Smat = np.reshape(np.array([float(i) for i in textsplit[1:ndim*ndim+1]]), (ndim, ndim))
            Emat = np.reshape(np.array([float(i) for i in textsplit[ndim*ndim+1:2*ndim*ndim+1]]), (ndim, ndim))
            Tdt = np.array([float(i) for i in textsplit[2*ndim*ndim+1:2*ndim*ndim+ndim+1]])
            tper = float(textsplit[2*ndim*ndim+ndim+1])
            ctype = textsplit[2*ndim*ndim+ndim+2]
            ncyc = int(textsplit[2*ndim*ndim+ndim+3])
            nprint = int(textsplit[2*ndim*ndim+ndim+4])
            if nprint%2 == 1: nprint += 1
            nsub = int(textsplit[2*ndim*ndim+ndim+5])
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
        elif keyword == "*strain_inc":
            if history: history_rec.append(line)
            eps_inc = readv()
            nprint = int(textsplit[ndim+2])
            nsub = int(textsplit[ndim+3])
            t_inc = float(textsplit[1])
            deps = eps_inc / float(nprint*nsub)
            dt = t_inc / float(nprint*nsub)
            print("Strain increment:", eps_inc, "deps =", deps)
            for iprint in range(nprint):
                for isub in range(nsub): strain_inc(deps,dt)
                record(eps, sig)
            print("Increment complete")
        elif keyword == "*strain_targ":
            if history: history_rec.append(line)
            eps_targ = readv()
            nprint = int(textsplit[ndim+2])
            nsub = int(textsplit[ndim+3])
            t_inc = float(textsplit[1])
            deps = (eps_targ - eps) / float(nprint*nsub)
            dt = t_inc / float(nprint*nsub)
            print("Strain target:", eps_targ, "deps =", deps)
            for iprint in range(nprint):
                for isub in range(nsub): strain_inc(deps,dt)
                record(eps, sig)
            print("Increment complete")
        elif keyword == "*strain_cyc":
            if history: history_rec.append(line)
            eps_cyc = readv()
            tper = float(textsplit[1])
            ctype = textsplit[ndim+2]
            ncyc = int(textsplit[ndim+3])
            nprint = int(textsplit[ndim+4])
            if nprint%2 == 1: nprint += 1
            nsub = int(textsplit[ndim+5])
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
        elif keyword == "*stress_inc":
            if history: history_rec.append(line)
            sig_inc = readv()
            nprint = int(textsplit[ndim+2])
            nsub = int(textsplit[ndim+3])
            t_inc = float(textsplit[1])
            dsig = sig_inc / float(nprint*nsub)
            dt = t_inc / float(nprint*nsub)
            print("Stress increment:", sig_inc,"dsig =", dsig,dt)
            for iprint in range(nprint):
                for isub in range(nsub): stress_inc(dsig,dt)
                record(eps, sig)
            print("Increment complete")
        elif keyword == "*stress_targ":
            if history: history_rec.append(line)
            sig_targ = readv()
            nprint = int(textsplit[ndim+2])
            nsub = int(textsplit[ndim+3])
            t_inc = float(textsplit[1])
            dsig = (sig_targ - sig) / float(nprint*nsub)
            dt = t_inc / float(nprint*nsub)
            print("Stress target:", sig_targ, "dsig =", dsig)
            for iprint in range(nprint):
                for isub in range(nsub): stress_inc(dsig,dt)
                record(eps, sig)
            print("Increment complete")
        elif keyword == "*stress_cyc":
            if history: history_rec.append(line)
            sig_cyc = readv()
            tper = float(textsplit[1])
            ctype = textsplit[ndim+2]
            ncyc = int(textsplit[ndim+3])
            nprint = int(textsplit[ndim+4])
            if nprint%2 == 1: nprint += 1
            nsub = int(textsplit[ndim+5])
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
        elif keyword == "*strain_test" or keyword == "*stress_test":
            if history: history_rec.append(line)
            test = True
            testname = textsplit[1]
            if testname[-4:] != ".csv": testname = testname + ".csv"
            print("Reading from", testname)
            test_file = open(testname,"r")
            test_text = test_file.readlines()
            test_file.close()
            nsub = int(textsplit[2])
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
                #print(epstest,sigtest)
                if iprint == 0:
                    eps = copy.deepcopy(epstest)
                    sig = copy.deepcopy(sigtest)
                    delrec()
                    record(eps, sig)
                    recordt(epstest, sigtest)
                else:
                    dt = (ttest - t) / float(nsub)
                    if keyword == "*strain_test":
                        deps = (epstest - eps) / float(nsub)
                        for isub in range(nsub): strain_inc(deps,dt)
                    elif keyword == "*stress_test":
                        dsig = (sigtest - sig) / float(nsub)
                        for isub in range(nsub): stress_inc(dsig,dt)                    
                    recordt(epstest, sigtest)
                    record(eps, sig)
        elif keyword == "*strain_path" or keyword == "*stress_path":
            if history: history_rec.append(line)
            testname = textsplit[1]
            if testname[-4:] != ".csv": testname = testname + ".csv"
            print("Reading from", testname)
            test_file = open(testname,"r")
            test_text = test_file.readlines()
            test_file.close()
            nsub = int(textsplit[2])
            for iprint in range(len(test_text)):
                test_split = re.split(r'[ ,;]',test_text[iprint])
                ttest = float(test_split[0])
                if mode==0:
                    val = float(test_split[1])
                else:                
                    val = np.array([float(tex) for tex in test_split[1:ndim+1]])
                if iprint == 0:
                    if keyword == "*strain_path":
                        eps = copy.deepcopy(val)
                    elif keyword == "*stress_path":
                        sig = copy.deepcopy(val)
                    delrec()
                    record(eps, sig)
                else:
                    dt = (ttest - t) / float(nsub)
                    if keyword == "*strain_path":
                        deps = (val - eps) / float(nsub)
                        for isub in range(nsub): strain_inc(deps,dt)
                    elif keyword == "*stress_path":
                        dsig = (val - sig) / float(nsub)
                        for isub in range(nsub): stress_inc(dsig,dt)                    
                    record(eps, sig)
        elif keyword == "*high":
            if history: history_rec.append(line)
            hstart = len(rec[curr_test])-1
        elif keyword == "*unhigh":
            if history: history_rec.append(line)
            hend = len(rec[curr_test])
            high[curr_test].append([hstart,hend])
        elif keyword == "*plot":
            pname = "hplot_" + model
            if len(textsplit) > 1: pname = textsplit[1]
            results_plot(pname)
        elif keyword == "*graph":
            pname = "hplot_" + model
            axes = textsplit[1:3]
            if len(textsplit) > 3: pname = textsplit[3]
            results_graph(pname, axes)
        elif keyword == "*specialplot":
            pname = "hplot_" + model
            if len(textsplit) > 1: pname = textsplit[1]
            hm.results_plot(pname, rec)
        elif keyword == "*print":
            oname = "hout_" + model
            if len(textsplit) > 1: oname = textsplit[1]
            results_print(oname)
        elif keyword == "*specialprint":
            oname = "hout_" + model
            if len(textsplit) > 1: oname = textsplit[1]
            hm.results_print(oname)
        elif keyword == "*start_history":
            history = True
            history_rec = []
        elif keyword == "*end_history":
            history = False
        elif keyword == "*run_history":
            history = False
            if len(textsplit) > 1: runs = int(textsplit[1])
            else: runs = 1
            for irun in range(runs):
                process(history_rec)
        elif keyword == "*end":
            now = time.process_time()
            print("Time:",now-start,now-last)
            print("End of test")
            break
        else:
            print("\x1b[0;31mWARNING - keyword not recognised:",keyword,"\x1b[0m")

#define some utilities

def readv():
    if mode == 0: return float(textsplit[2])
    elif mode == 1: return np.array([float(i) for i in textsplit[2:ndim+2]])
def readvs():
    if mode == 0: return float(textsplit[1])
    elif mode == 1: return np.array([float(i) for i in textsplit[1:ndim+1]])
        
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
    global t
    if rate:
        if gform: strain_inc_g_r(deps,dt)
        else: strain_inc_f_r(deps,dt) #default is f-form for this case, even if not set explicitly
    else:
        if gform: strain_inc_g(deps)
        else: strain_inc_f(deps) #default is f-form for this case, even if not set explicitly
    t = t + dt

def stress_inc(dsig, dt):
    global t
    if rate:
        if fform: stress_inc_f_r(dsig,dt)
        else: stress_inc_g_r(dsig,dt) #default is g-form for this case, even if not set explicitly
    else:
        if fform: stress_inc_f(dsig)
        else: stress_inc_g(dsig) #default is g-form for this case, even if not set explicitly
    t = t + dt
    
def general_inc(Smat, Emat, dTdt, dt):
    global t
    if rate:
        if fform: general_inc_f_r(Smat, Emat, dTdt, dt)
        elif gform: general_inc_g_r(Smat, Emat, dTdt, dt)
        else: error("Error in general_inc - f-form or g-form needs to be specified")
    else:
        if fform: general_inc_f(Smat, Emat, dTdt)
        elif gform: general_inc_g(Smat, Emat, dTdt)
        else: error("Error in general_inc - f-form or g-form needs to be specified")
    t = t + dt

def update_f(deps, dalp):
    global eps, sig, alp, chi
    eps = eps + deps
    alp = alp + dalp
    sigold = copy.deepcopy(sig)
    chiold = copy.deepcopy(chi)
    sig = sig_f(eps,alp)
    chi = chi_f(eps,alp)
    dsig = sig - sigold
    dchi = chi - chiold
    if hasattr(hm,"update"): hm.update(eps,sig,alp,chi,deps,dsig,dalp,dchi)

def update_g(dsig, dalp):
    global eps, sig, alp, chi
    sig = sig + dsig
    alp = alp + dalp
    epsold = copy.deepcopy(eps)
    chiold = copy.deepcopy(chi)
    eps = eps_g(sig,alp)
    chi = chi_g(sig,alp)
    deps = eps - epsold
    dchi = chi - chiold
    if hasattr(hm,"update"): hm.update(eps,sig,alp,chi,deps,dsig,dalp,dchi)

def general_inc_f_r(Smat, Emat, dTdt, dt):
    global eps, sig, alp, chi
    #global P, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    if iprint==0 and isub==0: print("Using general_inc_f_r for increment")
    P = np.linalg.inv(Emat + np.einsum("ij,jk->ik", Smat, d2fdede(eps,alp)))
    dalp = dwdc_f(chi,eps,alp)*dt
    deps = np.einsum(ein_b, P, (dTdt - np.einsum("ij,mjk,mk->i", Smat, d2fdeda(eps,alp), dalp)))
    update_f(deps, dalp)

def general_inc_g_r(Smat, Emat, dTdt, dt):
    global eps, sig, alp, chi
    #global Q, temp, Lmatp, Lrhsp, Lmat, Lrhs, L, yo
    if iprint==0 and isub==0: print("Using general_inc_g_r for increment")    
    Q = np.linalg.inv(Smat - np.einsum("ij,jk->ik", Emat, d2gdsds(sig,alp)))
    dalp = dwdc_g(chi,sig,alp)*dt
    dsig = np.einsum(ein_b, Q, (dTdt + np.einsum("ij,mjk,mk->i", Emat, d2gdsda(sig,alp), dalp)))
    update_g(dsig, dalp)
    
def strain_inc_f_r(deps,dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_f_r for strain increment")
    dalp = dwdc_f(chi,eps,alp)*dt
    update_f(deps, dalp)

def strain_inc_g_r(deps,dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_g_r for strain increment")
    if mode == 0: D = -1.0 / d2gdsds(sig,alp)
    else: D = -np.linalg.inv(d2gdsds(sig,alp))
    dalp = dwdc_g(chi,sig,alp)*dt
    dsig = np.einsum(ein_b, D, (deps + np.einsum(ein_c, d2gdsda(sig,alp), dalp)))
    update_g(dsig, dalp)

def stress_inc_g_r(dsig,dt):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using stress_inc_g_r for stress increment")
    dalp = dwdc_g(chi,sig,alp)*dt
    update_g(dsig, dalp)

def stress_inc_f_r(dsig,dt):
    global eps, sig, alp, chi
    #global C, dyde_minus, dyda_minus, Lmatp, Lrhsp
    if iprint==0 and isub==0: print("Using stress_inc_f_r for stress increment")
    if mode == 0: C = 1.0/d2fdede(eps,alp)
    else: C = np.linalg.inv(d2fdede(eps,alp))
    dalp = dwdc_f(chi,eps,alp)*dt
    deps = np.einsum(ein_b, C, (dsig - np.einsum(ein_c, d2fdeda(eps,alp), dalp)))
    update_f(deps, dalp)

def general_inc_f(Smat, Emat, dTdt):
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
    update_f(deps, dalp)
    
def general_inc_g(Smat, Emat, dTdt):
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
 
def strain_inc_f(deps):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using strain_inc_f for strain increment")
#    if isub==0: print("iprint =", iprint)
    yo = hm.y_f(chi,eps,alp)
    dyda_minus = dyda_f(chi,eps,alp) - np.einsum(ein_i, dydc_f(chi,eps,alp), d2fdada(eps,alp))
    dyde_minus = dyde_f(chi,eps,alp) - np.einsum(ein_h, dydc_f(chi,eps,alp), d2fdade(eps,alp))
    Lmatp = -np.einsum(ein_j, dyda_minus, dydc_f(chi,eps,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, dyde_minus, deps)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d,L,dydc_f(chi,eps,alp))
    update_f(deps, dalp)
    
def strain_inc_g(deps):
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
    update_g(dsig, dalp)

def stress_inc_g(dsig):
    global eps, sig, alp, chi
    if iprint==0 and isub==0: print("Using stress_inc_g for stress increment")
    yo = hm.y_g(chi,sig,alp)
    dyda_minus = dyda_g(chi,sig,alp) - np.einsum(ein_i, dydc_g(chi,sig,alp), d2gdada(sig,alp))
    dyds_minus = dyds_g(chi,sig,alp) - np.einsum(ein_h, dydc_g(chi,sig,alp), d2gdads(sig,alp))
    Lmatp = -np.einsum(ein_j, dyda_minus, dydc_g(chi,sig,alp))
    Lrhsp = acc*yo + np.einsum(ein_e, dyds_minus, dsig)
    L = solve_L(yo, Lmatp, Lrhsp)
    dalp = np.einsum(ein_d,L,dydc_g(chi,sig,alp))
    update_g(dsig, dalp)

def stress_inc_f(dsig):
    global eps, sig, alp, chi
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
    update_f(deps, dalp)

def record(eps, sig):
    result = True
    if recording:
#        print(t,eps,sig)
        if mode == 0:
            if np.isnan(eps) or np.isnan(sig):
                result = False
            else:
                rec[curr_test].append(np.concatenate(([t],[eps],[sig])))
        else:
            if np.isnan(sum(eps)) or np.isnan(sum(sig)):
                result = False
            else:
                rec[curr_test].append(np.concatenate(([t],eps,sig)))
    return result

def delrec():
    del rec[curr_test][-1]

def recordt(eps, sig):
    if mode == 0: test_rec.append(np.concatenate(([t],[eps],[sig])))
    else: test_rec.append(np.concatenate(([t],eps,sig)))

def results_print(oname):
    print("")
    if oname[-4:] != ".csv": oname = oname + ".csv"
    out_file = open(oname, 'w')
    for recl in rec:
        if mode == 0:
            if hasattr(hm,"names"): names = hm.names
            else: names = ["t","eps","sig"]
            print("{:>10} {:>14} {:>14}".format(*names))
            out_file.write(",".join(names)+"\n")
            for item in recl:
                print("{:10.4f} {:14.8f} {:14.8f}".format(*item))
                out_file.write(",".join([str(num) for num in item])+"\n")
        elif mode == 1:
            if hasattr(hm,"names"): names = hm.names
            else: names = ["t","eps1","eps2","sig1","sig2"]
            print("{:>10} {:>14} {:>14} {:>14} {:>14}".format(*names))
            out_file.write(",".join(names)+"\n")
            for item in recl:
                print("{:10.4f} {:14.8f} {:14.8f} {:14.8f} {:14.8f}".format(*item))
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
    #global x, y
    if pname[-4:] != ".png": pname = pname + ".png"
    if hasattr(hm,"names"): names = hm.names
    elif mode == 0: names = ["t","eps","sig"]
    else: names = ["t","eps1","eps2","sig1","sig2"]
    plt.rcParams["figure.figsize"]=(6,6)
    for i in range(len(rec)):
        recl = rec[i]
        for j in range(len(names)):
            if axes[0] == names[j]: x = [item[j] for item in recl]
            if axes[1] == names[j]: y = [item[j] for item in recl]
        plothigh(x, y, test_col[i], high[i], axes[0], axes[1])
    print("Graph of",axes[1],"v.",axes[0])
    plt.title(title)
    if pname != "null.png": plt.savefig(pname)
    plt.show()

def results_plot(pname):
    #global x, y
    if pname[-4:] != ".png": pname = pname + ".png"
    if mode == 0:
# old code
#        plt.subplot()
#        fig, ax = plt.subplots()
#        ax.plot(eps_rec, sig_rec)
#        ax.set(xlabel=hm.name_eps, ylabel=hm.name_sig, title=title)
#        ax.grid()
#        fig.savefig("hyper.png")
        if hasattr(hm,"names"): names = hm.names
        else: names = ["t","eps","sig"]
        plt.rcParams["figure.figsize"]=(6,6)
        for i in range(len(rec)):
            recl = rec[i]
            e = [item[1] for item in recl]
            s = [item[2] for item in recl]
            if test:
                et = [item[1] for item in test_rec]
                st = [item[2] for item in test_rec]
            plothigh(e, s, test_col[i], high[i], names[1], names[2])
            if test: plothigh(et, st, 'g', test_high, "", "")
        plt.title(title)
        if pname != "null.png": plt.savefig(pname)
        plt.show()
    elif mode == 1:
        #fig, ax = plt.subplots()
        if hasattr(hm,"names"): names = hm.names
        else: names = ["t","eps1","eps2","sig1","sig2"]
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
            plothigh(s1, s2, test_col[i], high[i], names[3],names[4])
            if test: plothigh(st1, st2, 'g', test_high, names[3],names[4])
            plt.subplot(2,2,2)
            plothigh(e1, e2, test_col[i], high[i], names[1],names[2])
            if test: plothigh(et1, et2, 'g', test_high, names[1],names[2])
            plt.subplot(2,2,3)
            plothigh(e1, s1, test_col[i], high[i], names[1],names[3])
            if test: plothigh(et1, st1, 'g', test_high, names[1],names[3])
            plt.subplot(2,2,4)
            plothigh(e2, s2, test_col[i], high[i], names[2],names[4])
            if test: plothigh(et2, st2, 'g', test_high, names[2],names[4])
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
        
#main program
print("")
print("+-------------------------------------------------------------------------------+")
print("|  HyperDrive: driving routine for hyperplasticity models                       |")
print("|  (c) G.T. Houlsby, 2018, 2019                                                 |")
print("|                                                                               |")
print("|  \x1b[0;31mThis program is provided in good faith, but with no warranty of correctness\x1b[0m  |")
print("+-------------------------------------------------------------------------------+")

input_file_name = "hyper.dat"
if os.path.isfile("hyperdrivelast.dat"):
    input_file = open("hyperdrivelast.dat", 'r')
    input_file_name = input_file.readline()
    input_file.close() 

input_found = False
while not input_found:
    input_file_temp = input("Enter input filename [" + input_file_name + "]: ",)
    #input_file_temp = input_file_name
    if len(input_file_temp) > 0: input_file_name = input_file_temp
    if input_file_name[-4:] != ".dat": input_file_name = input_file_name + ".dat"
    if os.path.isfile(input_file_name):
        print("Reading from file: " + input_file_name)
        print("")
        input_file = open(input_file_name, 'r')
        input_found = True
    else:
        print("File not found:", input_file_name)

last_file = open("hyperdrivelast.dat", 'w')
last_file.write(input_file_name)
last_file.close()

process(input_file)
print("End of (processing from) file: " + input_file_name)
input_file.close()
