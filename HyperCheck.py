#HyperCheck - check some aspects of plasticity models
import sys
import re
import importlib
import os.path
import numpy as np

def error():
    sys.exit()
    
def test(text1,val1,text2,val2):
    global npass, nfail
    print(text1, val1)
    print(text2, val2)
    if all(np.isclose(val1, val2, rtol=0.0001, atol=0.000001).reshape(-1)): 
        print("***PASSED***\n")
        npass += 1
    else:
        print("\x1b[0;31m",np.isclose(val1, val2, rtol=0.0001, atol=0.000001),"\x1b[0m")
        print("\x1b[0;31m***FAILED***\n\x1b[0m")
        nfail += 1
        input("Hit ENTER to continue",)
        
def sig_f(eps, alp): return  dfde(eps, alp)
def chi_f(eps, alp): return -dfda(eps, alp)
def eps_g(sig, alp): return -dgds(sig, alp)
def chi_g(sig, alp): return -dgda(sig, alp)
def dfde(eps,alp):
    if not hasattr(hm,"dfde"): return hu.numdiff_1(mode,ndim,hm.f,eps,alp,epsi)
    else: return hm.dfde(eps,alp)
def dfda(eps,alp):
    if not hasattr(hm,"dfda"): return hu.numdiff_2(mode,ndim,n_int,hm.f,eps,alp,alpi)
    else: return hm.dfda(eps,alp)
def dgds(sig,alp):
    if not hasattr(hm,"dgds"): return hu.numdiff_1(mode,ndim,hm.g,sig,alp,sigi)
    else: return hm.dgds(sig,alp)
def dgda(sig,alp):
    if not hasattr(hm,"dgda"): return hu.numdiff_2(mode,ndim,n_int,hm.g,sig,alp,alpi)
    else: return hm.dgda(sig,alp)
def d2fdede(eps,alp):
    if not hasattr(hm,"d2fdede"): return hu.numdiff2_1(mode,ndim,hm.f,eps,alp,epsi)
    else: return hm.d2fdede(eps,alp)
def d2gdsds(sig,alp):
    if not hasattr(hm,"d2gdsds"): return hu.numdiff2_1(mode,ndim,hm.g,sig,alp,sigi)
    else: return hm.d2gdsds(sig,alp)

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

def modestart_short():
    global ein1, ein2, an_int
    global sig, sig_inc, sig_targ, sig_cyc, dsig
    global eps, eps_inc, eps_targ, eps_cyc, deps
    global chi, alp, dalp, yo

    print("n_int =",n_int)
    an_int = max(1,n_int)
    if mode == 0:
        sig = 0
        eps = 0
        alp  = np.zeros(an_int)
        chi  = np.zeros(an_int)
        ein1 = ",->"
        ein2 = ",->"
    elif mode == 1:
        sig = np.zeros(ndim)
        eps = np.zeros(ndim)
        alp  = np.zeros([an_int,ndim])
        chi  = np.zeros([an_int,ndim])
        ein1 = "ij,jl->il"
        ein2 = "i,i->"
    elif mode == 2:
        sig = np.zeros([ndim,ndim])
        eps = np.zeros([ndim,ndim])
        alp  = np.zeros([an_int,ndim,ndim])
        chi  = np.zeros([an_int,ndim,ndim])
        ein1 = "iIjJ,jJlL->iIlL"
    else:
        print("Mode not recognised:", mode)
        error()

print("")
print("+-----------------------------------------------------------+")
print("|  HyperCheck: checking routine for hyperplasticity models  |")
print("|  (c) G.T. Houlsby, 2019                                   |")
print("+-----------------------------------------------------------+")

hu = importlib.import_module("HyperUtils")

input_file_name = "unknown"
if os.path.isfile("hyperchecklast.dat"):
    input_file = open("hyperchecklast.dat", 'r')
    input_file_name = input_file.readline()
    input_file.close() 

input_found = False
while not input_found:
    model_temp = input("Input model for checking [" + input_file_name + "]: ",)
    if len(model_temp) == 0:
        model_temp = input_file_name
    if os.path.isfile(model_temp + ".py"):
        model = model_temp
        print("Importing hyperplasticity model: ", model)
        hm = importlib.import_module(model)
        #hm.setvals()
        print("Description: ", hm.name)
        input_found = True
    else:
        print("File not found:", model_temp)

last_file = open("hyperchecklast.dat", 'w')
last_file.write(model)
last_file.close()
    
npass = 0
nfail = 0
nmiss = 0
mode  = hm.mode
ndim  = hm.ndim
n_int = hm.n_int
n_y   = hm.n_y

epsi = 0.0005
sigi = 0.001
alpi = 0.00055
chii = 0.0011

modestart_short()

if hasattr(hm, "check_eps"): eps = hm.check_eps
else:
    text = input("Input strain: ",)
    if mode == 0: eps = float(text)
    else: eps = np.array([float(item) for item in re.split(r'[ ,;]',text)])
print("eps =", eps)

if hasattr(hm, "check_sig"): sig = hm.check_sig
else:
    text = input("Input stress: ",)
    if mode == 0: sig = float(text)
    else: sig = np.array([float(item) for item in re.split(r'[ ,;]',text)])
print("sig =", sig)

if hasattr(hm, "check_alp"): alp = hm.check_alp
print("alp =", alp)

if hasattr(hm, "check_chi"): chi = hm.check_chi
print("chi =", chi)
#printderivs()

hu.princon(hm)

input("Hit ENTER to start checks",)

if hasattr(hm, "f") and hasattr(hm, "g"):
    print("Checking consistency of f and g formulations ...\n")

    print("Checking inverse relations...")
    test("Strain:", eps, "-> stress -> strain", eps_g(sig_f(eps,alp),alp))
    test("Stress:", sig, "-> strain -> stress", sig_f(eps_g(sig,alp),alp))

    print("Checking chi from different routes...")
    test("from eps and f:", chi_f(eps,alp), "from g", chi_g(sig_f(eps,alp), alp))
    test("from sig and g:", chi_g(sig,alp), "from f", chi_f(eps_g(sig,alp), alp))

    print("Checking Legendre transform...")
    sigt = sig_f(eps,alp)
    W = np.einsum(ein2,sigt,eps)
    test("from eps: f + (-g):", hm.f(eps,alp)-hm.g(sigt,alp), "sig.eps", W)
    epst = eps_g(sig,alp)
    W = np.einsum(ein2,sig,epst)
    test("from sig: f + (-g):", hm.f(epst,alp)-hm.g(sig,alp), "sig.eps", W)

    print("Checking elastic stiffness and compliance at specified strain...")
    D = d2fdede(eps,alp)
    C = -d2gdsds(sig_f(eps,alp),alp)
    DC = np.einsum(ein1,D,C)
    print("Stiffness matrix D =\n",D)
    print("Compliance matrix C =\n",C)
    test("Product DC =\n",DC,"unit matrix\n", np.diag(np.ones(ndim)))
    print("and at specified stress...")
    C = -d2gdsds(sig,alp)
    D = d2fdede(eps_g(sig,alp),alp)
    CD = np.einsum(ein1,C,D)
    print("Compliance matrix D =\n",C)
    print("Stiffness matrix C =\n",D)
    test("Product CD =\n",CD,"unit matrix\n", np.diag(np.ones(ndim)))

if hasattr(hm, "f"):
    print("Checking derivatives of f...")
    if hasattr(hm, "dfde"):
        num = hu.numdiff_1(mode,ndim,hm.f,eps,alp,epsi)
        test("Numerical dfde =\n", num,"hm.dfde =\n", hm.dfde(eps,alp))
    else: 
        print("dfde not present")
        nmiss += 1
    if hasattr(hm, "dfda"):
        num = hu.numdiff_2(mode,ndim,n_int,hm.f,eps,alp,alpi)
        test("Numerical dfda =\n", num,"hm.dfda =\n", hm.dfda(eps,alp))
    else: 
        print("dfda not present")
        nmiss += 1
    if hasattr(hm, "d2fdede"):
        num = hu.numdiff2_1(mode,ndim,hm.f,eps,alp,epsi)
        test("Numerical d2fdede =\n", num, "hm.d2fdede =\n", hm.d2fdede(eps,alp))
    else: 
        print("d2fdede not present")
        nmiss += 1
    if hasattr(hm, "d2fdeda"):
        num = hu.numdiff2_2(mode,ndim,n_int,hm.f,eps,alp,epsi,alpi)
        test("Numerical d2fdeda =\n", num,"hm.d2fdeda =\n", hm.d2fdeda(eps,alp))
    else: 
        print("d2fdeda not present")
        nmiss += 1
    if hasattr(hm, "d2fdade"):
        num = hu.numdiff2_3(mode,ndim,n_int,hm.f,eps,alp,epsi,alpi)
        test("Numerical d2fdade =\n", num,"hm.d2fdade =\n", hm.d2fdade(eps,alp))
    else: 
        print("d2fdade not present")
        nmiss += 1
    if hasattr(hm, "d2fdada"):
        num = hu.numdiff2_4(mode,ndim,n_int,hm.f,eps,alp,alpi)
        test("Numerical d2fdada =\n", num,"hm.d2fdada =\n", hm.d2fdada(eps,alp))
    else: 
        print("d2fdada not present")
        nmiss += 1

if hasattr(hm, "g"):
    print("Checking derivatives of g...")
    if hasattr(hm, "dgds"):
        num = hu.numdiff_1(mode,ndim,hm.g,sig,alp,sigi)
        test("Numerical dgds =\n", num, "hm.dgds =\n", hm.dgds(sig,alp))
    else: 
        print("dgds not present")
        nmiss += 1
    if hasattr(hm, "dgda"):
        num = hu.numdiff_2(mode,ndim,n_int,hm.g,sig,alp,alpi)
        test("Numerical dgda =\n", num,"hm.dgda =\n", hm.dgda(sig,alp))
    else: 
        print("dgda not present")
        nmiss += 1
    if hasattr(hm, "d2gdsds"):
        num = hu.numdiff2_1(mode,ndim,hm.g,sig,alp,sigi)
        test("Numerical d2gdsds =\n", num,"hm.d2gdsds =\n", hm.d2gdsds(sig,alp))
    else: 
        print("d2gdsds not present")
        nmiss += 1
    if hasattr(hm, "d2gdsda"):
        num = hu.numdiff2_2(mode,ndim,n_int,hm.g,sig,alp,sigi,alpi)
        test("Numerical d2gdsda =\n", num,"hm.d2gdsda =\n", hm.d2gdsda(sig,alp))
    else: 
        print("d2gdsda not present")
        nmiss += 1
    if hasattr(hm, "d2gdads"):
        num = hu.numdiff2_3(mode,ndim,n_int,hm.g,sig,alp,sigi,alpi)
        test("Numerical d2gdads =\n", num, "hm.d2gdads =\n", hm.d2gdads(sig,alp))
    else: 
        print("d2gdads not present")
        nmiss += 1
    if hasattr(hm, "d2gdada"):
        num = hu.numdiff2_4(mode,ndim,n_int,hm.g,sig,alp,alpi)
        test("Numerical d2gdada =\n", num, "hm.d2gdada =\n", hm.d2gdada(sig,alp))
    else: 
        print("d2gdada not present")
        nmiss += 1

if hasattr(hm, "f") and hasattr(hm, "g") and hasattr(hm, "y_f") and hasattr(hm, "y_g"): 
    print("Checking consistency of y from different routines...")
    test("from eps and alp and y_f:", hm.y_f(chi_f(eps,alp),eps,alp), 
          "...from y_g", hm.y_g(chi_f(eps,alp),sig_f(eps,alp),alp))
    test("from sig and alp and y_g:", hm.y_g(chi_g(sig,alp),sig,alp), 
          "...from y_f", hm.y_f(chi_g(sig,alp),eps_g(sig,alp),alp))

if hasattr(hm, "y_f"):
    print("Checking derivatives of y_f...")
    if hasattr(hm, "dydc_f"):
        num = hu.numdiff_3(mode,ndim,n_int,n_y,hm.y_f,chi,eps,alp,chii)
        test("Numerical dydc_f =\n", num, "hm.dydc_f =\n", hm.dydc_f(chi,eps,alp))
    else: 
        print("dydc_f not present")
        nmiss += 1
    if hasattr(hm, "dyde_f"):
        num = hu.numdiff_4(mode,ndim,n_int,n_y,hm.y_f,chi,eps,alp,epsi)
        test("Numerical dyde_f =\n", num, "hm.dyde_f =\n", hm.dyde_f(chi,eps,alp))
    else: 
        print("dyde_f not present")
        nmiss += 1
    if hasattr(hm, "dyda_f"):
        num = hu.numdiff_5(mode,ndim,n_int,n_y,hm.y_f,chi,eps,alp,alpi)
        test("Numerical dyda_f =\n", num, "hm.dyda_f =\n", hm.dyda_f(chi,eps,alp))
    else: 
        print("dyda_f not present")
        nmiss += 1
    
if hasattr(hm, "y_g"):
    print("Checking derivatives of y_g...")
    if hasattr(hm, "dydc_g"):
        num = hu.numdiff_3(mode,ndim,n_int,n_y,hm.y_g,chi,sig,alp,chii)
        test("Numerical dydc_g =\n", num, "hm.dydc_g =\n", hm.dydc_g(chi,sig,alp))
    else: 
        print("dydc_g not present")
        nmiss += 1
    if hasattr(hm, "dyds_g"):
        num = hu.numdiff_4(mode,ndim,n_int,n_y,hm.y_g,chi,sig,alp,sigi)
        test("Numerical dyds_g =\n", num, "hm.dyds_g =\n", hm.dyds_g(chi,sig,alp))
    else: 
        print("dyds_g not present")
        nmiss += 1
    if hasattr(hm, "dyda_g"):
        num = hu.numdiff_5(mode,ndim,n_int,n_y,hm.y_g,chi,sig,alp,alpi)
        test("Numerical dyda_g =\n", num,"hm.dyda_g =\n", hm.dyda_g(chi,sig,alp))
    else: 
        print("dyda_g not present")
        nmiss += 1
    
if hasattr(hm, "f") and hasattr(hm, "g") and hasattr(hm, "w_f") and hasattr(hm, "w_g"): 
    print("Checking consistency of w from different routines...")
    test("from eps and alp and w_f:", hm.w_f(chi_f(eps,alp),eps,alp), 
          "...from w_g", hm.w_g(chi_f(eps,alp),sig_f(eps,alp),alp))
    test("from sig and alp and w_g:", hm.w_g(chi_g(sig,alp),sig,alp), 
          "...from w_f", hm.w_f(chi_g(sig,alp),eps_g(sig,alp),alp))

if hasattr(hm, "w_f"):
    print("Checking derivative of w_f...")
    if hasattr(hm, "dwdc_f"):
        num = hu.numdiff_6(mode,ndim,n_int,hm.w_f,chi,eps,alp,chii)
        test("Numerical dwdc_f =\n", num, "hm.dwdc_f =\n", hm.dwdc_f(chi,eps,alp))
    else: 
        print("dwdc_f not present")
        nmiss += 1
    
if hasattr(hm, "w_g"):
    print("Checking derivative of w_g...")
    if hasattr(hm, "dwdc_g"):
        num = hu.numdiff_6(mode,ndim,n_int,hm.w_g,chi,sig,alp,chii)
        test("Numerical dwdc_g =\n", num, "hm.dwdc_g =\n", hm.dwdc_g(chi,sig,alp))
    else: 
        print("dwdc_g not present")
        nmiss += 1
    
print("Checks complete for:",model+",",npass,"passed,",nfail,"failed,",nmiss,"missing derivatives")
if not hasattr(hm, "f"): print("hm.f not present")
if not hasattr(hm, "g"): print("hm.g not present")
if not hasattr(hm, "y_f"): print("hm.y_f not present")
if not hasattr(hm, "y_g"): print("hm.y_g not present")
if not hasattr(hm, "w_f"): print("hm.w_f not present")
if not hasattr(hm, "w_g"): print("hm.w_g not present")