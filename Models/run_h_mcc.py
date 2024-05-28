from HyperDrive import Commands as H
H.f_form()
H.model("h_mcc")
#H.check()

for steps in [10, 20, 50]:
    H.colour("b")
    H.acc(0.9)
    H.start()
    H.init_strain([0.0, 0.0])
    H.stress_targ(1.0, [200.0, 0.0], 10, 100)
    H.stress_targ(1.0, [190.0, 0.0],  5, 100)
    H.strain_inc(1.0, [0.0, 0.1],   100, 100)
    H.colour("r")
    H.acc(0.0)
    H.restart()
    H.init_strain([0.0, 0.0])
    H.stress_targ(1.0, [200.0, 0.0], 10, 100)
    H.stress_targ(1.0, [190.0, 0.0],  5, 100)
    H.strain_inc(1.0, [0.0, 0.1], steps,   1)
    H.colour("g")
    H.acc(0.9)
    H.restart()
    H.init_strain([0.0, 0.0])
    H.stress_targ(1.0, [200.0, 0.0], 10, 100)
    H.stress_targ(1.0, [190.0, 0.0],  5, 100)
    H.strain_inc(1.0, [0.0, 0.1], steps,   1)
    H.plotCS()
    H.graph("p","q","gr1_"+str(steps))
    H.graph("eps_s","q","gr2_"+str(steps))
H.end()