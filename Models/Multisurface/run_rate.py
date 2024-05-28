from HyperDrive import Commands as H
H.f_form()

def history():
    tinc = 100000.0
    H.strain_inc(tinc, [0.15, 0.0], 200, 20)
    H.stress_targ(tinc, [-0.2, 0.0], 200, 20)
     #*stress_cyc 0.01 1.0 5 200 10

H.model("hnepmk_ser")
H.start()
history()
#H.graph("eps1","sig1","null")

H.colour("r")
H.rate()
H.restart()
history()

H.graph("eps1","sig1","null")
H.end()
