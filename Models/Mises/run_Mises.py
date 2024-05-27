from HyperDrive import Commands as H
H.title('von Mises')
H.voigt()
H.g_form()

H.model('Mises')
#H.check()
H.const([150.0, 100.0, 1.0])

H.start()
H.strain_inc(1.0, [0.1, -0.05, -0.05, 0.0, 0.0, 0.0], 100, 10)
H.graph("eps11", "sig11")

H.start()
H.strain_inc(1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.2], 100, 10)
H.strain_inc(1.0, [0.0, 0.0, 0.0, 0.0, 0.2, 0.0], 100, 10)
H.strain_inc(1.0, [0.0, 0.0, 0.0, 0.0, 0.0, -0.2], 100, 10)
H.strain_inc(1.0, [0.0, 0.0, 0.0, 0.0, -0.2, 0.0], 100, 10)
H.strain_inc(1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.2], 100, 10)
H.graph("gam12", "tau12")
H.graph("gam31", "tau31")
H.graph("t", "tau12")
H.graph("t", "tau31")
H.graph("tau12", "tau31")

H.end()
