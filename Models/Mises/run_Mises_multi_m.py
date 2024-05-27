from HyperDrive import Commands as H
H.title('von Mises')
H.voigt()
H.g_form()

H.model('Mises_multi')

H.start()
H.strain_inc(1.0, [0.025, -0.0125, -0.0125, 0.0, 0.0, 0.0], 100, 20)
H.printrec()
H.graph("eps11", "sig11")
H.pause()

H.start()
H.strain_inc(1.0, [0.0, 0.0, 0.0, 0.05, 0.0, 0.0], 100, 20)
H.printrec()
H.graph("gam23", "tau23")
H.pause()

H.start()
H.v_txl_d(1.0, 0.1, 100, 20)
H.printrec()
H.csv("txl_d")
H.graph("eps11", "sig11")
H.pause()

H.start()
H.v_txl_u(1.0, 0.025, 100, 20)
H.printrec()
H.csv("txl_u")
H.graph("eps11", "sig11")
H.pause()

H.start()
H.v_dss(1.0, 0.05, 100, 40)
H.printrec()
H.csv("dss")
H.graph("gam23", "tau23")
H.end()