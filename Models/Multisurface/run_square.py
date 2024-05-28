from HyperDrive import Commands as H
H.title('Hyperplasticity test run - square strain path')
H.g_form()
def hist():
    H.strain_inc(0.01, [ 0.1,  0.0], 100, 10)
    H.strain_inc(0.01, [ 0.0,  0.1], 100, 10)
    H.strain_inc(0.02, [-0.2,  0.0], 200, 10)
    H.strain_inc(0.02, [ 0.0, -0.2], 200, 10)
    H.strain_inc(0.02, [ 0.2,  0.0], 200, 10)
    H.strain_inc(0.01, [ 0.0,  0.1], 100, 10)
    H.strain_inc(0.01, [-0.1,  0.0], 100, 10)

H.model('hnepmk_ser')
H.colour('r')
H.start()
hist()

H.model('hnepmk_par')
H.colour('g')
H.restart()
hist()

H.model('hnepmk_nest')
H.colour('g')
H.restart()
hist()

H.plot('null')
H.end()
