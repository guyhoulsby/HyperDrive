from HyperDrive import Commands as H
H.title('Hyperplasticity test run - spiral stress path')
H.f_form()

H.model('hnepmk_ser')
H.colour('r')
H.start()
H.stress_path('spiralpath',20)

H.model('hnepmk_par')
H.colour('g')
H.restart()
H.stress_path('spiralpath',20)

H.model('hnepmk_nest')
H.colour('b')
H.restart()
H.stress_path('spiralpath',20)

H.plot('null')
H.pause()

H.model('hnepmk_ser_b')
H.colour('r')
H.start()
H.stress_path('spiralpath',20)

H.model('hnepmk_par_b')
H.colour('g')
H.restart()
H.stress_path('spiralpath',20)

H.model('hnepmk_nest_b')
H.colour('b')
H.restart()
H.stress_path('spiralpath',20)

#H.plot('null')
#H.pause()

#H.model('hnepmk_ser_h')
#H.colour('r')
#H.start()
#H.stress_path('spiralpath',20)

#H.model('hnepmk_par_h')
#H.colour('g')
#H.restart()
#H.stress_path('spiralpath',20)

#H.model('hnepmk_nest_h')
#H.colour('b')
#H.restart()
#H.stress_path('spiralpath',20)

#H.plot('null')
#H.pause()

#H.model('hnepmk_ser_cbh')
#H.colour('r')
#H.start()
#H.stress_path('spiralpath',20)

#H.model('hnepmk_par_cbh')
#H.colour('g')
#H.restart()
#H.stress_path('spiralpath',20)

#H.model('hnepmk_nest_cbh')
#H.colour('b')
#H.restart()
#H.stress_path('spiralpath',20)

H.plot('null')
H.end()
