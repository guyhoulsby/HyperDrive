from HyperDrive import Commands as H
def hist(n):
    for i in range(n):
        H.stress_inc(5.0, [-0.43], 15, 10)
        H.stress_inc(5.0, [ 0.43], 15, 10)
def hist2(n):
    for i in range(n):
        H.stress_inc(5.0, [-0.43], 15, 10)
        H.stress_inc(5.0, [ 0.43], 15, 10)
        
H.title("HABB test run - accelerated")
H.model("habb1")
#H.check()
H.const([85.0, 1.0, 1.0, 3.2, 4000.0, 0.05, 4.5, 5.0, 0.0, 40, 0.1, 1.0])
#H.rate()

H.start()
#H.strain_inc(5.0, [0.25], 60, 25)
H.stress_inc(5.0, [0.51], 60, 25)
#H.pause()

H.high()
hist(1)
H.unhigh()

hist(8)

H.high()
hist(1)
H.unhigh()

H.tweak("Rfac", 8.0)
hist(10)

H.tweak("Rfac", 1.0)
hist(9)

H.high()
hist(1)
H.unhigh()

H.tweak("Rfac", 89.0)
hist(10)

H.tweak("Rfac", 1.0)
hist(9)

H.high()
hist(1)
H.unhigh()

#H.plot()
H.graph("eps", "sig", "habb1_acc", 12.0, 7.0)
#H.pause()

H.title("HABB test run - no acceleration")
H.start()
H.stress_inc(5.0, [0.51], 60, 25)

H.high()
hist2(1)
H.unhigh()

hist2(8)

H.high()
hist2(1)
H.unhigh()

hist2(89)

H.high()
hist2(1)
H.unhigh()

hist2(899)

H.high()
hist2(1)
H.unhigh()

#H.plot()
H.graph("eps", "sig", "habb1_noacc", 12.0, 7.0)
H.end()
