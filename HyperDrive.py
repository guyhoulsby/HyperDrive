#HyperDrive - driving routine for hyperplasticity models
import os.path
import HyperCommands as hc

print("")
print("+-----------------------------------------------------------------------------+")
print("| HyperDrive: driving routine for hyperplasticity models                      |")
print("| (c) G.T. Houlsby, 2018, 2019                                                |")
print("|                                                                             |")
print("| \x1b[0;31mThis program is provided in good faith, but with no warranty of correctness\x1b[0m |")
print("+-----------------------------------------------------------------------------+")

input_file_name = "hyper.dat"
if os.path.isfile("hyperdrivelast.dat"):
    input_file = open("hyperdrivelast.dat", 'r')
    input_file_name = input_file.readline()
    input_file.close() 

input_found = False
while not input_found:
    input_file_temp = input("Enter input filename [" + input_file_name + "]: ",)
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

hc.startup()
hc.process(input_file)

print("End of (processing from) file: " + input_file_name)
input_file.close()