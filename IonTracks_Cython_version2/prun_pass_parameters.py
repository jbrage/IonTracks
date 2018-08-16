import sys, os
import prun_jobs
from main import initialise

parameters  = prun_jobs.alljobs[int(sys.argv[1])]

# s = initialise(w_dir, center_cm, rMax_cm, voltage, simulation_time_s, energy_MeV)
s = initialise(parameters)
print(s)


