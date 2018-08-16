import numpy as np
import sys, os


# Create job script for prun with N tasks
def write_job_script(N):
    job_script = open("job", "w")
    
    job_name = "IonTracks_%d" % (N)
    walltime = "00:30:00"
    processors = "nodes=2:ppn=20"
    command = "prun %d python3 prun_pass_parameters.py \"{}\"" % (N)
    job_string = """#!/bin/bash
#PBS -q workq
#PBS -N %s
#PBS -l walltime=%s
#PBS -l %s
###PBS -o /dev/null
###PBS -e /dev/null
cd $PBS_O_WORKDIR
%s
""" % (job_name, walltime, processors, command)
    
    job_script.write(job_string)
    job_script.close()


step = 50
voltages  = np.arange(50, 200 + step, step)
MUs = [0.08, 0.2, 0.4, 1.0]
# MUs = [1]
energies_MeV = [70, 150, 226]
# energies_MeV = [100]
center_list = np.arange(0, 0.5, 0.02)
w_dir = os.getcwd()
rMax = 0.0015
sim_time = 200e-6
IC_radius_cm = 0.5

par_file = open("prun_jobs.py", "w")
par_file.write("alljobs = [\n")
task=0
for volt in voltages:
    for MU in MUs:
        for energy in energies_MeV:
            for center in center_list:
# w_dir, center_cm, rMax_cm, voltage, simulation_time_s, energy_MeV, IC_radius_cm, MU
                text = "['%s',%g,%g,%g,%g,%g,%g,%g],\n" % (w_dir, center, rMax, volt, sim_time, energy, IC_radius_cm, MU)
                par_file.write(text)
                task+=1
par_file.write("    ]")
par_file.close()

write_job_script(task)

os.system("qsub job")

# os.remove("prun_job_list.py")
# os.remove("job")
