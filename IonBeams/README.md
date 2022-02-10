The following is related to the simulation of continuous beams only.

## Stability analysis

The current scheme to solve the partial differential equations (PDEs) involves a grid with parameters
- ```unit_length_cm``` defines the grid voxel size. The smaller, the finer resolution and longer computation time
- ```radius_cm``` defines the radius of the simulation cylinder where the ion tracks are sampled. The longer the radius, the more tracks are sampled, and the longer the compuation time

The simualtion converges only when 
- a sufficient large simulation radius is chosen
- a sufficient small voxel size is used  

A previous version used ```radius_cm = 0.006``` appears to be too small based on the results below, where the recombination correction factor was simulated for different dose-rates and electrode gaps:

![Underestimation](figures/convergence_plot.png)

Plotting the underestimation of the correction factor *k_s* as a function of the simulation track radius shows that the previous choice of ```radius_cm = 0.006``` was too small. A new default of twice the value ```radius_cm = 0.012``` is chosen, which however increases the simulation time by a factor of 4. The *ID* numbers in the figure legend refers to the *ID* numbers in the subtitles in the figure above.

![Underestimation](figures/Underestimation_plot.png)




## Number of tracks in a simulation
Currently, the simulation of a continuous beam is divided into two parts defined by the time it takes to collect the charges of a given ion track. The first time is regarded as a build-up time where no recombination is scored. The second part mimics is steady-state case where the recombination is scored and the recombination correction factor is calculated.

Specifically, given a time step ```dt```, it takes  
```
cdef int separation_time_steps = int(d_cm/(2.*ion_mobility*Efield_V_cm*dt))  
```  
to move a charge from one electrode to the opposite, where 
- ```d_cm``` denotes the eletrode gap (cm)
- ```ion_mobility``` (cm^2/V/s) is the mobility of ions and 
- ```Efield_V_cm``` (V/cm) is the electric field

The total simulation time is per default given as 
```
n_separation_times = 5
cdef int computation_time_steps = separation_time_steps * n_separation_times
```
and consequently a simulation time
```
cdef double simulation_time_s = computation_time_steps*dt
```
The number of tracks to be initialised is computed from the fluence-rate ```fluence_cm2_s``` and simulation radius area ```area_cm2``` as
```
cdef int number_of_tracks = int(fluence_cm2_s* simulation_time_s*area_cm2)
```
As a consequence, varying the electrode gap ```d_cm``` or electric field strength ```Efield_V_cm``` will change the time it takes to collect the charges ```separation_time_steps``` and eventually the simulation time and number of tracks inserted in the array.


### Charge carrier build-up
The first ```separation_time_steps``` are regarded as a build-up time where no recombnation is counted, see
```
if time_step > separation_time_steps:
    if k > no_z_electrode and k < (no_z + no_z_electrode):
        if sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) < inner_radius:
            no_recombined_charge_carriers += recomb_temp
```
In this particular simulation (```computation_time_steps = separation_time_steps * 5```), the recombination would be counted over the remaining  ```computation_time_steps = separation_time_steps * (5 - 1)``` time steps as the first ```separation_time_steps*1``` only serve as a build-up.

Note, that this approach may not necessarily be correct. However, if the recombination would be counted for all time steps (as in a real ionization chamber), then choosing a large number of charge separation times ```n``` would remedy the effect of the build-up.

