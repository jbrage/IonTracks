from __future__ import division
import numpy as np
import numpy.random as rnd
from copy import deepcopy
cimport numpy as np

from libc.math cimport exp, sqrt, M_PI as pi, log, cos, sin
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # turn off checks for zero division


def continuous_beam_PDEsolver(list parameter_list):
    '''
    Define the parameters from Kanai (1998)
    '''
    # cdef double W = 33.9  # eV/ion pair for air
    cdef double ion_mobility = 1.73    # cm s^-1 V^-1, averaged for positive and negative ions
    cdef double ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
    cdef double alpha = 1.60e-6         # cm^3/s, recombination constant

    cdef double electron_density_per_cm3   = parameter_list[0] # fluence-rate [/cm^2/s]
    cdef double voltage_V       = parameter_list[1] # [V/cm] magnitude of the electric field
    cdef double d_cm            = parameter_list[2] # [cm] # electrode gap
    cdef bint SHOW_PLOT         = parameter_list[3] # show frames of the simulation
    cdef bint PRINT             = parameter_list[4] # print parameters?


    '''
    Define the grid size parameters
    '''
    # cdef double LET_eV_cm = LET_keV_um*1e7
    cdef double r_cm = 0.002                 # radius of the sampled cylinder
    cdef double area_cm2 = r_cm**2*pi           # the "recombination" area
    cdef double unit_length_cm = 6e-4           # cm, size of every voxel length
    cdef int no_xy = int(2*r_cm/unit_length_cm) # no of voxels in xy-directions
    cdef int buffer_radius = 5
    no_xy += 2*buffer_radius
    cdef int no_z = int(d_cm/unit_length_cm) #number of elements in the z direction
    cdef int no_z_electrode = 5 #length of the electrode-buffer to ensure no ions drift through the array in one time step
    cdef int no_z_with_buffer = 2*no_z_electrode + no_z
    # depending on the cluster/computer, the upper limit may be changed
    if (no_xy*no_xy*no_z) > 1e8:
        raise ValueError("Too many elements in the array: %i" % (no_xy*no_xy*no_z_with_buffer))

    # find the middle of the arrays
    cdef int mid_xy_array = int(no_xy/2.)
    cdef int mid_z_array = int(no_z/2.)
    cdef double outer_radius = (no_xy/2.)
    cdef double inner_radius = outer_radius - buffer_radius

    cdef int no_figure_updates = 5              # update the figure this number of times
    # cdef double number_of_iterations = 1e2      # preallocate track coordinates

    '''
    find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
    decreases and eventually damps out
    '''
    cdef double dt = 1.
    cdef bint von_neumann_expression = False
    cdef double sx, sy, sz, cx, cy, cz
    cdef double Efield_V_cm = voltage_V/d_cm # to ensure the same dt in all simulations
    while not von_neumann_expression:
        dt /= 1.01
        # as defined in the Deghan (2004) paper
        # sx = ion_diff*dt/(unit_length_cm**2)
        # sy = ion_diff*dt/(unit_length_cm**2)
        sx = sy = 0.0 # uniform charge => no gradient driven diffusion in the xy plane
        sz = ion_diff*dt/(unit_length_cm**2)
        cx = cy = 0.0
        cz = ion_mobility*Efield_V_cm*dt/unit_length_cm
        # check von Neumann's criterion
        von_neumann_expression = (2*(sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1 and cx**2*cy**2*cz**2 <= 8*sx*sy*sz)

    '''
    Calculate the number of step required to drag the two charge carrier distributions apart
    along with the number of tracks to be uniformly distributed over the domain
    '''
    cdef int separation_time_steps = int(d_cm/(2.*ion_mobility*Efield_V_cm*dt))
    cdef int computation_time_steps = separation_time_steps*3
    cdef double collection_time_s = 2*separation_time_steps*dt

    computation_time_steps = separation_time_steps*3
    cdef double simulation_time_s = computation_time_steps*dt
    cdef np.ndarray[DTYPE_t, ndim=3] positive_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] negative_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] positive_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] negative_array_temp = np.zeros((no_xy,  no_xy, no_z_with_buffer))
    cdef long double no_recombined_charge_carriers = 0.0
    cdef long double no_initialised_charge_carriers = 0.0

    '''
    The following part is only used to plot the figure; consider to remove
    '''
    cdef double MINVAL = 0.
    cdef double MAXVAL = 0.
    if SHOW_PLOT:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        plt.close('all')
        fig = plt.figure()
        plt.ion()
        gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        # define locations of subfigures in the grid
        ax1 = plt.subplot2grid((5,3),(0,0),rowspan=4)
        ax2 = plt.subplot2grid((5,3),(0,1),rowspan=4)
        ax3 = plt.subplot2grid((5,3),(0,2),rowspan=4)

        ax1.imshow(np.asarray(positive_array[mid_xy_array,:,:]).transpose(), vmin=MINVAL, vmax=MAXVAL)
        ax2.imshow(positive_array[:,:,mid_xy_array], vmin=MINVAL, vmax=MAXVAL)
        figure3 = ax3.imshow(positive_array[:,:,mid_xy_array], vmin=MINVAL, vmax=MAXVAL)
        # adjust the 3 subfigures
        ax1.set_aspect('equal'); ax1.axis('off')
        ax2.set_aspect('equal'); ax2.axis('off')
        ax3.set_aspect('equal'); ax3.axis('off')
        # fix colorbar
        cbar_ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.03])
        cb = fig.colorbar(figure3, cax=cbar_ax1, orientation='horizontal', label="Charge carrier density [per cm^3]")
        cb.set_clim(vmin=MINVAL, vmax=MAXVAL)

    if PRINT:
        print("Simul. radius = %0.5g cm" % (inner_radius*unit_length_cm))
        print("Simul. area   = %0.5g cm^2" % ((inner_radius*unit_length_cm)**2*pi))
        print("Electric field = %s V/cm" % Efield_V_cm)
        print("Separation time = %3.2E s" % (separation_time_steps*dt))
        print("Simulation time = %3.2E s" % simulation_time_s)
        print("Time steps      = %3.2E" % computation_time_steps)
        print("Time step dt    = %3.2E s" % dt)
        print("Number of voxels = %3.2E" % (no_xy**2 * no_z_with_buffer))
        print("Number of pixels = %d (x = y directions)" % no_xy)
        print("Number of pixels = %d (z direction)" % no_z)

    '''
    Create an array which include the number of track to be inserted for each time step
    The tracks are distributed uniformly in time
    '''
    cdef double time_s = computation_time_steps * dt
    cdef np.ndarray[DTYPE_t, ndim=1] bins = np.arange(0.0, time_s + dt, dt)

    cdef int coordinate_counter = 0, number_of_initialized_tracks = 0, tracks_to_be_initialised
    cdef double x, y, f, distance_from_center, positive_temp_entry, negative_temp_entry, recomb_temp = 0.0
    cdef int i,j,k, time_step

    cdef double step_recombined, step_initialized
    cdef np.ndarray[DTYPE_t, ndim=1] f_step = np.zeros(computation_time_steps)

    cdef int delta_border = 2
    cdef np.ndarray[DTYPE_t, ndim=1] f_steps_list = np.zeros(computation_time_steps)

    cdef double electron_density_per_cm3_s = electron_density_per_cm3 * dt


    '''
    Start the simulation by evolving the distribution one step at a time
    '''
    for time_step in range(computation_time_steps):

        step_recombined = 0.
        step_initialized = 0.

        '''
        Refill the array with the electron density each time step
        '''
        for k in range(no_z_electrode, no_z + no_z_electrode):
            for i in range(delta_border, no_xy - delta_border):
                for j in range(delta_border, no_xy - delta_border):

                    positive_array[i, j, k] += electron_density_per_cm3_s
                    negative_array[i, j, k] += electron_density_per_cm3_s
                    if positive_array[i, j, k] > MAXVAL:
                        MAXVAL = positive_array[i, j, k]
                    no_initialised_charge_carriers += electron_density_per_cm3_s
                    step_initialized += electron_density_per_cm3_s

        # update the figure
        if SHOW_PLOT:
            update_figure_step=int(computation_time_steps/no_figure_updates)
            if time_step % update_figure_step == 0:
                print("Updated figure; %s" % time_step)
                ax1.imshow(np.asarray(negative_array[:,mid_xy_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
                ax2.imshow(np.asarray(positive_array[:,mid_xy_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
                ax3.imshow(np.asarray(positive_array[:,:,mid_z_array],dtype=np.double),vmin=MINVAL,vmax=MAXVAL)
                cb.set_clim(vmin=MINVAL, vmax=MAXVAL)
                plt.pause(1e-3)

        # calculate the new densities and store them in temporary arrays
        for i in range(1,no_xy-1):
            for j in range(1,no_xy-1):
                for k in range(1,no_z_with_buffer-1):
                    # using the Lax-Wendroff scheme
                    positive_temp_entry = (sz+cz*(cz+1.)/2.)*positive_array[i,j,k-1]
                    positive_temp_entry += (sz+cz*(cz-1.)/2.)*positive_array[i,j,k+1]

                    positive_temp_entry += (sy+cy*(cy+1.)/2.)*positive_array[i,j-1,k]
                    positive_temp_entry += (sy+cy*(cy-1.)/2.)*positive_array[i,j+1,k]

                    positive_temp_entry += (sx+cx*(cx+1.)/2.)*positive_array[i-1,j,k]
                    positive_temp_entry += (sx+cx*(cx-1.)/2.)*positive_array[i+1,j,k]

                    positive_temp_entry += (1.- cx*cx - cy*cy - cz*cz - 2.*(sx+sy+sz))*positive_array[i,j,k]

                    # same for the negative charge carriers
                    negative_temp_entry = (sz+cz*(cz+1.)/2.)*negative_array[i,j,k+1]
                    negative_temp_entry += (sz+cz*(cz-1.)/2.)*negative_array[i,j,k-1]

                    negative_temp_entry += (sy+cy*(cy+1.)/2.)*negative_array[i,j+1,k]
                    negative_temp_entry += (sy+cy*(cy-1.)/2.)*negative_array[i,j-1,k]

                    negative_temp_entry += (sx+cx*(cx+1.)/2.)*negative_array[i+1,j,k]
                    negative_temp_entry += (sx+cx*(cx-1.)/2.)*negative_array[i-1,j,k]

                    negative_temp_entry += (1. - cx*cx - cy*cy - cz*cz - 2.*(sx+sy+sz))*negative_array[i,j,k]

                    # the recombination part
                    recomb_temp = alpha*positive_array[i,j,k]*negative_array[i,j,k]*dt
                    positive_array_temp[i,j,k] = positive_temp_entry - recomb_temp
                    negative_array_temp[i,j,k] = negative_temp_entry - recomb_temp

                    if k > no_z_electrode and k < (no_z + no_z_electrode):
                        # sum over the recombination between the virtual electrodes
                        no_recombined_charge_carriers += recomb_temp
                        step_recombined += recomb_temp

        f_steps_list[time_step] = (step_initialized - step_recombined) / step_initialized

        # update the positive and negative arrays
        for i in range(1,no_xy-1):
            for j in range(1,no_xy-1):
                for k in range(1,no_z_with_buffer-1):
                    positive_array[i,j,k] = positive_array_temp[i,j,k]
                    negative_array[i,j,k] = negative_array_temp[i,j,k]

    f = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers
    return f, f_steps_list, dt, separation_time_steps
