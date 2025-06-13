from __future__ import division

from copy import deepcopy

import numpy as np
import numpy.random as rnd

cimport numpy as np
from libc.math cimport M_PI as pi
from libc.math cimport cos, exp, log, sin, sqrt

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cimport cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # turn off checks for zero division

def continuous_beam_PDEsolver(dict parameter_dic, dict extra_params_dic):
    '''
    Define the parameters from Kanai (1998)
    '''
    cdef double W = 34.2  # eV/ion pair for air (protons; https://www.sciencedirect.com/science/article/abs/pii/S0969806X05003695)
    cdef double ion_mobility = 1.65     # cm^2 s^-1 V^-1, averaged for positive and negative ions
    cdef double ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
    cdef double alpha = 1.60e-6         # cm^3/s, recombination constant

    cdef double fluence_cm2_s   = parameter_dic["fluencerate_cm2_s"]    # fluence-rate [/cm^2/s]
    cdef double LET_keV_um      = parameter_dic["LET_keV_um"]           # linear energy transfer
    cdef double voltage_V       = parameter_dic["voltage_V"]            # [V/cm] magnitude of the electric field
    cdef double d_cm            = parameter_dic["electrode_gap_cm"]     # [cm] # electrode gap
    
    cdef double track_radius    = extra_params_dic["track_radius_cm"]   # Gaussian radius b [cm]
    cdef bint SHOW_PLOT         = extra_params_dic["SHOW_PLOT"]         # show frames of the simulation
    cdef int seed               = extra_params_dic["seed"]
    cdef object rng             = rnd.default_rng(seed)  # Use Generator for reproducibility
    cdef bint PRINT             = extra_params_dic["PRINT_parameters"]  # print parameters?

    cdef double unit_length_cm  = extra_params_dic["unit_length_cm"]    # cm, size of every voxel length

    '''
    Define the grid size parameters
    '''
    cdef double LET_eV_cm = LET_keV_um*1e7
    cdef double r_cm = 0.012                    # radius of the sampled cylinder
    cdef double area_cm2 = r_cm**2*pi           # the "recombination" area
    
    cdef int no_xy = int(2*r_cm/unit_length_cm) # no of voxels in xy-directions
    cdef int buffer_radius = 10
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
    cdef double number_of_iterations = 1e7      # preallocate track coordinates

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
        sx = ion_diff*dt/(unit_length_cm**2)
        sy = ion_diff*dt/(unit_length_cm**2)
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
    cdef int n_separation_times = 3
    cdef int computation_time_steps = separation_time_steps * n_separation_times
    cdef double simulation_time_s = computation_time_steps*dt
    cdef int number_of_tracks = int(fluence_cm2_s* simulation_time_s*area_cm2)
    if number_of_tracks < 1:
        number_of_tracks = 1 # initial recombination

    # Gaussian track structure
    cdef double N0 = LET_eV_cm/W                 # Linear charge carrier density
    cdef double b_cm = track_radius              # Gaussian track radius
    cdef double Gaussian_factor = N0/(pi*b_cm**2)

    # preallocate arrays
    cdef np.ndarray[DTYPE_t, ndim=1] x_coordinates_ALL = rnd.uniform(0, 1, int(number_of_iterations))*no_xy
    cdef np.ndarray[DTYPE_t, ndim=1] y_coordinates_ALL = rnd.uniform(0, 1, int(number_of_iterations))*no_xy
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
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

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
    cdef np.ndarray[DTYPE_t, ndim=1] randomized = rng.random(number_of_tracks)
    cdef np.ndarray[DTYPE_t, ndim=1] summed = np.cumsum(randomized)
    cdef np.ndarray[DTYPE_t, ndim=1] distributed_times = np.asarray(summed)/max(summed) * time_s
    cdef np.ndarray[DTYPE_t, ndim=1] bins = np.arange(0.0, time_s + dt, dt)
    cdef np.ndarray[int, ndim=1] initialise_tracks = np.asarray(np.histogram(distributed_times, bins)[0], dtype=np.int32) #
    # ensure that the simulation also evovles the last initialised track without insert new
    cdef np.ndarray[DTYPE_t, ndim=1] dont_initialise_tracks = np.zeros(separation_time_steps)
    cdef np.ndarray[int, ndim=1] track_times_histrogram = np.asarray(np.concatenate((initialise_tracks, dont_initialise_tracks)), dtype=np.int32)
    computation_time_steps += separation_time_steps

    cdef int coordinate_counter = 0, number_of_initialized_tracks = 0, tracks_to_be_initialised
    cdef double x, y, f, distance_from_center, ion_density, positive_temp_entry, negative_temp_entry, recomb_temp = 0.0
    cdef int i,j,k, time_step

    '''
    Start the simulation by evovling the distribution one step at a time
    '''
    for time_step in range(computation_time_steps):

        # number of track to be inserted randomly during this time step
        tracks_to_be_initialised = track_times_histrogram[time_step]
        for insert_track in range(tracks_to_be_initialised):
            number_of_initialized_tracks += 1

            # sample a coordinate point
            coordinate_counter += 1
            x = x_coordinates_ALL[coordinate_counter]
            y = y_coordinates_ALL[coordinate_counter]

            # check whether the point is inside the circle.
            # it is too time comsuming simply to set x=rand(0,1)*no_xy
            while sqrt((x - mid_xy_array) ** 2 + (y - mid_xy_array) ** 2) > inner_radius:
                coordinate_counter += 1
                x = x_coordinates_ALL[coordinate_counter]
                y = y_coordinates_ALL[coordinate_counter]

            for k in range(no_z_electrode, no_z + no_z_electrode):
                for i in range(no_xy):
                    for j in range(no_xy):
                        distance_from_center = sqrt((i - x) ** 2 + (j - y) ** 2) * unit_length_cm
                        ion_density = Gaussian_factor * exp(-distance_from_center ** 2 / b_cm ** 2)
                        positive_array[i, j, k] += ion_density
                        negative_array[i, j, k] += ion_density
                        if positive_array[i, j, k] > MAXVAL:
                           MAXVAL = positive_array[i, j, k]
                        # calculate the recombination only for charge carriers inside the circle
                        if sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) < inner_radius:
                            if time_step > separation_time_steps:
                                no_initialised_charge_carriers += ion_density

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

                    if time_step > separation_time_steps:
                        if k > no_z_electrode and k < (no_z + no_z_electrode):
                            if sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) < inner_radius:

                                no_recombined_charge_carriers += recomb_temp

        # update the positive and negative arrays
        for i in range(1,no_xy-1):
            for j in range(1,no_xy-1):
                for k in range(1,no_z_with_buffer-1):
                    positive_array[i,j,k] = positive_array_temp[i,j,k]
                    negative_array[i,j,k] = negative_array_temp[i,j,k]

    f = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers
    return 1./f
