from __future__ import division
import numpy as np
import numpy.random as rnd
from copy import deepcopy
cimport numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from libc.math cimport exp, sqrt, M_PI as pi, log, cos, sin

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # turn off checks for zero division

def PDEsolver(list parameter_list):

    cdef int number_of_tracks   = parameter_list[0] # number of tracks
    cdef double LET_eV_cm       = parameter_list[1] # linear energy transfer [eV/cm]
    cdef double track_radius    = parameter_list[2] # Gaussian radius b [cm]
    cdef double simulation_time_s = parameter_list[3] # simulation time
    cdef double Efield          = parameter_list[4] # [V/cm] magnitude of the electric field
    cdef double dx              = parameter_list[5] # [cm] distance between two neighbouring voxels
    cdef double theta           = parameter_list[6] # [rad] angle between electric field and the ion track(s)
    cdef double r_max_cm        = parameter_list[7] # [cm] radius of the given spot
    cdef double d               = parameter_list[8] # [cm] # electrode gap
    cdef bint SHOW_PLOT         = parameter_list[9] # show frames of the simulation
    cdef int myseed             = parameter_list[10] # ensure the coordinates are resampled

    rnd.seed(myseed)

    # Grid dimensions
    cdef int no_x = int(2*r_max_cm/dx)  # define the grid to be twice the radius
    no_x += 20 # the outer circle has a radius of rMax + "10" voxels

    cdef int no_z = int(d/dx) #number of elements in the z direction
    cdef int no_z_electrode = 15 #length of the electrode-buffer to ensure no ions drift through the array in one time step
    cdef int no_z_end = no_z + 4*no_z_electrode #total number of elements in the z direction
    # total length, i.e. electrode gap plus the two electrode buffers
    cdef int no_z_with_buffer = 2*no_z_electrode + no_z

    # depending on the cluster/computer, the upper limit may be changed
    if (no_x*no_x*no_z) > 1e8:
        raise ValueError("Too many elements in the array: %i" % (no_x*no_x*no_z))

    # Simulation parameters
    cdef double W = 33.9  # eV/ion pair for air
    # define the parameters from Kanai (1998)
    cdef double ion_mobility = 1.65     # cm s^-1 V^-1, averaged for positive and negative ions
    cdef double ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
    cdef double alpha = 1.60e-6         # cm^3/s, recombination constant

    # number of times the plot is updated
    cdef int no_figure_updates = 5
    cdef double number_of_iterations = 1e7

    # preallocate arrays
    cdef np.ndarray[DTYPE_t, ndim=1] x_coordinates_ALL = rnd.uniform(0,1,int(number_of_iterations))*no_x
    cdef np.ndarray[DTYPE_t, ndim=1] y_coordinates_ALL = rnd.uniform(0,1,int(number_of_iterations))*no_x

    cdef np.ndarray[DTYPE_t, ndim=3] positive_array=np.zeros((no_x,no_x,no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] positive_array_temp=np.zeros((no_x,no_x,no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] negative_array_temp = np.zeros((no_x, no_x,no_z_with_buffer))
    cdef double no_recombined_charge_carriers = 0.
    cdef double no_initialised_charge_carriers = 0.

    # find the middle of the arrays (mainly for the Gaussian distribution and updating the figures)
    cdef int mid_x_array=int(no_x/2.)
    cdef int mid_y_array=mid_x_array
    cdef int mid_z_array=int(no_z/2.)

    cdef int my_counter = 0
    cdef double outer_radius = (no_x/2.)
    cdef double inner_radius = outer_radius-10

    print("Simul. radius = %0.4g cm" % (inner_radius*dx))
    print("Simul. area   = %0.4g cm^2" % ((inner_radius*dx)**2*pi))

    # for the colorbars
    cdef double MINVAL = 0.
    cdef double MAXVAL = 0.

    cdef double N0 = LET_eV_cm/W     # Linear charge carrier density
    cdef double b = track_radius       # Gaussian track radius
    cdef double front_factor = N0/(pi*b**2)

    cdef np.ndarray[DTYPE_t, ndim=3] negative_array=deepcopy(np.asarray(positive_array,dtype='double'))

    # if SHOW_PLOT:
    #     plt.close('all')
    #     fig = plt.figure()
    #     plt.ion()
    #     gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    #
    #     # define locations of subfigures in the grid
    #     ax1 = plt.subplot2grid((5,3),(0,0),rowspan=4)
    #     ax2 = plt.subplot2grid((5,3),(0,1),rowspan=4)
    #     ax3 = plt.subplot2grid((5,3),(0,2),rowspan=4)
    #
    #     ax1.imshow(np.asarray(positive_array[mid_x_array,:,:]).transpose(), vmin=MINVAL, vmax=MAXVAL)
    #     ax2.imshow(positive_array[:,:,mid_y_array], vmin=MINVAL, vmax=MAXVAL)
    #     figure3 = ax3.imshow(positive_array[:,:,mid_x_array], vmin=MINVAL, vmax=MAXVAL)
    #
    #     # adjust the 3 subfigures
    #     ax1.set_aspect('equal'); ax1.axis('off')
    #     ax2.set_aspect('equal'); ax2.axis('off')
    #     ax3.set_aspect('equal'); ax3.axis('off')
    #
    #     # fix colorbar
    #     cbar_ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    #     cb = fig.colorbar(figure3, cax=cbar_ax1, orientation='horizontal', label="Charge carrier density [per cm^3]")
    #     cb.set_clim(vmin=MINVAL, vmax=MAXVAL)

    # find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
    # decreases and eventually damps out
    cdef double dt = 1.
    cdef bint von_neumann_expression = False
    cdef double sx, sy, sz, cx, cy, cz
    cdef int separation_time_steps, computation_time_steps

    while not von_neumann_expression:
        dt /= 1.01
        # as defined in the Deghan (2004) paper
        sx = ion_diff*dt/(dx**2)
        sy = ion_diff*dt/(dx**2)
        sz = ion_diff*dt/(dx**2)

        cx = ion_mobility*Efield*dt/dx*sin(theta)
        cy = 0
        cz = ion_mobility*Efield*dt/dx*cos(theta)

        # check von Neumann's criterion
        von_neumann_expression = (2*(sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1 and cx**2*cy**2*cz**2 <= 8*sx*sy*sz)

    # calculate the number of step required to drag the two charge carrier distributions apart
    separation_time_steps = int(d/(2.*ion_mobility*Efield*dt))
    computation_time_steps = int(simulation_time_s/dt)

    # create an array which include the number of track to be inserted for each time step
    # The tracks are distributed uniformly in time
    cdef double time_s = computation_time_steps * dt
    cdef np.ndarray[DTYPE_t, ndim=1] randomized = rnd.random(number_of_tracks)
    cdef np.ndarray[DTYPE_t, ndim=1] summed = np.cumsum(randomized)
    cdef np.ndarray[DTYPE_t, ndim=1] distributed_times = np.asarray(summed)/max(summed) * time_s
    cdef np.ndarray[DTYPE_t, ndim=1] bins = np.arange(0.0, time_s + dt, dt)
    cdef np.ndarray[int, ndim=1] initialise_tracks = np.asarray(np.histogram(distributed_times, bins)[0], dtype=np.int32) #

    # ensure that the simulation also evovles the last initialised track without insert new
    cdef np.ndarray[DTYPE_t, ndim=1] dont_initialise_tracks = np.zeros(separation_time_steps)
    cdef np.ndarray[int, ndim=1] track_times_histrogram = np.asarray(np.concatenate((initialise_tracks, dont_initialise_tracks)), dtype=np.int32)
    computation_time_steps += separation_time_steps

    # one time step at a time
    cdef int coordinate_counter = 0, number_of_initialized_tracks = 0, tracks_to_be_initialised
    cdef double x, y, distance_from_center, ion_density, positive_temp_entry, negative_temp_entry, recomb_temp
    cdef int i,j,k, time_step

    for time_step in range(computation_time_steps):

        tracks_to_be_initialised = track_times_histrogram[time_step]

        for insert_track in range(tracks_to_be_initialised):
            number_of_initialized_tracks += 1

            # sample a coordinate point
            coordinate_counter += 1
            x = x_coordinates_ALL[coordinate_counter]
            y = y_coordinates_ALL[coordinate_counter]

            # check whether the point is inside the circle.
            # it is too time comsuming simply to set x=rand(0,1)*no_x
            while sqrt((x - mid_x_array) ** 2 + (y - mid_y_array) ** 2) > inner_radius:
                coordinate_counter += 1
                x = x_coordinates_ALL[coordinate_counter]
                y = y_coordinates_ALL[coordinate_counter]

            for k in range(no_z_electrode, no_z + no_z_electrode):
                for i in range(no_x):
                    for j in range(no_x):
                        distance_from_center = sqrt((i - x) ** 2 + (j - y) ** 2) * dx
                        ion_density = front_factor * exp(-distance_from_center ** 2 / b ** 2)
                        positive_array[i, j, k] += ion_density
                        negative_array[i, j, k] += ion_density

                        if positive_array[i, j, k] > MAXVAL:
                            MAXVAL = positive_array[i, j, k]

                        # calculate the recombination only for charge carriers inside the circle
                        if sqrt((i - mid_x_array) ** 2 + (j - mid_y_array) ** 2) < inner_radius:
                            if time_step > separation_time_steps:
                                no_initialised_charge_carriers += ion_density

        # update the figure
        # if SHOW_PLOT:
        #     update_figure_step=int(computation_time_steps/no_figure_updates)
        #     if time_step % update_figure_step == 0:
        #         ax1.imshow(np.asarray(negative_array[:,mid_y_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
        #         ax2.imshow(np.asarray(positive_array[:,mid_y_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
        #         ax3.imshow(np.asarray(positive_array[:,:,mid_z_array],dtype=np.double),vmin=MINVAL,vmax=MAXVAL)
        #         cb.set_clim(vmin=MINVAL, vmax=MAXVAL)
        #         plt.pause(1e-3)

        # calculate the new densities and store them in temporary arrays
        for i in range(1,no_x-1):
            for j in range(1,no_x-1):
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

                    # the recombination inside the "inner_radius" circle
                    if sqrt((i - mid_x_array) ** 2 + (j - mid_y_array) ** 2) < inner_radius:
                        if time_step > separation_time_steps:
                            no_recombined_charge_carriers += recomb_temp

                    # positive array
                    positive_array_temp[i,j,k] = positive_temp_entry - recomb_temp
                    # negative array
                    negative_array_temp[i,j,k] = negative_temp_entry - recomb_temp

        # update the positive and negative arrays
        for i in range(1,no_x-1):
            for j in range(1,no_x-1):
                for k in range(1,no_z_with_buffer-1):
                    positive_array[i,j,k] = positive_array_temp[i,j,k]
                    negative_array[i,j,k] = negative_array_temp[i,j,k]

    return [no_initialised_charge_carriers,no_recombined_charge_carriers]
