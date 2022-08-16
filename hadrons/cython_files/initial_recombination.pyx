from __future__ import division
import numpy as np
import numpy.random as rnd
import time
from copy import deepcopy
cimport numpy as np
from libc.math cimport exp, sqrt, M_PI as pi, log, cos, sin

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True) # turn off checks for zero division


def Geiss_r_max(E_MeV_u, rho_material):
    # https://www.sciencedirect.com/science/article/pii/S1350448710001903
    rho_water = 1.0 # g /cm3
    r_max_cm = 4e-5 * E_MeV_u ** 1.5 * rho_water / rho_material
    return r_max_cm
    

def Geiss_RRD_cm(r_cm, c, a0_cm, r_max_cm):
    # inner core    
    if r_cm < a0_cm:
        return c / (a0_cm * a0_cm)
    # radial fall-off
    elif ((a0_cm <= r_cm) and (r_cm <= r_max_cm)):
        return c / (r_cm * r_cm)
    else:
      return 0  


def single_track_PDEsolver(dict parameter_dic, dict extra_params_dic, debug=False):

    cdef double LET_keV_um      = parameter_dic["LET_keV_um"] # linear energy transfer [keV/um]
    cdef double voltage_V       = parameter_dic["voltage_V"] # [V/cm] magnitude of the electric field
    cdef double theta_rad       = parameter_dic["IC_angle_rad"] # [rad] angle between electric field and the ion track(s)
    cdef double d_cm            = parameter_dic["electrode_gap_cm"] # [cm] # electrode gap
    cdef double E_MeV_u         = parameter_dic["E_MeV_u"] 
    cdef double a0_nm           = parameter_dic["a0_nm"]
    cdef str RDD_model_name     = parameter_dic["RDD_model"] 
    
    cdef double unit_length_cm  = extra_params_dic["unit_length_cm"] 
    cdef double track_radius_cm = extra_params_dic["track_radius_cm"] # Gaussian radius b [cm]
    cdef bint SHOW_PLOT         = extra_params_dic["SHOW_PLOT"] # show frames of the simulation
    cdef bint PRINT             = extra_params_dic["PRINT_parameters"] # print parameters?

    cdef double LET_eV_cm = LET_keV_um*1e7

    # define the parameters from Kanai (1998)
    cdef double W = 33.9                # eV/ion pair for air
    cdef double ion_mobility = 1.65     # cm^2 s^-1 V^-1, averaged for positive and negative ions
    cdef double ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
    cdef double alpha = 1.60e-6         # cm^3/s, recombination constant

    cdef int no_figure_updates = 5
    cdef double number_of_iterations = 1e7
    
    
    cdef int n_track_radii = 6; # scaling factor to determine the grid width
    #cdef double unit_length_cm = 6e-4   # [cm], grid resolution
    
    cdef int no_z_electrode = 4 #length of the electrode-buffer to ensure no ions drift through the array in one time step
    
    # parameters for the Geiss RDD
    cdef double air_density_g_cm3 = 1.225e-3  # dry air
    cdef double water_density_g_cm3 = 1.0
    cdef double density_ratio = water_density_g_cm3 / air_density_g_cm3
    cdef double a0_cm = a0_nm * 1e-7 * density_ratio
    cdef double r_max_cm = Geiss_r_max(E_MeV_u, air_density_g_cm3)    
    cdef double c = LET_eV_cm / (pi * W) * (1 / (1 + 2 * log(r_max_cm / a0_cm)))
    
    # density parameters for the Gaussian structure
    cdef double N0 = LET_eV_cm/W     # Linear charge carrier density
    cdef double Gaussian_factor = N0/(pi*track_radius_cm**2)

    # grid dimension parameters
    cdef int no_x = int(track_radius_cm*n_track_radii/unit_length_cm)
    # print(track_radius_cm*n_track_radii)
    cdef int no_z = int(d_cm/unit_length_cm) #number of elements in the z direction
    cdef int no_z_with_buffer = 2*no_z_electrode + no_z
    # find the middle of the arrays
    cdef int mid_xy_array = int(no_x/2.)
    cdef int mid_z_array = int(no_z/2.)

    # depending on the cluster/computer, the upper limit may be changed
    if (no_x*no_x*no_z) > 1e8:
        raise ValueError("Too many elements in the array: %i" % (no_x*no_x*no_z_with_buffer))

    # preallocate arrays
    cdef np.ndarray[DTYPE_t, ndim=3] positive_array = np.zeros((no_x,no_x,no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] negative_array = np.zeros((no_x,no_x,no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] positive_array_temp = np.zeros((no_x,no_x,no_z_with_buffer))
    cdef np.ndarray[DTYPE_t, ndim=3] negative_array_temp = np.zeros((no_x, no_x,no_z_with_buffer))
    cdef long double no_recombined_charge_carriers = 0.0
    cdef long double no_initialised_charge_carriers = 0.0

    # for the colorbars
    cdef double MINVAL = 0.
    cdef double MAXVAL = Gaussian_factor

    # plot the calculation?
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

    cdef double dt = 1.
    cdef bint von_neumann_expression = False
    cdef double sx, sy, sz, cx, cy, cz
    cdef double Efield = voltage_V/d_cm

    # find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
    # decreases and eventually damps out
    while not von_neumann_expression:
        dt /= 1.01
        # as defined in the Deghan (2004) paper
        sx = ion_diff*dt/(unit_length_cm**2)
        sy, sz = sx, sx

        cx = ion_mobility*Efield*dt/unit_length_cm*sin(theta_rad)
        cy = 0
        cz = ion_mobility*Efield*dt/unit_length_cm*cos(theta_rad)

        # check von Neumann's criterion
        von_neumann_expression = (2*(sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1 and cx**2*cy**2*cz**2 <= 8*sx*sy*sz)

    # calculate the number of step required to drag the two charge carrier distributions apart
    cdef int separation_time_steps = int(d_cm/(2.*ion_mobility*Efield*dt))
    cdef int computation_time_steps = separation_time_steps*2

    if PRINT:
        print("Electric field = %s V/cm" % Efield)
        print("Separation time = %3.2E s" % (separation_time_steps*dt))
        print("Time steps      = %3.2E" % computation_time_steps)
        print("Time step dt    = %3.2E s" % dt)
        print("Number of voxels = %3.2E" % (no_x**2 * no_z_with_buffer))
        print("Number of pixels = %d (x = y directions)" % no_x)
        print("Number of pixels = %d (z direction)" % no_z)

    cdef double distance_from_center, ion_density, positive_temp_entry, negative_temp_entry, recomb_temp = 0.0
    cdef int i, j, k, time_step

    # define the radial dose model to be used
    
    if RDD_model_name == "Gauss":
        def RDD_function(r_cm):
            return Gaussian_factor * exp(-r_cm ** 2 / track_radius_cm ** 2)

    elif RDD_model_name == "Geiss":
        def RDD_function(r_cm):
            return Geiss_RRD_cm(r_cm, c, a0_cm, r_max_cm)
    else:
        print("RDD model {} undefined.".format(RDD_model_name))
        return 0
    


    # initialise the Gaussian distribution in the array
    start_time = time.time()
    for k in range(no_z_electrode, no_z + no_z_electrode):
        for i in range(no_x):
            for j in range(no_x):
                if debug: print(f'\rGaussian distribution loop: k - {k+1: >3}/{no_z + no_z_electrode}, i - {i+1: >3}/{no_x}, j - {j+1: >3}/{no_x} Time: {time.time()-start_time: .2f}', end='')
                distance_from_center_cm = sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) * unit_length_cm
                # ion_density = Gaussian_factor * exp(-distance_from_center ** 2 / track_radius_cm ** 2)
                # ion_density = Geiss_RRD_cm(distance_from_center, c, a0_cm, r_max_cm)
                ion_density = RDD_function(distance_from_center_cm)
                positive_array[i, j, k] += ion_density
                negative_array[i, j, k] += ion_density
                no_initialised_charge_carriers += ion_density

                if positive_array[i, j, k] > MAXVAL:
                   MAXVAL = positive_array[i, j, k]
    print('')

    # start the calculation
    calculation_time = time.time()
    for time_step in range(computation_time_steps):
        if debug: print(f'Calculation loop iteration {time_step+1}/{computation_time_steps}')
        # update the figure?
        if SHOW_PLOT:
            update_figure_step=int(computation_time_steps/no_figure_updates)
            if time_step % update_figure_step == 0:
                ax1.imshow(np.asarray(negative_array[:,mid_xy_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
                ax2.imshow(np.asarray(positive_array[:,mid_xy_array,:],dtype=np.double).transpose(),vmin=MINVAL,vmax=MAXVAL)
                ax3.imshow(np.asarray(positive_array[:,:,mid_z_array],dtype=np.double),vmin=MINVAL,vmax=MAXVAL)
                cb.set_clim(vmin=MINVAL, vmax=MAXVAL)
                plt.pause(1e-3)

        # calculate the new densities and store them in temporary arrays
        start_time = time.time()
        for i in range(1,no_x-1):
            for j in range(1,no_x-1):
                for k in range(1,no_z_with_buffer-1):
                    if debug: print(f'\rDensities calculation loop: i - {i+1: >3}/{no_x-1}, j - {j+1: >3}/{no_x-1}, k - {k+1: >3}/{no_z_with_buffer-1} Time: {time.time()-start_time: .2f}', end='')
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
                    no_recombined_charge_carriers += recomb_temp
        if debug: print('')

        # update the positive and negative arrays
        start_time = time.time()
        for i in range(1,no_x-1):
            for j in range(1,no_x-1):
                for k in range(1,no_z_with_buffer-1):
                    if debug: print(f'\rUpdate array loop:          i - {i+1: >3}/{no_x-1}, j - {j+1: >3}/{no_x-1}, k - {k+1: >3}/{no_z_with_buffer-1} Time: {time.time()-start_time: .2f}', end='')
                    positive_array[i,j,k] = positive_array_temp[i,j,k]
                    negative_array[i,j,k] = negative_array_temp[i,j,k]            
        if debug: print('')

    f = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers

    if debug: print('Calculation loop combined time: ', time.time()-calculation_time)
    return 1./f
    