from math import exp, pi, sqrt

import numpy as np
from numba import njit


def continuous_beam_PDEsolver(parameter_dic, extra_params_dic):
    '''
    Define the parameters from Kanai (1998)
    '''
    W = 34.2  # eV/ion pair for air (protons; https://www.sciencedirect.com/science/article/abs/pii/S0969806X05003695)
    ion_mobility = 1.65     # cm^2 s^-1 V^-1, averaged for positive and negative ions
    ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
    alpha = 1.60e-6         # cm^3/s, recombination constant
    
    fluence_cm2_s   = parameter_dic["fluencerate_cm2_s"]    # fluence-rate [/cm^2/s]
    LET_keV_um      = parameter_dic["LET_keV_um"]           # linear energy transfer
    voltage_V       = parameter_dic["voltage_V"]            # [V/cm] magnitude of the electric field
    d_cm            = parameter_dic["electrode_gap_cm"]     # [cm] # electrode gap
    
    track_radius    = extra_params_dic["track_radius_cm"]   # Gaussian radius b [cm]
    SHOW_PLOT         = extra_params_dic["SHOW_PLOT"]         # show frames of the simulation
    seed               = extra_params_dic["seed"]              # ensure the coordinates are sampled anew each run
    np.random.seed(seed)                                                      # set the new seed
    PRINT             = extra_params_dic["PRINT_parameters"]  # print parameters?

    unit_length_cm  = extra_params_dic["unit_length_cm"]    # cm, size of every voxel length

    '''
    Define the grid size parameters
    '''
    LET_eV_cm = LET_keV_um*1e7
    r_cm = 0.012                    # radius of the sampled cylinder
    area_cm2 = r_cm**2*pi           # the "recombination" area
    
    no_xy = int(2*r_cm/unit_length_cm) # no of voxels in xy-directions
    buffer_radius = 10
    no_xy += 2*buffer_radius
    no_z = int(d_cm/unit_length_cm) #number of elements in the z direction
    no_z_electrode = 5 #length of the electrode-buffer to ensure no ions drift through the array in one time step
    no_z_with_buffer = 2*no_z_electrode + no_z
    # depending on the cluster/computer, the upper limit may be changed
    if (no_xy*no_xy*no_z) > 1e8:
        raise ValueError("Too many elements in the array: %i" % (no_xy*no_xy*no_z_with_buffer))

    # find the middle of the arrays
    mid_xy_array = int(no_xy/2.)
    mid_z_array = int(no_z/2.)
    outer_radius = (no_xy/2.)
    inner_radius = outer_radius - buffer_radius

    no_figure_updates = 5              # update the figure this number of times
    number_of_iterations = 1e7      # preallocate track coordinates

    '''
    find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numerical error does not increase but
    decreases and eventually damps out
    '''
    dt = 1.
    von_neumann_expression = False
    Efield_V_cm = voltage_V/d_cm # to ensure the same dt in all simulations
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
    separation_time_steps = int(d_cm/(2.*ion_mobility*Efield_V_cm*dt))
    n_separation_times = 3
    computation_time_steps = separation_time_steps * n_separation_times
    simulation_time_s = computation_time_steps*dt
    number_of_tracks = int(fluence_cm2_s* simulation_time_s*area_cm2)
    if number_of_tracks < 1:
        number_of_tracks = 1 # initial recombination

    # Gaussian track structure
    N0 = LET_eV_cm/W                 # Linear charge carrier density
    b_cm = track_radius              # Gaussian track radius
    Gaussian_factor = N0/(pi*b_cm**2)

    # preallocate arrays
    x_coordinates_ALL = np.random.uniform(0, 1, int(number_of_iterations))*no_xy
    y_coordinates_ALL = np.random.uniform(0, 1, int(number_of_iterations))*no_xy
    positive_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    negative_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    positive_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
    negative_array_temp = np.zeros((no_xy,  no_xy, no_z_with_buffer))
    no_recombined_charge_carriers = 0.0
    no_initialised_charge_carriers = 0.0

    '''
    The following part is only used to plot the figure; consider to remove
    '''
    MINVAL = 0.
    MAXVAL = 0.
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

    time_s = computation_time_steps * dt
    randomized = np.random.random(number_of_tracks)
    summed = np.cumsum(randomized)
    distributed_times = np.asarray(summed)/max(summed) * time_s
    bins = np.arange(0.0, time_s + dt, dt)
    initialise_tracks = np.asarray(np.histogram(distributed_times, bins)[0], dtype=np.int32) #
    # ensure that the simulation also evovles the last initialised track without insert new
    dont_initialise_tracks = np.zeros(separation_time_steps)
    track_times_histrogram = np.asarray(np.concatenate((initialise_tracks, dont_initialise_tracks)), dtype=np.int32)
    computation_time_steps += separation_time_steps

    coordinate_counter, number_of_initialized_tracks = [0] * 2
    f = 0.0

    '''
    Start the simulation by evolving the distribution one step at a time
    '''
    for time_step in range(computation_time_steps):
        # number of track to be inserted randomly during this time step
        tracks_to_be_initialised = track_times_histrogram[time_step]
        coordinate_counter, MAXVAL, no_initialised_charge_carriers = insert_tracks_step(
            tracks_to_be_initialised,
            coordinate_counter,
            x_coordinates_ALL,
            y_coordinates_ALL,
            mid_xy_array,
            inner_radius,
            no_z_electrode,
            no_z,
            no_xy,
            unit_length_cm,
            b_cm,
            Gaussian_factor,
            time_step,
            separation_time_steps,
            positive_array,
            negative_array,
            MAXVAL,
            no_initialised_charge_carriers
        )
        number_of_initialized_tracks += tracks_to_be_initialised

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

        no_recombined_charge_carriers += calculate_new_densites(
            positive_array, negative_array,
            positive_array_temp, negative_array_temp,
            alpha, dt, sx, sy, sz, cx, cy, cz,
            no_xy, no_z_with_buffer, no_z, no_z_electrode, mid_xy_array, inner_radius
        )

        # Zamiana wskaźników tablic (żeby uniknąć kopiowania danych)
        positive_array, positive_array_temp = positive_array_temp, positive_array
        negative_array, negative_array_temp = negative_array_temp, negative_array


    f = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers
    return 1./f

@njit
def calculate_new_densites(positive_array, negative_array, positive_array_temp, negative_array_temp,
                         alpha, dt, sx, sy, sz, cx, cy, cz,
                         no_xy, no_z_with_buffer, no_z, no_z_electrode, mid_xy_array, inner_radius):
    no_recombined = 0.0

    for i in range(1, no_xy - 1):
        for j in range(1, no_xy - 1):
            for k in range(1, no_z_with_buffer - 1):
                pos = positive_array[i, j, k]
                neg = negative_array[i, j, k]

                pos_temp = (
                    (sz + cz * (cz + 1.) / 2.) * positive_array[i, j, k - 1] +
                    (sz + cz * (cz - 1.) / 2.) * positive_array[i, j, k + 1] +
                    (sy + cy * (cy + 1.) / 2.) * positive_array[i, j - 1, k] +
                    (sy + cy * (cy - 1.) / 2.) * positive_array[i, j + 1, k] +
                    (sx + cx * (cx + 1.) / 2.) * positive_array[i - 1, j, k] +
                    (sx + cx * (cx - 1.) / 2.) * positive_array[i + 1, j, k] +
                    (1. - cx * cx - cy * cy - cz * cz - 2. * (sx + sy + sz)) * pos
                )

                neg_temp = (
                    (sz + cz * (cz + 1.) / 2.) * negative_array[i, j, k + 1] +
                    (sz + cz * (cz - 1.) / 2.) * negative_array[i, j, k - 1] +
                    (sy + cy * (cy + 1.) / 2.) * negative_array[i, j + 1, k] +
                    (sy + cy * (cy - 1.) / 2.) * negative_array[i, j - 1, k] +
                    (sx + cx * (cx + 1.) / 2.) * negative_array[i + 1, j, k] +
                    (sx + cx * (cx - 1.) / 2.) * negative_array[i - 1, j, k] +
                    (1. - cx * cx - cy * cy - cz * cz - 2. * (sx + sy + sz)) * neg
                )

                recomb = alpha * pos * neg * dt
                positive_array_temp[i, j, k] = pos_temp - recomb
                negative_array_temp[i, j, k] = neg_temp - recomb

                if no_z_electrode < k < (no_z + no_z_electrode):
                    if ((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) < inner_radius ** 2:
                        no_recombined += recomb

    return no_recombined

@njit
def insert_tracks_step(
    tracks_to_be_initialised,
    coordinate_counter,
    x_coordinates_ALL,
    y_coordinates_ALL,
    mid_xy_array,
    inner_radius,
    no_z_electrode,
    no_z,
    no_xy,
    unit_length_cm,
    b_cm,
    Gaussian_factor,
    time_step,
    separation_time_steps,
    positive_array,
    negative_array,
    MAXVAL,
    no_initialised_charge_carriers
):
    for insert_track in range(tracks_to_be_initialised):
        coordinate_counter += 1
        x = x_coordinates_ALL[coordinate_counter]
        y = y_coordinates_ALL[coordinate_counter]

        while sqrt((x - mid_xy_array) ** 2 + (y - mid_xy_array) ** 2) > inner_radius:
            coordinate_counter += 1
            x = x_coordinates_ALL[coordinate_counter]
            y = y_coordinates_ALL[coordinate_counter]

        for k in range(no_z_electrode, no_z + no_z_electrode):
            for i in range(no_xy):
                for j in range(no_xy):
                    dx = i - x
                    dy = j - y
                    distance_from_center = sqrt(dx * dx + dy * dy) * unit_length_cm
                    ion_density = Gaussian_factor * exp(-distance_from_center ** 2 / b_cm ** 2)

                    positive_array[i, j, k] += ion_density
                    negative_array[i, j, k] += ion_density

                    if positive_array[i, j, k] > MAXVAL:
                        MAXVAL = positive_array[i, j, k]

                    # Tylko w "obszarze fizycznym", nie buforze
                    if ((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) < inner_radius ** 2:
                        if time_step > separation_time_steps:
                            no_initialised_charge_carriers += ion_density

    return coordinate_counter, MAXVAL, no_initialised_charge_carriers
