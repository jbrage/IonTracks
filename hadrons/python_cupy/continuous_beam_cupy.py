from math import pi

import cupy as cp


def continuous_beam_PDEsolver(parameter_dic, extra_params_dic):
    '''
    Define the parameters from Kanai (1998)
    '''
    W = 34.2  # eV/ion pair for air (protons; https://www.sciencedirect.com/science/article/abs/pii/S0969806X05003695)
    ion_mobility = 1.65  # cm^2 s^-1 V^-1, averaged for positive and negative ions
    ion_diff = 3.7e-2  # cm^2/s, averaged for positive and negative ions
    alpha = 1.60e-6  # cm^3/s, recombination constant

    fluence_cm2_s = parameter_dic["fluencerate_cm2_s"]  # fluence-rate [/cm^2/s]
    LET_keV_um = parameter_dic["LET_keV_um"]  # linear energy transfer
    voltage_V = parameter_dic["voltage_V"]  # [V/cm] magnitude of the electric field
    d_cm = parameter_dic["electrode_gap_cm"]  # [cm] # electrode gap

    track_radius = extra_params_dic["track_radius_cm"]  # Gaussian radius b [cm]
    seed = extra_params_dic["seed"]  # ensure the coordinates are sampled anew each run
    cp.random.seed(seed)  # set the new seed

    unit_length_cm = extra_params_dic["unit_length_cm"]  # cm, size of every voxel length

    '''
    Define the grid size parameters
    '''
    LET_eV_cm = LET_keV_um * 1e7
    r_cm = 0.012  # radius of the sampled cylinder
    area_cm2 = r_cm ** 2 * pi  # the "recombination" area

    no_xy = int(2 * r_cm / unit_length_cm)  # no of voxels in xy-directions
    buffer_radius = 10
    no_xy += 2 * buffer_radius
    no_z = int(d_cm / unit_length_cm)  # number of elements in the z direction
    no_z_electrode = 5  # length of the electrode-buffer to ensure no ions drift through the array in one time step
    no_z_with_buffer = 2 * no_z_electrode + no_z

    # Ensure min dimensions for slicing (at least 3 for 1:-1)
    if no_xy < 3: no_xy = 3
    if no_z_with_buffer < 3: no_z_with_buffer = 3

    # depending on the cluster/computer, the upper limit may be changed
    if (no_xy * no_xy * no_z) > 1e8:
        raise ValueError("Too many elements in the array: %i" % (no_xy * no_xy * no_z_with_buffer))

    # find the middle of the arrays
    mid_xy_array = int(no_xy / 2.)
    mid_z_array = int(no_z / 2.)
    outer_radius = (no_xy / 2.)
    inner_radius = outer_radius - buffer_radius

    number_of_iterations = 1e7  # preallocate track coordinates

    '''
    find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
    decreases and eventually damps out
    '''
    dt = 1.
    von_neumann_expression = False
    Efield_V_cm = voltage_V / d_cm  # to ensure the same dt in all simulations
    while not von_neumann_expression:
        dt /= 1.01
        # as defined in the Deghan (2004) paper
        sx = ion_diff * dt / (unit_length_cm ** 2)
        sy = ion_diff * dt / (unit_length_cm ** 2)
        sz = ion_diff * dt / (unit_length_cm ** 2)
        cx = cy = 0.0
        cz = ion_mobility * Efield_V_cm * dt / unit_length_cm
        # check von Neumann's criterion
        von_neumann_expression = (2 * (
                sx + sy + sz) + cx ** 2 + cy ** 2 + cz ** 2 <= 1 and cx ** 2 * cy ** 2 * cz ** 2 <= 8 * sx * sy * sz)

    # Pre-calculate constants for Lax-Wendroff scheme outside the loop
    szcz_pos = sz + cz * (cz + 1.) / 2.
    szcz_neg = sz + cz * (cz - 1.) / 2.
    sycy_pos = sy + cy * (cy + 1.) / 2.
    sycy_neg = sy + cy * (cy - 1.) / 2.
    sxcx_pos = sx + cx * (cx + 1.) / 2.
    sxcx_neg = sx + cx * (cx - 1.) / 2.
    diffusion_term_coeff = 1. - cx * cx - cy * cy - cz * cz - 2. * (sx + sy + sz)

    '''
    Calculate the number of step required to drag the two charge carrier distributions apart
    along with the number of tracks to be uniformly distributed over the domain
    '''
    separation_time_steps = int(d_cm / (2. * ion_mobility * Efield_V_cm * dt))
    n_separation_times = 3
    computation_time_steps = separation_time_steps * n_separation_times
    simulation_time_s = computation_time_steps * dt
    number_of_tracks = int(fluence_cm2_s * simulation_time_s * area_cm2)
    if number_of_tracks < 1:
        number_of_tracks = 1  # initial recombination

    # Gaussian track structure
    N0 = LET_eV_cm / W  # Linear charge carrier density
    b_cm = track_radius  # Gaussian track radius
    Gaussian_factor = N0 / (pi * b_cm ** 2)

    # preallocate arrays on GPU
    x_coordinates_ALL = cp.random.uniform(0, 1, int(number_of_iterations)) * no_xy
    y_coordinates_ALL = cp.random.uniform(0, 1, int(number_of_iterations)) * no_xy
    positive_array = cp.zeros((no_xy, no_xy, no_z_with_buffer), dtype=cp.float64)
    negative_array = cp.zeros((no_xy, no_xy, no_z_with_buffer), dtype=cp.float64)
    # Temporary arrays for Lax-Wendroff updates. Use zeros_like for convenience.
    positive_array_temp = cp.zeros_like(positive_array)
    negative_array_temp = cp.zeros_like(negative_array)
    no_recombined_charge_carriers = 0.0
    no_initialised_charge_carriers = 0.0

    # Pre-create meshgrid for RDD calculation to avoid recomputing in loop
    Y_coords, X_coords = cp.meshgrid(cp.arange(no_xy), cp.arange(no_xy))

    '''
    Create an array which include the number of track to be inserted for each time step
    The tracks are distributed uniformly in time
    '''

    time_s = computation_time_steps * dt
    randomized = cp.random.random(number_of_tracks)
    summed = cp.cumsum(randomized)
    distributed_times = summed / cp.max(summed).item() * time_s
    bins = cp.arange(0.0, time_s + dt, dt, dtype=cp.float64)
    initialise_tracks = cp.asarray(cp.histogram(distributed_times, bins)[0], dtype=cp.int32)
    # ensure that the simulation also evovles the last initialised track without insert new
    dont_initialise_tracks = cp.zeros(separation_time_steps, dtype=cp.int32)
    track_times_histogram = cp.concatenate((initialise_tracks, dont_initialise_tracks))
    computation_time_steps += separation_time_steps

    coordinate_counter, number_of_initialized_tracks = [0] * 2

    '''
    Start the simulation by evolving the distribution one step at a time
    '''
    for time_step in range(computation_time_steps):
        # number of track to be inserted randomly during this time step
        tracks_to_be_initialised = track_times_histogram[time_step].item()

        if tracks_to_be_initialised > 0:
            for insert_track in range(tracks_to_be_initialised):
                number_of_initialized_tracks += 1

                # sample a coordinate point
                coordinate_counter += 1
                x = x_coordinates_ALL[coordinate_counter]
                y = y_coordinates_ALL[coordinate_counter]

                # check whether the point is inside the circle.
                # it is too time consuming simply to set x=rand(0,1)*no_xy
                while cp.sqrt((x - mid_xy_array) ** 2 + (y - mid_xy_array) ** 2) > inner_radius:
                    coordinate_counter += 1
                    x = x_coordinates_ALL[coordinate_counter]
                    y = y_coordinates_ALL[coordinate_counter]

                # Calculate distances for the entire XY plane relative to the track center
                distance_from_center_grid = cp.sqrt((X_coords - x) ** 2 + (Y_coords - y) ** 2) * unit_length_cm

                # Calculate ion density for the entire XY plane
                ion_density_grid = Gaussian_factor * cp.exp(-distance_from_center_grid ** 2 / b_cm ** 2)

                # Add this density to the relevant z-slice for positive and negative arrays
                positive_array[:, :, no_z_electrode: (no_z + no_z_electrode)] += ion_density_grid[:, :, cp.newaxis]
                negative_array[:, :, no_z_electrode: (no_z + no_z_electrode)] += ion_density_grid[:, :, cp.newaxis]

                if time_step > separation_time_steps:
                    inner_xy_mask = cp.sqrt(
                        (X_coords - mid_xy_array) ** 2 + (Y_coords - mid_xy_array) ** 2) < inner_radius
                    no_initialised_charge_carriers += cp.sum(ion_density_grid[inner_xy_mask]).item() * no_z

        # Calculate the new densities using vectorized operations (Lax-Wendroff scheme)
        # Define slices for neighbors and current points for the inner core of the array
        curr_p = positive_array[1:-1, 1:-1, 1:-1]
        p_k_minus_1 = positive_array[1:-1, 1:-1, :-2]
        p_k_plus_1 = positive_array[1:-1, 1:-1, 2:]
        p_j_minus_1 = positive_array[1:-1, :-2, 1:-1]
        p_j_plus_1 = positive_array[1:-1, 2:, 1:-1]
        p_i_minus_1 = positive_array[:-2, 1:-1, 1:-1]
        p_i_plus_1 = positive_array[2:, 1:-1, 1:-1]

        curr_n = negative_array[1:-1, 1:-1, 1:-1]
        n_k_minus_1 = negative_array[1:-1, 1:-1, :-2]
        n_k_plus_1 = negative_array[1:-1, 1:-1, 2:]
        n_j_minus_1 = negative_array[1:-1, :-2, 1:-1]
        n_j_plus_1 = negative_array[1:-1, 2:, 1:-1]
        n_i_minus_1 = negative_array[:-2, 1:-1, 1:-1]
        n_i_plus_1 = negative_array[2:, 1:-1, 1:-1]

        # Calculate positive_temp_entry for the entire inner volume
        positive_temp_entry_region = (szcz_pos * p_k_minus_1) + \
                                     (szcz_neg * p_k_plus_1) + \
                                     (sycy_pos * p_j_minus_1) + \
                                     (sycy_neg * p_j_plus_1) + \
                                     (sxcx_pos * p_i_minus_1) + \
                                     (sxcx_neg * p_i_plus_1) + \
                                     (diffusion_term_coeff * curr_p)

        # Calculate negative_temp_entry for the entire inner volume
        negative_temp_entry_region = (szcz_pos * n_k_plus_1) + \
                                     (szcz_neg * n_k_minus_1) + \
                                     (sycy_pos * n_j_plus_1) + \
                                     (sycy_neg * n_j_minus_1) + \
                                     (sxcx_pos * n_i_plus_1) + \
                                     (sxcx_neg * n_i_minus_1) + \
                                     (diffusion_term_coeff * curr_n)

        # Calculate recombination for the entire inner volume
        recomb_region = alpha * curr_p * curr_n * dt

        # Update temporary arrays for the inner region
        positive_array_temp[1:-1, 1:-1, 1:-1] = positive_temp_entry_region - recomb_region
        negative_array_temp[1:-1, 1:-1, 1:-1] = negative_temp_entry_region - recomb_region

        # Accumulate recombined charge carriers (vectorized)
        if time_step > separation_time_steps:
            inner_x, inner_y = cp.meshgrid(cp.arange(1, no_xy - 1), cp.arange(1, no_xy - 1))
            distance_sq_from_mid_xy_inner = (inner_x - mid_xy_array) ** 2 + (inner_y - mid_xy_array) ** 2
            xy_mask_recomb = cp.sqrt(distance_sq_from_mid_xy_inner) < inner_radius

            local_k_start_recomb = no_z_electrode - 1
            local_k_end_recomb = (no_z + no_z_electrode) - 1

            full_recomb_mask_3D = cp.zeros_like(recomb_region, dtype=bool)

            if local_k_start_recomb < local_k_end_recomb:
                full_recomb_mask_3D[:, :, local_k_start_recomb:local_k_end_recomb] = xy_mask_recomb[:, :, cp.newaxis]

            no_recombined_charge_carriers += cp.sum(recomb_region[full_recomb_mask_3D]).item()

        # update the positive and negative arrays in place for the inner region
        positive_array[1:-1, 1:-1, 1:-1] = positive_array_temp[1:-1, 1:-1, 1:-1]
        negative_array[1:-1, 1:-1, 1:-1] = negative_array_temp[1:-1, 1:-1, 1:-1]

    f = (no_initialised_charge_carriers - no_recombined_charge_carriers) / no_initialised_charge_carriers
    return 1. / f