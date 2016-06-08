import numpy as np
from distribute_and_evolve_pulsed import distribute_and_evolve

W = 33.9  # eV/ion pair for air
# define the parameters from Kanai (1998)
ion_mobility = 1.65     # cm s^-1 V^-1, averaged for positive and negative ions
ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
alpha = 1.60e-6         # cm^3/s, recombination constant


def calculate_collection_efficiency(all_parameters,SHOW_PLOT,track_LETs,track_radii):

    b,d,theta,electric_field_list111,unit_length, W, ion_mobility,ion_diff,alpha = all_parameters
    dx,dy,dz = [unit_length,unit_length,unit_length]

    collection_efficiency = np.empty(len(electric_field_list))

    for entry,electric_field_entry in enumerate(electric_field_list):

        # time it takes to drag the chare carrier distributions apart
        computation_time = d/(2*ion_mobility*electric_field_entry)

        # roughly how far (in terms of unit lengts) an ion diffuses during the simulation time
        no_x = int(np.sqrt(ion_diff * computation_time)/dx)

        no_z = int(d/dz) #number of elements in the z direction
        no_z_buffer = 15 #length of the electrode-buffer to ensure no ions drift through the array in one time step
        no_z_end = no_z + 4*no_z_buffer #total number of elements in the z direction

        # the Gaussian radius (in terms of unit lengths) added
        length_b = int(b/unit_length)
        no_x += length_b

        no_x *= 10
        no_y = no_x # same number of voxels in the x- and y-directions

        # depending on the cluster/computer, the upper limit may be changed
        if (no_x*no_y*no_z) > 1e8:
            raise ValueError("Too many elements in the array: %i" % (no_x*no_y*no_z))

        parameter_list = [ion_diff, ion_mobility, electric_field_entry, dx, dy, dz, theta, alpha, no_x,no_y,no_z,no_z_buffer,d,W]

        all_results = distribute_and_evolve(track_radii,track_LETs, parameter_list, SHOW_PLOT)

        no_initialised_charge_carriers, no_recombined_charge_carriers = all_results

        collection_efficiency[entry] = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers

    return collection_efficiency


if __name__ == "__main__":

    # define the electric fields for which the collection efficiency should be calculated
    Emin_V_cm = 1000  # lowest electric field
    Emax_V_cm = 2000  # largest electric field
    number_of_Efields = 1  # number of electric fields between min and max
    electric_field_list = np.linspace(Emin_V_cm, Emax_V_cm, number_of_Efields)

    LET_keV_um = 0.1        # linear energy transfer [keV/micrometer]
    LET_eV_cm = LET_keV_um*1e7

    b_cm = 0.002            # Gaussian track radius [cm]
    d_cm = 0.2              # electrode gap [cm]
    theta = 0.              # angle between track and electric field [rad]

    unit_length = 2e-4      # cm, size of every voxel length
    SHOW_PLOT = True
    number_of_tracks = 50

    track_radii = np.ones(number_of_tracks)*b_cm        # cm
    track_LETs = np.ones(number_of_tracks)*LET_eV_cm    # eV/cm

    params = [b_cm, d_cm, theta, electric_field_list, unit_length, W, ion_mobility,ion_diff,alpha]
    f_IonTracks = calculate_collection_efficiency(params,SHOW_PLOT,track_LETs,track_radii)

    print()
    print("#Electric field \tf_IonTracks")
    for i,f in enumerate(f_IonTracks):

        electric_field = int(electric_field_list[i])

        text = "%i [V/cm]\t\t%g" % (electric_field,f)
        print(text)
