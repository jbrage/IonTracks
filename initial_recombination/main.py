import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from distr_ion_tracks import distribute_and_evolve
from scipy.special import expi
import matplotlib.pyplot as plt
from math import pi, exp, sin, log, sqrt
import mpmath
mpmath.mp.dps = 50 # number of figures for computing exponential integrals

W = 33.9  # eV/ion pair for air
# define the parameters from Kanai (1998)
ion_mobility = 1.65     # cm s^-1 V^-1, averaged for positive and negative ions
ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
alpha = 1.60e-6         # cm^3/s, recombination constant


def Jaffe_theory(d_cm,LET_eV_cm,b_cm,theta,electric_field):

    N0 = LET_eV_cm / W
    g = alpha * N0 / (8. * pi * ion_diff)

    # for the case of an ion track parallel to the electric
    if theta > 0:
        x = (b_cm * ion_mobility * electric_field * sin(theta) / (2 * ion_diff)) ** 2
        def nasty_function(y):
            order = 0.
            if y < 1e3:
                # exp() overflows for larger y
                value = exp(y) * (1j * pi / 2) * ss.hankel1(order, 1j * y)
            else:
                # approximation from Zankowski and Podgorsak (1998)
                value = sqrt(2./(pi*y))
            return value
        f = 1. / (1 + g * nasty_function(x))

    # the ion track rotated an angle theta to the electric field
    elif theta==0.:
        factor = mpmath.exp(-1.0/g)*ion_mobility*b_cm**2*electric_field/(2.*g*d_cm*ion_diff)
        first_term = mpmath.ei(1.0/g + log(1.0 + (2.0*d_cm*ion_diff/(ion_mobility*b_cm**2*electric_field))))
        second_term = mpmath.ei(1.0/g)

        f = factor*(first_term - second_term)
    else:
        raise ValueError("Is theta between 0 and pi/2?")
    return f


def calculate_collection_efficiency(all_parameters,SHOW_PLOT):

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

        no_x *= 5
        no_y = no_x # same number of voxels in the x- and y-directions

        # depending on the cluster/computer, the upper limit may be changed
        if (no_x*no_y*no_z) > 1e8:
            raise ValueError("Too many elements in the array: %i" % (no_x*no_y*no_z))

        parameter_list = [ion_diff, ion_mobility, electric_field_entry, dx, dy, dz, theta, alpha, no_x,no_y,no_z,no_z_buffer,d,W]

        track_radii=np.array([50*1e-4,50*1e-4]) #cm
        track_LETs=np.array([1.02,1.02])*1e7 #eV/cm

        track_radii = np.array([0.002,0.002])  # cm
        track_LETs = np.array([0.1,0.1])*1e7   # eV/cm

        all_results = distribute_and_evolve(track_radii,track_LETs, parameter_list, SHOW_PLOT)

        no_initialised_charge_carriers, no_recombined_charge_carriers = all_results

        collection_efficiency[entry] = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers

    return collection_efficiency


if __name__ == "__main__":

    # define the electric fields for which the collection efficiency should be calculated
    Emin_V_cm = 1000  # lowest electric field
    Emax_V_cm = 2000  # largest electric field
    number_of_Efields = 2  # number of electric fields between min and max
    electric_field_list = np.linspace(Emin_V_cm, Emax_V_cm, number_of_Efields)

    LET_keV_um = 0.1        # linear energy transfer [keV/micrometer]
    LET_eV_cm = LET_keV_um*1e7

    b_cm = 0.002            # Gaussian track radius [cm]
    
    d_cm = 0.2              # electrode gap [cm]
    theta = 0.              # angle between track and electric field [rad]
    
    unit_length = 5e-4      # cm, size of every voxel length
    SHOW_PLOT = True

    params = [b_cm, d_cm, theta, electric_field_list, unit_length, W, ion_mobility,ion_diff,alpha]
    f_IonTracks = calculate_collection_efficiency(params,SHOW_PLOT)

    print()
    print("#Electric field \tf_IonTracks\tf_Jaffe")
    for i,f in enumerate(f_IonTracks):

        electric_field = int(electric_field_list[i])
        f_jaffe = Jaffe_theory(d_cm, LET_eV_cm, b_cm, theta, electric_field)

        text = "%i [V/cm]\t\t%g\t%g" % (electric_field,f,f_jaffe)
        print(text)
