import sys

import numpy as np
from general_and_initial_recombination import total_recombination_PDEsolver
from functions import E_MeV_to_LET_keV_um, calc_b_cm, doserate_to_fluence

sys.path.append('./cython')


def IonTracks_total_recombination(parameters, PRINT_parameters=False):
    '''
    Calculate the stopping power, fluence-rate, and track radius as a
    function of proton energy and dose-rate.

    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    '''
    voltage_V, energy_MeV, doserate_Gy_min, electrode_gap_cm = parameters

    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    track_radius_cm = calc_b_cm(LET_keV_um)
    fluence_cm2_s = doserate_to_fluence(doserate_Gy_min, energy_MeV)

    SHOW_PLOT = False
    myseed = int(np.random.randint(1, 1e7))
    simulation_parameters = [
                                fluence_cm2_s,
                                LET_keV_um,
                                track_radius_cm,
                                voltage_V,
                                electrode_gap_cm,
                                SHOW_PLOT,
                                myseed,
                                PRINT_parameters
                            ]
    f = total_recombination_PDEsolver(simulation_parameters)
    return 1. / f


if __name__ == "__main__":

    electrode_gap_cm = 0.2
    voltage_V = 200
    energy_MeV = 226
    doserate_Gy_min = 700
    input_parameters = [voltage_V, energy_MeV, doserate_Gy_min, electrode_gap_cm]

    ks_IonTracks = IonTracks_total_recombination(input_parameters)
    print("k_s = {:0.6f}".format(ks_IonTracks))
