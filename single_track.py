import numpy as np
import sys
sys.path.append('./cython')
from initial_recombination import initial_PDEsolver
from functions import *


def IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm,
                                    PRINT_parameters = False, SHOW_PLOT = False):
    '''
    Calculate the stopping power and track radius as a function of proton energy 
    
    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    '''

    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    track_radius_cm = calc_b_cm(LET_keV_um)

    LET_eV_cm = LET_keV_um*1e7
    simulation_parameters = [
                                LET_eV_cm,
                                track_radius_cm,
                                voltage_V,
                                IC_angle_rad,
                                electrode_gap_cm,
                                SHOW_PLOT,
                                PRINT_parameters
                            ]

    f_IonTracks = initial_PDEsolver(simulation_parameters)
    return 1./f_IonTracks


if __name__ == "__main__":

    electrode_gap_cm = 0.2
    voltage_V = 50
    energy_MeV = 226

    # the energy at 2 cm water depth using Monte Carlo during SKANDION experiments
    energy_MeV = E_MeV_at_reference_depth_cm(energy_MeV)

    # LET in air
    # LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)

    # calculate the collection effciency with IonTracks and the Jaffe theory
    ks_IonTracks = IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm)
    ks_Jaffe = Jaffe_theory(energy_MeV, voltage_V, electrode_gap_cm)

    results = [energy_MeV, LET_keV_um, ks_Jaffe, ks_IonTracks]

    header = "# E [MeV], LET [keV/um],  ks_Jaffe, ks_IonTracks\n"
    text = "     {},\t{:0.5f},   {:0.4f},\t{:0.4f}".format(*results)
    print(header, text)
