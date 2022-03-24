import sys; sys.path.append('./cython')
import numpy as np
from general_and_initial_recombination import total_recombination_PDEsolver
from functions import E_MeV_u_to_LET_keV_um, calc_b_cm, doserate_to_fluence
from math import erf, pi, exp, sqrt
import pandas as pd
from random import random


def IonTracks_total_recombination(parameters, PRINT_parameters=False):
    '''
    Calculate the stopping power, fluence-rate, and track radius as a
    function of proton energy and dose-rate.

    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    '''
    voltage_V, energy_MeV, doserate_Gy_min, electrode_gap_cm = parameters

    LET_keV_um = E_MeV_u_to_LET_keV_um(energy_MeV, particle="proton")
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


def gaussian_density(r, sigma=3.3, n_sigma=2):
    r_max = n_sigma * sigma
    C = lambda x: 1 / (sqrt(2*pi) * sigma) * 2/(erf(r_max / (sqrt(2) * sigma))) * exp(- x**2 /(2 * sigma ** 2))
    # C = lambda x: 1 / (2*pi * sigma**2) * exp(- x**2 /(2 * sigma ** 2)) # 1 = 2*pi* int_0^\infty C(r) r dr

    if isinstance(r, int) or isinstance(r, float):
        return C(r)
    else:
        return [C(i) for i in r]


if __name__ == "__main__":

#    doserate_Gy_s = 10
    electrode_gap_cm = 0.5
    electrode_gap_cm = 0.9
    beam_sigma_cm = 3.3

    n_sigma = 2
    voltage_V = 2000
    energy_MeV = 250

    doserates = [10, 100, 500, 1000, 2000, 4000]
    doserates = [3000, 2000]
#    doserates = [33, 330]
#    doserates = [10, 100]
    doserates = [1]

    fname = "data/full_beam_d{:0.0f}mm_{:0.0f}.csv".format(electrode_gap_cm*10, random()*100000 )

    print("\n\t", fname)
    print("\t", doserates)
    print()

    df = pd.DataFrame()
    for doserate_Gy_s in doserates:

        for r in np.linspace(0, n_sigma *beam_sigma_cm, 5):
 
            density_r = gaussian_density(r, beam_sigma_cm, n_sigma)
            doserate_Gy_min = doserate_Gy_s * 60 * density_r

            input_parameters = [voltage_V, energy_MeV, doserate_Gy_min, electrode_gap_cm]


            ks_IonTracks = IonTracks_total_recombination(input_parameters)
            #print("k_s = {:0.6f}".format(ks_IonTracks))

            row = {"r_cm": r, "density_r": density_r, "doserate_Gy_s": doserate_Gy_s,
                   "ks": ks_IonTracks, "d_cm": electrode_gap_cm, "sigma_cm": beam_sigma_cm,
                    }
            print(row)
            df = df.append(row, ignore_index=True)
        df.to_csv(fname, index=False)
               

