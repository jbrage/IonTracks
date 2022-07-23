from initial_recombination import single_track_PDEsolver
from continuous_beam import continuous_beam_PDEsolver
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import hankel1
from math import pi, exp, sin, log, sqrt
import numpy as np
import mpmath
import sys
from pathlib import Path

sys.path.append('./cython_files')

ABS_PATH = str(Path(__file__).parent.absolute())

mpmath.mp.dps = 50  # number of figures for computing exponential integrals
# general parameters
ion_mobility = 1.65     # TODO: units
W = 33.9                # eV/ion pair for air
# define the parameters from Kanai (1998)
ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
alpha = 1.60e-6         # cm^3/s, recombination constant


def Jaffe_theory(x, voltage_V, electrode_gap_cm, input_is_LET=True, particle="proton", IC_angle_rad=0., **rest):
    '''
    The Jaffe theory for initial recombination. Returns the inverse
    collection efficiency, i.e. the recombination correction factor
    '''

    # provide either x as LET (keV/um) or enery (MeV/u), default is input_is_LET=True
    if not input_is_LET:
        LET_keV_um = E_MeV_u_to_LET_keV_um(x, particle=particle)

    LET_eV_cm = LET_keV_um * 1e7
    electric_field = voltage_V/electrode_gap_cm

    # estimate the Gaussian track radius for the given LET
    b_cm = calc_b_cm(LET_keV_um)

    N0 = LET_eV_cm / W
    g = alpha * N0 / (8. * pi * ion_diff)

    # ion track inclined with respect to the electric field?
    if abs(IC_angle_rad) > 0:
        x = (b_cm * ion_mobility * electric_field *
             sin(IC_angle_rad) / (2 * ion_diff)) ** 2

        def nasty_function(y):
            order = 0.
            if y < 1e3:
                # exp() overflows for larger y
                value = exp(y) * (1j * pi / 2) * hankel1(order, 1j * y)
            else:
                # approximation from Zankowski and Podgorsak (1998)
                value = sqrt(2./(pi*y))
            return value
        f = 1. / (1 + g * nasty_function(x)).real

    else:
        '''
        Pretty ugly function splitted up in three parts using mpmath package for precision
        '''
        factor = mpmath.exp(-1.0/g)*ion_mobility*b_cm**2 * \
            electric_field/(2.*g*electrode_gap_cm*ion_diff)
        first_term = mpmath.ei(
            1.0/g + log(1.0 + (2.0*electrode_gap_cm*ion_diff/(ion_mobility*b_cm**2*electric_field))))
        second_term = mpmath.ei(1.0/g)
        f = factor*(first_term - second_term)

    result_dic = {"particle": particle, "LET_keV_um": LET_keV_um,
                  "voltage_V": voltage_V, "electrode_gap_cm": electrode_gap_cm,
                  "IC_angle_rad": IC_angle_rad, "ks_Jaffe": np.float(1/f)}

    if not input_is_LET:
        result_dic["E_MeV_u"] = x

    return pd.DataFrame([result_dic])


def E_MeV_u_to_LET_keV_um(E_MeV_u, particle="proton", material="dry_air"):
    '''
    Calculate the stopping power in dry air or water using PSTAR data
    '''

    folder_name = ABS_PATH + "/data_LET/"
    if material == "dry_air":
        fname = folder_name + "stopping_power_air.csv"
    elif material == "water":
        fname = folder_name + "stopping_power_water.csv"
    else:
        print("Material {} not supported".format(material))
        return 0

    # load the data frame
    df = pd.read_csv(fname, skiprows=3)

    # LET data for the chosen particle
    particle_col_name = "{}_LET_keV_um".format(particle)

    # check if the particle LET was included
    if not particle_col_name in df.columns:
        print("Particle {} is not supported".format(particle))
        return 0

    # energy column
    E_col_name = "E_MeV_u"

    # interpolate the data
    interpolate_LET = interp1d(df[E_col_name], df[particle_col_name])

    if isinstance(E_MeV_u, (list, tuple, np.ndarray)):
        LET_keV_um = [interpolate_LET(i) for i in E_MeV_u]
    else:
        LET_keV_um = interpolate_LET(E_MeV_u)
    return LET_keV_um


def doserate_to_fluence(dose_Gy_min, energy_MeV_u, particle="proton"):
    '''
    Convert the dose-rate to a fluence-rate in air for the given proton energy
    '''

    air_density_kg_m3 = 1.225
    joule_to_keV = 6.241E+15

    # convert
    dose_Gy_s = dose_Gy_min / 60.0
    density_kg_cm3 = air_density_kg_m3 * 1e-6

    # get the LET
    LET_keV_um = E_MeV_u_to_LET_keV_um(energy_MeV_u, particle)
    LET_keV_cm = LET_keV_um * 1e4
    fluence_cm2_s = dose_Gy_s * joule_to_keV * density_kg_cm3 / LET_keV_cm
    return fluence_cm2_s


def calc_b_cm(LET_keV_um):
    '''
    Calculate the Gaussian track radius as suggested by Rossomme et al.
    Returns the track radius in cm given a LET [keV/um]
    '''
    data = np.genfromtxt(f"{ABS_PATH}/data_LET/LET_b.dat", delimiter=",", dtype=float)
    scale = 1e-3
    LET = data[:, 0] * scale
    b = data[:, 1]
    logLET = np.log10(LET)
    z = np.polyfit(logLET, b, 2)
    p = np.poly1d(z)

    b_cm = p(np.log10(LET_keV_um)) * 1e-3
    threshold = 2e-3
    if b_cm < threshold:
        b_cm = threshold
    # b_cm = 1.05*1e-3 # cm   ... Kanai, avoids edges
    return b_cm


def ks_initial_IonTracks(E_MeV_u=200,
                         voltage_V=200,
                         electrode_gap_cm=0.2,
                         particle="proton",
                         RDD_model="Gauss",
                         grid_size_um=3.0,
                         a0_nm=8.0,
                         use_beta=False,
                         PRINT_parameters=False,
                         SHOW_PLOT=False,
                         theta_rad=0,
                         **rest):

    # requires the installation of the libamtrack package
    if use_beta:
        import pyamtrack.libAT as libam
        # scale the inner track structure core by "beta = v/c"
        beta = libam.AT_beta_from_E_single(E_MeV_u)
        a0_nm *= beta

    LET_keV_um = E_MeV_u_to_LET_keV_um(E_MeV_u, particle)
    track_radius_cm = calc_b_cm(LET_keV_um)

    result_dic = {"E_MeV_u": E_MeV_u,
                  "voltage_V": voltage_V,
                  "electrode_gap_cm": electrode_gap_cm,
                  "LET_keV_um": float(LET_keV_um),
                  "a0_nm": a0_nm,
                  "particle": particle,
                  "RDD_model": RDD_model,
                  "IC_angle_rad": theta_rad,
                  }

    extra_params_dic = {
        "unit_length_cm": grid_size_um*1e-4,
        "track_radius_cm": track_radius_cm,
        "SHOW_PLOT": SHOW_PLOT,
        "PRINT_parameters": PRINT_parameters,
    }

    ks = single_track_PDEsolver(result_dic, extra_params_dic)
    result_dic["ks"] = ks
    return pd.DataFrame([result_dic])


def IonTracks_continuous_beam(E_MeV_u,
                              voltage_V,
                              doserate_Gy_min,
                              electrode_gap_cm=0.2,
                              particle="proton",
                              grid_size_um=5.0,
                              PRINT_parameters=False,
                              SHOW_PLOT=False,
                              myseed=int(np.random.randint(1, 1e7))
                              ):
    '''
    Calculate the stopping power, fluence-rate, and track radius as a
    function of proton energy and dose-rate.

    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    '''

    LET_keV_um = E_MeV_u_to_LET_keV_um(E_MeV_u, particle=particle)
    fluencerate_cm2_s = doserate_to_fluence(
        doserate_Gy_min, E_MeV_u, particle=particle)

    track_radius_cm = calc_b_cm(LET_keV_um)

    result_dic = {"E_MeV_u": E_MeV_u,
                  "LET_keV_um": float(LET_keV_um),
                  "voltage_V": voltage_V,
                  "particle": particle,
                  "electrode_gap_cm": electrode_gap_cm,
                  "doserate_Gy_min": doserate_Gy_min,
                  "fluencerate_cm2_s": fluencerate_cm2_s
                  }

    extra_params_dic = {
        "unit_length_cm": grid_size_um*1e-4,
        "track_radius_cm": track_radius_cm,
        "SHOW_PLOT": SHOW_PLOT,
        "PRINT_parameters": PRINT_parameters,
        "seed": myseed
    }

    ks = continuous_beam_PDEsolver(result_dic, extra_params_dic)
    result_dic["ks_IonTracks"] = ks

    return pd.DataFrame([result_dic])
