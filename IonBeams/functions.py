import mpmath
import numpy as np
from math import pi, exp, sin, log, sqrt
from scipy.special import hankel1
from scipy.interpolate import interp1d
import pandas as pd

mpmath.mp.dps = 50  # number of figures for computing exponential integrals

# general parameters
ion_mobility = 1.65     # TODO: units
W = 33.9                # eV/ion pair for air
# define the parameters from Kanai (1998)
ion_diff = 3.7e-2       # cm^2/s, averaged for positive and negative ions
alpha = 1.60e-6         # cm^3/s, recombination constant
IC_angle_rad = 0.0      # not available in this version


def Jaffe_theory(energy_MeV, voltage_V, electrode_gap_cm):
    '''
    The Jaffe theory for initial recombination. Returns the inverse
    collection efficiency, i.e. the recombination correction factor
    '''
    electric_field = voltage_V/electrode_gap_cm
    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    LET_eV_cm = LET_keV_um * 1e7

    b_cm = calc_b_cm(LET_keV_um)

    N0 = LET_eV_cm / W
    g = alpha * N0 / (8. * pi * ion_diff)

    if IC_angle_rad > 0:
        x = (b_cm * ion_mobility * electric_field * sin(IC_angle_rad) / (2 * ion_diff)) ** 2

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
        factor = mpmath.exp(-1.0/g)*ion_mobility*b_cm**2*electric_field/(2.*g*electrode_gap_cm*ion_diff)
        first_term = mpmath.ei(1.0/g + log(1.0 + (2.0*electrode_gap_cm*ion_diff/(ion_mobility*b_cm**2*electric_field))))
        second_term = mpmath.ei(1.0/g)
        f = factor*(first_term - second_term)
    return float(1./f)


def E_MeV_u_to_LET_keV_um(E_MeV_u, particle="proton", material="air"):
    '''
    Calculate the stopping power in air using PSTAR data
    '''
    
    if material == "air":
        fname = "input_data/stopping_power_air.csv"
    else: # water
        fname = "input_data/stopping_power_water.csv"
    
    df = pd.read_csv(fname)
    
    E_col_name = "E_MeV_u"
    particle_col_name = "{}_LET_keV_um".format(particle)
    
    interpolate_LET = interp1d(df[E_col_name], df[particle_col_name])
   
    if not particle_col_name in df.columns:
        print("Particle {} is not supported".format(particle))
        return 0
           
    return float(interpolate_LET(E_MeV_u))


def E_MeV_at_reference_depth_cm(energy_MeV):
    '''
    Geant4-calculated depth at 2 cm water depth
    '''
    E_MeV_at_2cm = {
                70: 48,
                150: 138,
                226: 215
               }
    return E_MeV_at_2cm[energy_MeV]


def doserate_to_fluence(dose_Gy_min, energy_MeV):
    '''
    Convert the dose-rate to a fluence-rate for the given proton energy
    '''
    dose_Gy_s = dose_Gy_min / 60.0
    density_kg_m3 = 1.225
    density_kg_cm3 = density_kg_m3 * 1e-6
    joule_to_keV = 6.241E+15
    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    LET_keV_cm = LET_keV_um * 1e4
    fluence_cm2_s = dose_Gy_s * joule_to_keV * density_kg_cm3 / LET_keV_cm
    return fluence_cm2_s


def calc_b_cm(LET_keV_um):
    '''
    Calculate the Gaussian track radius as suggested by Rossomme et al.
    Returns the track radius in cm given a LET [keV/um]
    '''
    data = np.genfromtxt("input_data/LET_b.dat", delimiter=",", dtype=float)
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
    # b_cm = 1.05*1e-3 # cm   ... Kanai, avoids ridges
    return b_cm
