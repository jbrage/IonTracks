import numpy as np
import matplotlib.pyplot as plt


alpha = 1.6e-6 # recombination coefficient [cm^3 / s]
k_1 = 1.36 # positive charge carriers [cm^2 / (V s)]
k_2 = 2.1  # negative charge carriers [cm^2 / (V s)]
e_charge = 1.60217662e-19 # [C]


'''
References:
    [1] Boag and Currant (1980) "Current collection and ionic recombination in small cylindrical ionization chambers exposed to pulsed radiation"
    [2] Andreo, Burns, Nahum et al, Fundamentals of Ionizing Radiation
    [3] Boutillon (1998) "Volume recombination parameter in ionization chambers"
    [4] Liszka et al (2018) "Ion recombination and polarity correction factors for a plane–parallel ionization chamber in a proton scanning beam"
'''

def Boag_pulsed(Qdensity_C_cm3, d_cm, V):
    '''
    The Boag theory (ref [1]) for recombination in a parallel-plate ionization chamber in pulsed beams

    The theory assumes:
        - uniform charge carrier distribution
        - uniform electric field
        - enables the inclusion of a free electron component (not included here)
    '''

    mu = alpha / (e_charge * (k_1 + k_2))
    # ICRU report 34 suggests mu = 3.02e10 V m /C, see ref [2] page 518

    r = Qdensity_C_cm3

    u = mu * r * d_cm**2 / V
    f = 1./u * np.log(1 + u)
    return f


def Boag_Continuous(Qdensity_C_cm3_s, d_cm, V):
    '''
    The theory (ref [3, 4]) for recombination in a parallel-plate ionization chamber in continuous beams

    The theory assumes:
        - uniform charge carrier distribution
        - uniform electric field

    Returns:
        - f
        - f minus 1 std
        - f plus 1 std
    '''

    def f_c(mu):
        ksi_squared = mu * (d_cm**4 / V**2) * Qdensity_C_cm3_s
        return 1./(1 + ksi_squared)

    mu_c = alpha / (6*e_charge*k_1*k_2)  # [V^2 s / (cm C)]
    # mu_c = 6.73e11 # [V^2 s / (cm C)]

    # uncertainty:
    # Ionization chambers, Chapter 3, in The Dosimetry of Ionizing Radiation (1987)
    sigma = 2
    mu_c_std = 0.8e11*sigma
    mu_c_high = mu_c + mu_c_std
    mu_c_low = mu_c - mu_c_std

    return f_c(mu_c), f_c(mu_c_low), f_c(mu_c_high)
