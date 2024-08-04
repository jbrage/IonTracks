import mpmath

mpmath.mp.dps = 50  # number of figures for computing exponential integrals
# general parameters
ion_mobility = 1.65  # TODO: units
W = 33.9  # eV/ion pair for air
# define the parameters from Kanai (1998)
ion_diff = 3.7e-2  # cm^2/s, averaged for positive and negative ions
alpha = 1.60e-6  # cm^3/s, recombination constant
