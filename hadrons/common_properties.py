# define the parameters from Kanai (1998)
W = 33.9  # eV/ion pair for air
ion_mobility = 1.65  # cm^2 s^-1 V^-1, averaged for positive and negative ions
ion_diff = 3.7e-2  # cm^2/s, averaged for positive and negative ions
alpha = 1.60e-6  # cm^3/s, recombination constantno_figure_updates = 5

n_track_radii = 6  # scaling factor to determine the grid width
# unit_length_cm = 6e-4   # [cm], grid resolution

no_z_electrode = 4  # length of the electrode-buffer to ensure no ions drift through the array in one time step

# parameters for the Geiss RDD
air_density_g_cm3 = 1.225e-3  # dry air
water_density_g_cm3 = 1.0
