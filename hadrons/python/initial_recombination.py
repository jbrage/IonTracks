import numpy as np
import numpy.random as rnd
import time
from math import exp, sqrt, pi, log, cos, sin
from ..geiss_utils import Geiss_r_max, Geiss_RRD_cm


# define the parameters from Kanai (1998)
W  = 33.9                # eV/ion pair for air
ion_mobility  = 1.65     # cm^2 s^-1 V^-1, averaged for positive and negative ions
ion_diff  = 3.7e-2       # cm^2/s, averaged for positive and negative ions
alpha  = 1.60e-6         # cm^3/s, recombination constantno_figure_updates = 5
        
n_track_radii = 6 # scaling factor to determine the grid width
#unit_length_cm = 6e-4   # [cm], grid resolution

no_z_electrode = 4 #length of the electrode-buffer to ensure no ions drift through the array in one time step

# parameters for the Geiss RDD
air_density_g_cm3 = 1.225e-3  # dry air
water_density_g_cm3 = 1.0

class single_track_PDEsolver():

    def __init__(self,
                 LET_keV_um: float,
                 voltage_V: float,
                 IC_angle_rad: float,
                 electrode_gap_cm: float,
                 E_MeV_u: float,
                 a0_nm: float,
                 RDD_model: str,
                 unit_length_cm: float,
                 track_radius_cm: float,
                 debug: bool = False):
    
        LET_eV_cm  = LET_keV_um*1e7

        density_ratio = water_density_g_cm3 / air_density_g_cm3
        a0_cm = a0_nm * 1e-7 * density_ratio
        r_max_cm = Geiss_r_max(E_MeV_u, air_density_g_cm3)    
        c = LET_eV_cm / (pi * W) * (1 / (1 + 2 * log(r_max_cm / a0_cm)))
        
        # density parameters for the Gaussian structure
        N0 = LET_eV_cm/W     # Linear charge carrier density
        Gaussian_factor = N0/(pi*track_radius_cm**2)

        # grid dimension parameters
        no_x = int(track_radius_cm*n_track_radii/unit_length_cm)
        # print(track_radius_cm*n_track_radii)
        no_z = int(electrode_gap_cm/unit_length_cm) #number of elements in the z direction
        no_z_with_buffer = 2*no_z_electrode + no_z
        # find the middle of the arrays
        mid_xy_array = int(no_x/2.)

        # depending on the cluster/computer, the upper limit may be changed
        if (no_x*no_x*no_z) > 1e8:
            raise ValueError("Too many elements in the array: %i" % (no_x*no_x*no_z_with_buffer))

        # preallocate arrays
        positive_array = np.zeros((no_x,no_x,no_z_with_buffer))
        negative_array = np.zeros((no_x,no_x,no_z_with_buffer))
        no_initialised_charge_carriers = 0.0

        dt = 1.
        von_neumann_expression = False
        Efield = voltage_V/electrode_gap_cm

        # find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        # decreases and eventually damps out
        while not von_neumann_expression:
            dt /= 1.01
            # as defined in the Deghan (2004) paper
            sx = ion_diff*dt/(unit_length_cm**2)
            sy, sz = sx, sx

            cx = ion_mobility*Efield*dt/unit_length_cm*sin(IC_angle_rad)
            cy = 0
            cz = ion_mobility*Efield*dt/unit_length_cm*cos(IC_angle_rad)

            # check von Neumann's criterion
            von_neumann_expression = (2*(sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1 and cx**2*cy**2*cz**2 <= 8*sx*sy*sz)

        # calculate the number of step required to drag the two charge carrier distributions apart
        separation_time_steps = int(electrode_gap_cm/(2.*ion_mobility*Efield*dt))
        computation_time_steps = separation_time_steps*2

        if debug:
            print("Electric field = %s V/cm" % Efield)
            print("Separation time = %3.2E s" % (separation_time_steps*dt))
            print("Time steps      = %3.2E" % computation_time_steps)
            print("Time step dt    = %3.2E s" % dt)
            print("Number of voxels = %3.2E" % (no_x**2 * no_z_with_buffer))
            print("Number of pixels = %d (x = y directions)" % no_x)
            print("Number of pixels = %d (z direction)" % no_z)

        # define the radial dose model to be used
        if RDD_model == "Gauss":
            def RDD_function(r_cm):
                return Gaussian_factor * exp(-r_cm ** 2 / track_radius_cm ** 2)

        elif RDD_model == "Geiss":
            def RDD_function(r_cm):
                return Geiss_RRD_cm(r_cm, c, a0_cm, r_max_cm)
        else:
            print("RDD model {} undefined.".format(RDD_model))
            return 0

        # initialise the Gaussian distribution in the array
        start_time = time.time()
        for i in range(no_x):
            for j in range(no_x):
                if debug: print(f'\rGaussian distribution loop: i - {i+1: >3}/{no_x}, j - {j+1: >3}/{no_x} Time: {time.time()-start_time: .2f}', end='')
                distance_from_center_cm = sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) * unit_length_cm
                ion_density = RDD_function(distance_from_center_cm)
                no_initialised_charge_carriers += no_z*ion_density
                positive_array[i, j, no_z_electrode:(no_z_electrode+no_z)] += ion_density
                negative_array[i, j, no_z_electrode:(no_z_electrode+no_z)] += ion_density
        if debug: print('')

        self.computation_time_steps = computation_time_steps
        self.sx = sy
        self.cx = cy
        self.sy = sy
        self.cy = cy
        self.sz = sz
        self.cz = cz
        self.no_x = no_x
        self.no_z_with_buffer = no_z_with_buffer
        self.positive_array = positive_array
        self.negative_array = negative_array
        self.dt = dt
        self.no_initialised_charge_carriers = no_initialised_charge_carriers
        self.debug = debug


    def solve(self):
        positive_array_temp = np.zeros((self.no_x, self.no_x, self.no_z_with_buffer))
        negative_array_temp = np.zeros((self.no_x, self.no_x, self.no_z_with_buffer))
        no_recombined_charge_carriers = 0.0

        calculation_time = time.time()
        for time_step in range(self.computation_time_steps):
            if self.debug: print(f'Calculation loop iteration {time_step+1}/{self.computation_time_steps}')

            # calculate the new densities and store them in temporary arrays
            start_time = time.time()

            #dunno what would be a good name for those coefficients
            szcz_pos = (self.sz+self.cz*(self.cz+1.)/2.)
            szcz_neg = (self.sz+self.cz*(self.cz-1.)/2.)

            sycy_pos = (self.sy+self.cy*(self.cy+1.)/2.)
            sycy_neg = (self.sy+self.cy*(self.cy-1.)/2.)

            sxcx_pos = (self.sx+self.cx*(self.cx+1.)/2.)
            sxcx_neg = (self.sx+self.cx*(self.cx-1.)/2.)

            for i in range(1,self.no_x-1):
                for j in range(1,self.no_x-1):
                    for k in range(1,self.no_z_with_buffer-1):
                        if self.debug: print(f'\rDensities calculation loop: i - {i+1: >3}/{self.no_x-1}, j - {j+1: >3}/{self.no_x-1}, k - {k+1: >3}/{self.no_z_with_buffer-1} Time: {time.time()-start_time: .2f}', end='')
                        # using the Lax-Wendroff scheme

                        positive_temp_entry = szcz_pos*self.positive_array[i,j,k-1]
                        positive_temp_entry += szcz_neg*self.positive_array[i,j,k+1]

                        positive_temp_entry += sycy_pos*self.positive_array[i,j-1,k]
                        positive_temp_entry += sycy_neg*self.positive_array[i,j+1,k]

                        positive_temp_entry += sxcx_pos*self.positive_array[i-1,j,k]
                        positive_temp_entry += sxcx_neg*self.positive_array[i+1,j,k]

                        positive_temp_entry += (1.- self.cx*self.cx - self.cy*self.cy - self.cz*self.cz - 2.*(self.sx+self.sy+self.sz))*self.positive_array[i,j,k]

                        # same for the negative charge carriers
                        negative_temp_entry = szcz_pos*self.negative_array[i,j,k+1]
                        negative_temp_entry += szcz_neg*self.negative_array[i,j,k-1]

                        negative_temp_entry += sycy_pos*self.negative_array[i,j+1,k]
                        negative_temp_entry += sycy_neg*self.negative_array[i,j-1,k]

                        negative_temp_entry += sxcx_pos*self.negative_array[i+1,j,k]
                        negative_temp_entry += sxcx_neg*self.negative_array[i-1,j,k]

                        negative_temp_entry += (1. - self.cx*self.cx - self.cy*self.cy - self.cz*self.cz - 2.*(self.sx+self.sy+self.sz))*self.negative_array[i,j,k]

                        # the recombination part
                        recomb_temp = alpha*self.positive_array[i,j,k]*self.negative_array[i,j,k]*self.dt

                        positive_array_temp[i,j,k] = positive_temp_entry - recomb_temp
                        negative_array_temp[i,j,k] = negative_temp_entry - recomb_temp
                        no_recombined_charge_carriers += recomb_temp
            if self.debug: print('')

            # update the positive and negative arrays
            start_time = time.time()

            self.positive_array[1:-1,1:-1,1:-1] = positive_array_temp[1:-1,1:-1,1:-1]
            self.negative_array[1:-1,1:-1,1:-1] = negative_array_temp[1:-1,1:-1,1:-1]

        f = (self.no_initialised_charge_carriers - no_recombined_charge_carriers)/self.no_initialised_charge_carriers

        if self.debug: print('Calculation loop combined time: ', time.time()-calculation_time)
        return 1./f