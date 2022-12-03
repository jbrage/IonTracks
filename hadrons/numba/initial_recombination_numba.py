import numpy as np
import numpy.random as rnd
import time
from math import exp, sqrt, pi, log, cos, sin
from numba import njit
from ..geiss_utils import Geiss_r_max, Geiss_RRD_cm
from ..python.initial_recombination import single_track_PDEsolver as single_track_PDEsolver_base


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

class single_track_PDEsolver(single_track_PDEsolver_base):
    def solve(self):
        # initialise the Gaussian distribution in the array
        start_time = time.time()
        if self.RDD_model == "Gauss":
            self.no_initialised_charge_carriers, self.positive_array, self.negative_array = self.initialize_Gaussian_distriution_Gauss(no_x, no_z_with_buffer, mid_xy_array, unit_length_cm, no_z, no_z_electrode, track_radius_cm, Gaussian_factor)
        elif self.RDD_model == "Geiss":
            self.no_initialised_charge_carriers, self.positive_array, self.negative_array = self.initialize_Gaussian_distriution_Geiss(no_x, no_z_with_buffer, mid_xy_array, unit_length_cm, no_z, no_z_electrode, c, a0_cm, r_max_cm)
        else:
            raise ValueError(f'Invalid RDD model: {self.RDD_model}')
            
        if self.debug: print(f'Gaussian distriution initialization time {time.time()-start_time}')

        calculation_time = time.time()
        
        f = self.main_loop()

        if self.debug: print('Calculation loop combined time: ', time.time()-calculation_time)

        return 1./f

    @njit
    def initialize_Gaussian_distriution_Gauss(self):
        self.no_initialised_charge_carriers = 0.0
        self.positive_array = np.zeros((self.no_x,self.no_x,self.no_z_with_buffer))
        self.negative_array = np.zeros((self.no_x,self.no_x,self.no_z_with_buffer))

        for i in range(self.no_x):
            for j in range(self.no_x):
                distance_from_center_cm = sqrt((i - self.mid_xy_array) ** 2 + (j - self.mid_xy_array) ** 2) * self.unit_length_cm
                ion_density = self.Gaussian_factor * exp(-distance_from_center_cm ** 2 / self.track_radius_cm ** 2)
                self.no_initialised_charge_carriers += self.no_z*ion_density
                self.positive_array[i, j, no_z_electrode:(no_z_electrode+self.no_z)] += ion_density
                self.negative_array[i, j, no_z_electrode:(no_z_electrode+self.no_z)] += ion_density


    @njit
    def initialize_Gaussian_distriution_Geiss(self):
        self.no_initialised_charge_carriers = 0.0
        self.positive_array = np.zeros((self.no_x,self.no_x,self.no_z_with_buffer))
        self.negative_array = np.zeros((self.no_x,self.no_x,self.no_z_with_buffer))

        for i in range(self.no_x):
            for j in range(self.no_x):
                distance_from_center_cm = sqrt((i - self.mid_xy_array) ** 2 + (j - self.mid_xy_array) ** 2) * self.unit_length_cm
                ion_density = Geiss_RRD_cm(distance_from_center_cm, self.c, self.a0_cm, self.r_max_cm)
                self.no_initialised_charge_carriers += self.no_z*ion_density
                self.positive_array[i, j, no_z_electrode:(no_z_electrode+self.no_z)] += ion_density
                self.negative_array[i, j, no_z_electrode:(no_z_electrode+self.no_z)] += ion_density


    @njit
    def main_loop(self):
        positive_array_temp = np.zeros((self.no_x, self.no_x, self.no_z_with_buffer))
        negative_array_temp = np.zeros((self.no_x, self.no_x, self.no_z_with_buffer))
        no_recombined_charge_carriers = 0.0
        
        for time_step in range(self.computation_time_steps):

            # calculate the new densities and store them in temporary arrays
            szcz_pos = (self.sz+self.cz*(self.cz+1.)/2.)
            szcz_neg = (self.sz+self.cz*(self.cz-1.)/2.)

            sycy_pos = (self.sy+self.cy*(self.cy+1.)/2.)
            sycy_neg = (self.sy+self.cy*(self.cy-1.)/2.)

            sxcx_pos = (self.sx+self.cx*(self.cx+1.)/2.)
            sxcx_neg = (self.sx+self.cx*(self.cx-1.)/2.)

            for i in range(1, self.no_x-1):
                for j in range(1, self.no_x-1):
                    for k in range(1, self.no_z_with_buffer-1):
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

            for i in range(1, self.no_x-1):
                for j in range(1, self.no_x-1):
                    for k in range(1, self.no_z_with_buffer-1):
                        # update the positive and negative arrays
                        self.positive_array[i,j,k] = positive_array_temp[i,j,k]
                        self.negative_array[i,j,k] = negative_array_temp[i,j,k]

        f = (self.no_initialised_charge_carriers - no_recombined_charge_carriers)/self.no_initialised_charge_carriers

        return f