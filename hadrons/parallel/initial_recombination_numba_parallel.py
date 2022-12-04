import numpy as np
import time
from math import exp, sqrt
from numba import njit, prange
from ..geiss_utils import Geiss_RRD_cm
from ..python.initial_recombination import single_track_PDEsolver as single_track_PDEsolver_base
from ..common_properties import alpha, no_z_electrode

# jitted functions defined outside the solver class as 'self' is not recogized as a supported type 

@njit
def initialize_Gaussian_distriution_Gauss(no_x, no_z_with_buffer, mid_xy_array, unit_length_cm, no_z, track_radius_cm, Gaussian_factor):
    no_initialised_charge_carriers = 0.0
    positive_array = np.zeros((no_x, no_x, no_z_with_buffer))
    negative_array = np.zeros((no_x, no_x, no_z_with_buffer))

    for i in range(no_x):
        for j in range(no_x):
            distance_from_center_cm = sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) * unit_length_cm
            ion_density = Gaussian_factor * exp(-distance_from_center_cm ** 2 / track_radius_cm ** 2)
            no_initialised_charge_carriers += no_z*ion_density
            positive_array[i, j, no_z_electrode:(no_z_electrode+no_z)] += ion_density
            negative_array[i, j, no_z_electrode:(no_z_electrode+no_z)] += ion_density

    return no_initialised_charge_carriers, positive_array, negative_array


@njit
def initialize_Gaussian_distriution_Geiss(no_x, no_z_with_buffer, mid_xy_array, unit_length_cm, no_z, c, a0_cm, r_max_cm):
    no_initialised_charge_carriers = 0.0
    positive_array = np.zeros((no_x, no_x, no_z_with_buffer))
    negative_array = np.zeros((no_x, no_x, no_z_with_buffer))

    for i in range(no_x):
        for j in range(no_x):
            distance_from_center_cm = sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2) * unit_length_cm
            ion_density = Geiss_RRD_cm(distance_from_center_cm, c, a0_cm, r_max_cm)
            no_initialised_charge_carriers += no_z*ion_density
            positive_array[i, j, no_z_electrode:(no_z_electrode+no_z)] += ion_density
            negative_array[i, j, no_z_electrode:(no_z_electrode+no_z)] += ion_density

    return no_initialised_charge_carriers, positive_array, negative_array


@njit
def main_loop(sz, cz, sy, cy, sx, cx, no_x, no_z_with_buffer, positive_array, negative_array, dt, computation_time_steps, no_initialised_charge_carriers):
    no_recombined_charge_carriers = 0.0

    for _ in range(computation_time_steps):

        # calculate the new densities and store them in temporary arrays
        # dunno what would be a good name for those coefficients
        szcz_pos = (sz+cz*(cz+1.)/2.)
        szcz_neg = (sz+cz*(cz-1.)/2.)

        sycy_pos = (sy+cy*(cy+1.)/2.)
        sycy_neg = (sy+cy*(cy-1.)/2.)

        sxcx_pos = (sx+cx*(cx+1.)/2.)
        sxcx_neg = (sx+cx*(cx-1.)/2.)

        positive_array_temp, negative_array_temp, no_recombined_charge_carriers_chunk = lax_wendroff_wrapper(
            no_x, no_z_with_buffer, szcz_pos, szcz_neg, sycy_pos, sycy_neg, sxcx_pos, sxcx_neg, positive_array, negative_array, cx, cy, cz, sx, sy, sz, alpha, dt)

        no_recombined_charge_carriers += no_recombined_charge_carriers_chunk

        positive_array = positive_array_temp
        negative_array = negative_array_temp

    f = (no_initialised_charge_carriers - no_recombined_charge_carriers)/no_initialised_charge_carriers

    return f


@njit(parallel=True)
def lax_wendroff_wrapper(no_x, no_z_with_buffer, szcz_pos, szcz_neg, sycy_pos, sycy_neg, sxcx_pos, sxcx_neg, positive_array, negative_array, cx, cy, cz, sx, sy, sz, alpha, dt):
    positive_array_temp = np.zeros((no_x, no_x, no_z_with_buffer))
    negative_array_temp = np.zeros((no_x, no_x, no_z_with_buffer))
    no_recombined_charge_carriers = 0

    for i in prange(1, no_x-1):
        for j in prange(1, no_x-1):
            for k in prange(1, no_z_with_buffer-1):
                positive_array_temp[i, j, k], negative_array_temp[i, j, k], no_recombined_charge_carriers_chunk = lax_wendroff_scheme(
                    i, j, k, szcz_pos, szcz_neg, sycy_pos, sycy_neg, sxcx_pos, sxcx_neg, positive_array, negative_array, cx, cy, cz, sx, sy, sz, alpha, dt)
                no_recombined_charge_carriers += no_recombined_charge_carriers_chunk

    return positive_array_temp, negative_array_temp, no_recombined_charge_carriers


@njit
def lax_wendroff_scheme(i, j, k, szcz_pos, szcz_neg, sycy_pos, sycy_neg, sxcx_pos, sxcx_neg, positive_array, negative_array, cx, cy, cz, sx, sy, sz, alpha, dt):
    # using the Lax-Wendroff scheme
    positive_temp_entry = szcz_pos*positive_array[i, j, k-1]
    positive_temp_entry += szcz_neg*positive_array[i, j, k+1]

    positive_temp_entry += sycy_pos*positive_array[i, j-1, k]
    positive_temp_entry += sycy_neg*positive_array[i, j+1, k]

    positive_temp_entry += sxcx_pos*positive_array[i-1, j, k]
    positive_temp_entry += sxcx_neg*positive_array[i+1, j, k]

    positive_temp_entry += (1. - cx*cx - cy*cy - cz*cz - 2.*(sx+sy+sz))*positive_array[i, j, k]

    # same for the negative charge carriers
    negative_temp_entry = szcz_pos*negative_array[i, j, k+1]
    negative_temp_entry += szcz_neg*negative_array[i, j, k-1]

    negative_temp_entry += sycy_pos*negative_array[i, j+1, k]
    negative_temp_entry += sycy_neg*negative_array[i, j-1, k]

    negative_temp_entry += sxcx_pos*negative_array[i+1, j, k]
    negative_temp_entry += sxcx_neg*negative_array[i-1, j, k]

    negative_temp_entry += (1. - cx*cx - cy*cy - cz*cz - 2.*(sx+sy+sz))*negative_array[i, j, k]

    # the recombination part
    recomb_temp = alpha*positive_array[i, j, k]*negative_array[i, j, k]*dt

    positive_array_temp = positive_temp_entry - recomb_temp
    negative_array_temp = negative_temp_entry - recomb_temp
    no_recombined_charge_carriers = recomb_temp

    return positive_array_temp, negative_array_temp, no_recombined_charge_carriers


class single_track_PDEsolver(single_track_PDEsolver_base):
    def solve(self):
        # initialise the Gaussian distribution in the array

        start_time = time.time()
        if self.RDD_model == "Gauss":
            self.no_initialised_charge_carriers, self.positive_array, self.negative_array = initialize_Gaussian_distriution_Gauss(
                self.no_x, self.no_z_with_buffer, self.mid_xy_array, self.unit_length_cm, self.no_z, self.track_radius_cm, self.Gaussian_factor)
        elif self.RDD_model == "Geiss":
            self.no_initialised_charge_carriers, self.positive_array, self.negative_array = initialize_Gaussian_distriution_Geiss(
                self.no_x, self.no_z_with_buffer, self.mid_xy_array, self.unit_length_cm, self.no_z, self.c, self.a0_cm, self.r_max_cm)
        else:
            raise ValueError(f'Invalid RDD model: {self.RDD_model}')

        if self.debug:
            print(f'Gaussian distriution initialization time {time.time()-start_time}')

        calculation_time = time.time()

        f = main_loop(
            self.sz, self.cz, self.sy, self.cy, self.sx, self.cx, self.no_x, self.no_z_with_buffer, self.positive_array, self.negative_array, self.dt, self.computation_time_steps, self.no_initialised_charge_carriers
        )

        if self.debug:
            print('Calculation loop combined time: ', time.time()-calculation_time)

        return 1./f
