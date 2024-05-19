from __future__ import division
import numpy as np

from generic_electron_solver import GenericElectronSolver


class PulsedBeamPDEsolver(GenericElectronSolver):
    unit_length_cm = 5e-4  # cm, size of every voxel length

    def get_electron_density_after_beam(self, positive_array, negative_array):
        delta_border = 2

        for k in range(self.no_z_electrode, self.no_z + self.no_z_electrode):
            for i in range(delta_border, self.no_xy - delta_border):
                for j in range(delta_border, self.no_xy - delta_border):
                    positive_array[i, j, k] += self.electron_density_per_cm3
                    negative_array[i, j, k] += self.electron_density_per_cm3
                    if positive_array[i, j, k] > MAXVAL:
                        MAXVAL = positive_array[i, j, k]
                    no_initialised_charge_carriers += self.electron_density_per_cm3

        return positive_array, negative_array

    def calculate(self):
        # refering to each param with a `self.` prefix makes the code less readable so we unpack them here
        # this also prevents accidental mutations to the params - subsequent calls to `calculate` won't have side-effects
        no_xy = self.no_xy
        no_z_with_buffer = self.no_z_with_buffer
        computation_time_steps = self.computation_time_steps
        no_z_electrode = self.no_z_electrode
        dt = self.dt
        no_z = self.no_z
        sx = self.sx
        sy = self.sy
        sz = self.sz
        cx = self.cx
        cy = self.cy
        cz = self.cz
        alpha = self.alpha

        positive_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
        negative_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
        positive_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
        negative_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
        no_recombined_charge_carriers = 0.0
        no_initialised_charge_carriers = 0.0

        """
        Create an array which include the number of track to be inserted for each time step
        The tracks are distributed uniformly in time
        """

        f, positive_temp_entry, negative_temp_entry, recomb_temp = 0.0

        """
        Fill the array with the electron density according to the pulesed beam
        """
        positive_array, negative_array = self.get_electron_density_after_beam(
            positive_array, negative_array
        )

        szcz_pos = sz + cz * (cz + 1.0) / 2.0
        szcz_neg = sz + cz * (cz - 1.0) / 2.0

        sycy_pos = sy + cy * (cy + 1.0) / 2.0
        sycy_neg = sy + cy * (cy - 1.0) / 2.0

        sxcx_pos = sx + cx * (cx + 1.0) / 2.0
        sxcx_neg = sx + cx * (cx - 1.0) / 2.0

        """
        Start the simulation by evovling the distribution one step at a time
        """
        for _ in range(computation_time_steps):
            # calculate the new densities and store them in temporary arrays
            for i in range(1, no_xy - 1):
                for j in range(1, no_xy - 1):
                    for k in range(1, no_z_with_buffer - 1):
                        # using the Lax-Wendroff scheme
                        positive_temp_entry = szcz_pos * positive_array[i, j, k - 1]
                        positive_temp_entry += szcz_neg * positive_array[i, j, k + 1]

                        positive_temp_entry += sycy_pos * positive_array[i, j - 1, k]
                        positive_temp_entry += sycy_neg * positive_array[i, j + 1, k]

                        positive_temp_entry += sxcx_pos * positive_array[i - 1, j, k]
                        positive_temp_entry += sxcx_neg * positive_array[i + 1, j, k]

                        positive_temp_entry += (
                            1.0 - cx * cx - cy * cy - cz * cz - 2.0 * (sx + sy + sz)
                        ) * positive_array[i, j, k]

                        # same for the negative charge carriers
                        negative_temp_entry = szcz_pos * negative_array[i, j, k + 1]
                        negative_temp_entry += szcz_neg * negative_array[i, j, k - 1]

                        negative_temp_entry += sycy_pos * negative_array[i, j + 1, k]
                        negative_temp_entry += sycy_neg * negative_array[i, j - 1, k]

                        negative_temp_entry += sxcx_pos * negative_array[i + 1, j, k]
                        negative_temp_entry += sxcx_neg * negative_array[i - 1, j, k]

                        negative_temp_entry += (
                            1.0 - cx * cx - cy * cy - cz * cz - 2.0 * (sx + sy + sz)
                        ) * negative_array[i, j, k]

                        # the recombination part
                        recomb_temp = (
                            alpha
                            * positive_array[i, j, k]
                            * negative_array[i, j, k]
                            * dt
                        )
                        positive_array_temp[i, j, k] = positive_temp_entry - recomb_temp
                        negative_array_temp[i, j, k] = negative_temp_entry - recomb_temp

                        if k > no_z_electrode and k < (no_z + no_z_electrode):
                            # sum over the recombination between the virtual electrodes
                            no_recombined_charge_carriers += recomb_temp

            # update the positive and negative arrays
            for i in range(1, no_xy - 1):
                for j in range(1, no_xy - 1):
                    for k in range(1, no_z_with_buffer - 1):
                        positive_array[i, j, k] = positive_array_temp[i, j, k]
                        negative_array[i, j, k] = negative_array_temp[i, j, k]

        # return the collection efficiency
        f = (
            no_initialised_charge_carriers - no_recombined_charge_carriers
        ) / no_initialised_charge_carriers
        return f
