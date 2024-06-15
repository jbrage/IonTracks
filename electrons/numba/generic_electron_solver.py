import numpy as np
from numba import njit

from electrons.common.generic_electron_solver import (
    GenericElectronSolver as PythonGenericElectronSolver,
)


class NumbaGenericElectronSolver(PythonGenericElectronSolver):

    @njit
    def calculate(self):
        positive_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        negative_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        positive_array_temp = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        negative_array_temp = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        no_recombined_charge_carriers = 0.0
        no_initialised_charge_carriers = 0.0

        """
        Create an array which includes the number of track to be inserted for each time step
        The tracks are distributed uniformly in time
        """

        positive_temp_entry = negative_temp_entry = recomb_temp = 0.0

        f_steps_list = np.zeros(self.computation_time_steps)

        szcz_pos = self.sz + self.cz * (self.cz + 1.0) / 2.0
        szcz_neg = self.sz + self.cz * (self.cz - 1.0) / 2.0

        sycy_pos = self.sy + self.cy * (self.cy + 1.0) / 2.0
        sycy_neg = self.sy + self.cy * (self.cy - 1.0) / 2.0

        sxcx_pos = self.sx + self.cx * (self.cx + 1.0) / 2.0
        sxcx_neg = self.sx + self.cx * (self.cx - 1.0) / 2.0

        cxyzsyz = (
            1.0
            - self.cx * self.cx
            - self.cy * self.cy
            - self.cz * self.cz
            - 2.0 * (self.sx + self.sy + self.sz)
        )

        """
        Start the simulation by evolving the distribution one step at a time
        """

        for time_step in range(self.computation_time_steps):

            """
            Refill the array with the electron density each time step
            """
            if self.should_simulate_beam_for_time_step(time_step):
                no_initialised_step = self.simulate_beam(
                    positive_array,
                    negative_array,
                )
            else:
                no_initialised_step = 0.0

            no_initialised_charge_carriers += no_initialised_step

            # calculate the new densities and store them in temporary arrays
            for i in range(1, self.no_xy - 1):
                for j in range(1, self.no_xy - 1):
                    for k in range(1, self.no_z_with_buffer - 1):
                        # using the Lax-Wendroff scheme
                        positive_temp_entry = szcz_pos * positive_array[i, j, k - 1]
                        positive_temp_entry += szcz_neg * positive_array[i, j, k + 1]

                        positive_temp_entry += sycy_pos * positive_array[i, j - 1, k]
                        positive_temp_entry += sycy_neg * positive_array[i, j + 1, k]

                        positive_temp_entry += sxcx_pos * positive_array[i - 1, j, k]
                        positive_temp_entry += sxcx_neg * positive_array[i + 1, j, k]

                        positive_temp_entry += cxyzsyz * positive_array[i, j, k]

                        # same for the negative charge carriers
                        negative_temp_entry = szcz_pos * negative_array[i, j, k + 1]
                        negative_temp_entry += szcz_neg * negative_array[i, j, k - 1]

                        negative_temp_entry += sycy_pos * negative_array[i, j + 1, k]
                        negative_temp_entry += sycy_neg * negative_array[i, j - 1, k]

                        negative_temp_entry += sxcx_pos * negative_array[i + 1, j, k]
                        negative_temp_entry += sxcx_neg * negative_array[i - 1, j, k]

                        negative_temp_entry += cxyzsyz * negative_array[i, j, k]

                        # the recombination part
                        recomb_temp = (
                            self.alpha
                            * positive_array[i, j, k]
                            * negative_array[i, j, k]
                            * self.dt
                        )
                        positive_array_temp[i, j, k] = positive_temp_entry - recomb_temp
                        negative_array_temp[i, j, k] = negative_temp_entry - recomb_temp

                        if k > self.no_z_electrode and k < (
                            self.no_z + self.no_z_electrode
                        ):
                            # sum over the recombination between the virtual electrodes
                            no_recombined_charge_carriers += recomb_temp

            f_steps_list[time_step] = (
                no_initialised_charge_carriers - no_recombined_charge_carriers
            ) / no_initialised_charge_carriers
            np.copyto(positive_array, positive_array_temp)
            np.copyto(negative_array, negative_array_temp)

        return f_steps_list
