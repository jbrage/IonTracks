from __future__ import division
import numpy as np

from generic_electron_solver import GenericElectronSolver


class PulsedBeamPDEsolver(GenericElectronSolver):
    unit_length_cm = 6e-4  # cm, size of every voxel length

    def get_electron_density_after_beam(
        self,
        positive_array,
        negative_array,
        no_initialised_charge_carriers,
        step_initialized,
    ):
        delta_border = 2
        electron_density_per_cm3_s = self.electron_density_per_cm3 * self.dt

        for k in range(self.no_z_electrode, self.no_z + self.no_z_electrode):
            for i in range(delta_border, self.no_xy - delta_border):
                for j in range(delta_border, self.no_xy - delta_border):

                    positive_array[i, j, k] += electron_density_per_cm3_s
                    negative_array[i, j, k] += electron_density_per_cm3_s
                    if positive_array[i, j, k] > MAXVAL:
                        MAXVAL = positive_array[i, j, k]
                    no_initialised_charge_carriers += electron_density_per_cm3_s
                    step_initialized += electron_density_per_cm3_s

        return (
            positive_array,
            negative_array,
            no_initialised_charge_carriers,
            step_initialized,
        )

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

        positive_temp_entry, negative_temp_entry, recomb_temp = 0.0

        step_recombined, step_initialized

        f_steps_list = np.zeros(computation_time_steps)

        """
        Start the simulation by evolving the distribution one step at a time
        """
        for time_step in range(computation_time_steps):

            step_recombined = 0.0
            step_initialized = 0.0

            """
            Refill the array with the electron density each time step
            """
            (
                positive_array,
                negative_array,
                no_initialised_charge_carriers,
                step_initialized,
            ) = self.get_electron_density_after_beam(
                positive_array,
                negative_array,
                no_initialised_charge_carriers,
                step_initialized,
            )

            # calculate the new densities and store them in temporary arrays
            for i in range(1, no_xy - 1):
                for j in range(1, no_xy - 1):
                    for k in range(1, no_z_with_buffer - 1):
                        # using the Lax-Wendroff scheme
                        positive_temp_entry = (
                            sz + cz * (cz + 1.0) / 2.0
                        ) * positive_array[i, j, k - 1]
                        positive_temp_entry += (
                            sz + cz * (cz - 1.0) / 2.0
                        ) * positive_array[i, j, k + 1]

                        positive_temp_entry += (
                            sy + cy * (cy + 1.0) / 2.0
                        ) * positive_array[i, j - 1, k]
                        positive_temp_entry += (
                            sy + cy * (cy - 1.0) / 2.0
                        ) * positive_array[i, j + 1, k]

                        positive_temp_entry += (
                            sx + cx * (cx + 1.0) / 2.0
                        ) * positive_array[i - 1, j, k]
                        positive_temp_entry += (
                            sx + cx * (cx - 1.0) / 2.0
                        ) * positive_array[i + 1, j, k]

                        positive_temp_entry += (
                            1.0 - cx * cx - cy * cy - cz * cz - 2.0 * (sx + sy + sz)
                        ) * positive_array[i, j, k]

                        # same for the negative charge carriers
                        negative_temp_entry = (
                            sz + cz * (cz + 1.0) / 2.0
                        ) * negative_array[i, j, k + 1]
                        negative_temp_entry += (
                            sz + cz * (cz - 1.0) / 2.0
                        ) * negative_array[i, j, k - 1]

                        negative_temp_entry += (
                            sy + cy * (cy + 1.0) / 2.0
                        ) * negative_array[i, j + 1, k]
                        negative_temp_entry += (
                            sy + cy * (cy - 1.0) / 2.0
                        ) * negative_array[i, j - 1, k]

                        negative_temp_entry += (
                            sx + cx * (cx + 1.0) / 2.0
                        ) * negative_array[i + 1, j, k]
                        negative_temp_entry += (
                            sx + cx * (cx - 1.0) / 2.0
                        ) * negative_array[i - 1, j, k]

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
                            step_recombined += recomb_temp

            f_steps_list[time_step] = (
                step_initialized - step_recombined
            ) / step_initialized

            # update the positive and negative arrays
            for i in range(1, no_xy - 1):
                for j in range(1, no_xy - 1):
                    for k in range(1, no_z_with_buffer - 1):
                        positive_array[i, j, k] = positive_array_temp[i, j, k]
                        negative_array[i, j, k] = negative_array_temp[i, j, k]

        # ---- not necesarily sure what this is ----
        # charge_collection_df = pd.DataFrame()
        # charge_collection_df["f"] = f_steps_list
        # charge_collection_df["ks"] = 1 / f_steps_list
        # charge_collection_df["time_s"] = np.arange(0, len(f_steps_list) * dt, dt)
        # charge_collection_df["time_us"] = charge_collection_df["time_s"] * 1e6
        # return charge_collection_df
        # ------------------------------------------

        return f_steps_list
