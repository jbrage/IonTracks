from __future__ import division
import numpy as np

from math import M_PI as pi


class pulsed_beam_PDEsolver:
    def __init__(self, parameter_dic):
        """
        Define the parameters from Kanai (1998)
        """
        # W = 33.9  # eV/ion pair for air
        ion_mobility = 1.73  # cm s^-1 V^-1, averaged for positive and negative ions
        ion_diff = 3.7e-2  # cm^2/s, averaged for positive and negative ions
        alpha = 1.60e-6  # cm^3/s, recombination constant

        electron_density_per_cm3 = parameter_dic[
            "elec_per_cm3"
        ]  # fluence-rate [/cm^2/s]
        voltage_V = parameter_dic["voltage_V"]  # [V/cm] magnitude of the electric field
        d_cm = parameter_dic["d_cm"]  # [cm] # electrode gap

        """
        Define the grid size parameters
        """
        # LET_eV_cm = LET_keV_um*1e7
        r_cm = 0.002  # radius of the sampled cylinder
        unit_length_cm = 5e-4  # cm, size of every voxel length
        no_xy = int(2 * r_cm / unit_length_cm)  # no of voxels in xy-directions
        buffer_radius = 5
        no_xy += 2 * buffer_radius
        no_z = int(d_cm / unit_length_cm)  # number of elements in the z direction
        no_z_electrode = 5  # length of the electrode-buffer to ensure no ions drift through the array in one time step
        no_z_with_buffer = 2 * no_z_electrode + no_z
        # depending on the cluster/computer, the upper limit may be changed
        if (no_xy * no_xy * no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (no_xy * no_xy * no_z_with_buffer)
            )

        """
        find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        decreases and eventually damps out
        """
        dt = 1.0
        von_neumann_expression = False
        sx, sy, sz, cx, cy, cz
        Efield_V_cm = voltage_V / d_cm  # to ensure the same dt in all simulations
        while not von_neumann_expression:
            dt /= 1.01
            # as defined in the Deghan (2004) paper
            # sx = ion_diff*dt/(unit_length_cm**2)
            # sy = ion_diff*dt/(unit_length_cm**2)
            sx = sy = (
                0.0  # uniform charge => no gradient driven diffusion in the xy plane
            )
            sz = ion_diff * dt / (unit_length_cm**2)
            cx = cy = 0.0
            cz = ion_mobility * Efield_V_cm * dt / unit_length_cm
            # check von Neumann's criterion
            von_neumann_expression = (
                2 * (sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1
                and cx**2 * cy**2 * cz**2 <= 8 * sx * sy * sz
            )

        """
        Calculate the number of step required to drag the two charge carrier distributions apart
        along with the number of tracks to be uniformly distributed over the domain
        """
        separation_time_steps = int(d_cm / (2.0 * ion_mobility * Efield_V_cm * dt))

        self.computation_time_steps = separation_time_steps * 3
        self.no_xy = no_xy
        self.no_z_with_buffer = no_z_with_buffer
        self.no_z_electrode = no_z_electrode
        self.dt = dt
        self.no_z = no_z
        self.electron_density_per_cm3 = electron_density_per_cm3
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.alpha = alpha

    def calculate(self):
        # refering to each param with a `self.` prefix makes the code less readable so we unpack them here
        # this also prevents accidental mutations to the params - subsequent calls to `calculate` won't have side-effects
        no_xy = self.no_xy
        no_z_with_buffer = self.no_z_with_buffer
        computation_time_steps = self.computation_time_steps
        no_z_electrode = self.no_z_electrode
        dt = self.dt
        no_z = self.no_z
        electron_density_per_cm3 = self.electron_density_per_cm3
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

        delta_border = 2

        """
        Fill the array with the electron density according to the pulesed beam
        """
        for k in range(no_z_electrode, no_z + no_z_electrode):
            for i in range(delta_border, no_xy - delta_border):
                for j in range(delta_border, no_xy - delta_border):
                    positive_array[i, j, k] += electron_density_per_cm3
                    negative_array[i, j, k] += electron_density_per_cm3
                    if positive_array[i, j, k] > MAXVAL:
                        MAXVAL = positive_array[i, j, k]
                    no_initialised_charge_carriers += electron_density_per_cm3

        """
        Start the simulation by evovling the distribution one step at a time
        """
        for _ in range(computation_time_steps):
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
