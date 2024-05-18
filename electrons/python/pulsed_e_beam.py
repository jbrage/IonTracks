from __future__ import division
import numpy as np
import numpy.random as rnd
from copy import deepcopy
import numpy as np

from math import exp, sqrt, M_PI as pi, log, cos, sin


class pulsed_beam_PDEsolver:
    def pulsed_beam_PDEsolver(self, parameter_dic):
        """
        Define the parameters from Kanai (1998)
        """
        # self.W = 33.9  # eV/ion pair for air
        self.ion_mobility = (
            1.73  # cm s^-1 V^-1, averaged for positive and negative ions
        )
        self.ion_diff = 3.7e-2  # cm^2/s, averaged for positive and negative ions
        self.alpha = 1.60e-6  # cm^3/s, recombination constant

        self.electron_density_per_cm3 = parameter_dic[
            "elec_per_cm3"
        ]  # fluence-rate [/cm^2/s]
        self.voltage_V = parameter_dic[
            "voltage_V"
        ]  # [V/cm] magnitude of the electric field
        self.d_cm = parameter_dic["d_cm"]  # [cm] # electrode gap
        self.show_plot = parameter_dic["show_plot"]  # show frames of the simulation
        self.print_parameters = parameter_dic["print_parameters"]  # print parameters?

        """
        Define the grid size parameters
        """
        # self.LET_eV_cm = LET_keV_um*1e7
        self.r_cm = 0.002  # radius of the sampled cylinder
        self.area_cm2 = self.r_cm**2 * self.pi  # the "recombination" area
        self.unit_length_cm = 5e-4  # cm, size of every voxel length
        self.no_xy = int(
            2 * self.r_cm / self.unit_length_cm
        )  # no of voxels in xy-directions
        self.buffer_radius = 5
        self.no_xy += 2 * self.buffer_radius
        self.no_z = int(
            self.d_cm / self.unit_length_cm
        )  # number of elements in the z direction
        self.no_z_electrode = 5  # length of the electrode-buffer to ensure no ions drift through the array in one time step
        self.no_z_with_buffer = 2 * self.no_z_electrode + self.no_z
        # depending on the cluster/computer, the upper limit may be changed
        if (self.no_xy * self.no_xy * self.no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (self.no_xy * self.no_xy * self.no_z_with_buffer)
            )

        # find the middle of the arrays
        self.mid_xy_array = int(self.no_xy / 2.0)
        self.mid_z_array = int(self.no_z / 2.0)
        self.outer_radius = self.no_xy / 2.0
        self.inner_radius = self.outer_radius - self.buffer_radius

        self.no_figure_updates = 5  # update the figure this number of times
        # self.number_of_iterations = 1e2      # preallocate track coordinates

        """
        find a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        decreases and eventually damps out
        """
        self.dt = 1.0
        self.von_neumann_expression = False
        self.sx, sy, sz, cx, cy, cz
        self.Efield_V_cm = (
            self.voltage_V / self.d_cm
        )  # to ensure the same dt in all simulations
        while not von_neumann_expression:
            dt /= 1.01
            # as defined in the Deghan (2004) paper
            # sx = ion_diff*dt/(unit_length_cm**2)
            # sy = ion_diff*dt/(unit_length_cm**2)
            sx = (
                sy
            ) = 0.0  # uniform charge => no gradient driven diffusion in the xy plane
            sz = self.ion_diff * dt / (self.unit_length_cm**2)
            cx = cy = 0.0
            cz = self.ion_mobility * self.Efield_V_cm * dt / self.unit_length_cm
            # check von Neumann's criterion
            von_neumann_expression = (
                2 * (sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1
                and cx**2 * cy**2 * cz**2 <= 8 * sx * sy * sz
            )

        """
        Calculate the number of step required to drag the two charge carrier distributions apart
        along with the number of tracks to be uniformly distributed over the domain
        """
        self.separation_time_steps = int(
            d_cm / (2.0 * self.ion_mobility * self.Efield_V_cm * dt)
        )
        self.computation_time_steps = self.separation_time_steps * 3
        self.collection_time_s = 2 * self.separation_time_steps * dt

        computation_time_steps = self.separation_time_steps * 3
        self.simulation_time_s = computation_time_steps * dt
        self.positive_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        self.negative_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        self.positive_array_temp = np.zeros(
            (self.no_xy, self.no_xy, self.no_z_with_buffer)
        )
        self.negative_array_temp = np.zeros(
            (self.no_xy, self.no_xy, self.no_z_with_buffer)
        )
        self.no_recombined_charge_carriers = 0.0
        self.no_initialised_charge_carriers = 0.0

        """
        The following part is only used to plot the figure; consider to remove
        """
        self.MINVAL = 0.0
        self.MAXVAL = 0.0
        if self.show_plot:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            plt.close("all")
            fig = plt.figure()
            plt.ion()
            gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
            # define locations of subfigures in the grid
            ax1 = plt.subplot2grid((5, 3), (0, 0), rowspan=4)
            ax2 = plt.subplot2grid((5, 3), (0, 1), rowspan=4)
            ax3 = plt.subplot2grid((5, 3), (0, 2), rowspan=4)

            ax1.imshow(
                np.asarray(self.positive_array[mid_xy_array, :, :]).transpose(),
                vmin=self.MINVAL,
                vmax=MAXVAL,
            )
            ax2.imshow(
                self.positive_array[:, :, self.mid_xy_array],
                vmin=self.MINVAL,
                vmax=MAXVAL,
            )
            figure3 = ax3.imshow(
                self.positive_array[:, :, self.mid_xy_array],
                vmin=self.MINVAL,
                vmax=MAXVAL,
            )
            # adjust the 3 subfigures
            ax1.set_aspect("equal")
            ax1.axis("off")
            ax2.set_aspect("equal")
            ax2.axis("off")
            ax3.set_aspect("equal")
            ax3.axis("off")
            # fix colorbar
            cbar_ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.03])
            cb = fig.colorbar(
                figure3,
                cax=cbar_ax1,
                orientation="horizontal",
                label="Charge carrier density [per cm^3]",
            )
            cb.set_clim(vmin=self.MINVAL, vmax=MAXVAL)

        if self.print_parameters:
            print(
                "Simul. radius = %0.5g cm" % (self.inner_radius * self.unit_length_cm)
            )
            print(
                "Simul. area   = %0.5g cm^2"
                % ((self.inner_radius * self.unit_length_cm) ** 2 * pi)
            )
            print("Electric field = %s V/cm" % self.Efield_V_cm)
            print("Separation time = %3.2E s" % (self.separation_time_steps * dt))
            print("Simulation time = %3.2E s" % self.simulation_time_s)
            print("Time steps      = %3.2E" % computation_time_steps)
            print("Time step dt    = %3.2E s" % dt)
            print(
                "Number of voxels = %3.2E" % (self.no_xy**2 * self.no_z_with_buffer)
            )
            print("Number of pixels = %d (x = y directions)" % self.no_xy)
            print("Number of pixels = %d (z direction)" % self.no_z)

        """
        Create an array which include the number of track to be inserted for each time step
        The tracks are distributed uniformly in time
        """
        time_s = computation_time_steps * dt
        bins = np.arange(0.0, time_s + dt, dt)

        coordinate_counter = 0
        number_of_initialized_tracks = 0
        (
            x,
            y,
            f,
            distance_from_center,
            positive_temp_entry,
            negative_temp_entry,
            recomb_temp,
        ) = 0.0
        i, j, k, time_step

        f_step = np.zeros(computation_time_steps)

        delta_border = 2

        """
        Fill the array with the electron density according to the pulesed beam
        """
        for k in range(self.no_z_electrode, self.no_z + self.no_z_electrode):
            for i in range(delta_border, self.no_xy - delta_border):
                for j in range(delta_border, self.no_xy - delta_border):
                    self.positive_array[i, j, k] += self.electron_density_per_cm3
                    self.negative_array[i, j, k] += self.electron_density_per_cm3
                    if self.positive_array[i, j, k] > MAXVAL:
                        MAXVAL = self.positive_array[i, j, k]
                    no_initialised_charge_carriers += self.electron_density_per_cm3

        """
        Start the simulation by evovling the distribution one step at a time
        """
        for time_step in range(computation_time_steps):
            # update the figure
            if self.show_plot:
                update_figure_step = int(
                    computation_time_steps / self.no_figure_updates
                )
                if time_step % update_figure_step == 0:
                    print("Updated figure; %s" % time_step)
                    ax1.imshow(
                        np.asarray(
                            self.negative_array[:, self.mid_xy_array, :],
                            dtype=np.double,
                        ).transpose(),
                        vmin=self.MINVAL,
                        vmax=MAXVAL,
                    )
                    ax2.imshow(
                        np.asarray(
                            self.positive_array[:, self.mid_xy_array, :],
                            dtype=np.double,
                        ).transpose(),
                        vmin=self.MINVAL,
                        vmax=MAXVAL,
                    )
                    ax3.imshow(
                        np.asarray(
                            self.positive_array[:, :, self.mid_z_array], dtype=np.double
                        ),
                        vmin=self.MINVAL,
                        vmax=MAXVAL,
                    )
                    cb.set_clim(vmin=self.MINVAL, vmax=MAXVAL)
                    plt.pause(1e-3)

            # calculate the new densities and store them in temporary arrays
            for i in range(1, self.no_xy - 1):
                for j in range(1, self.no_xy - 1):
                    for k in range(1, self.no_z_with_buffer - 1):
                        # using the Lax-Wendroff scheme
                        positive_temp_entry = (
                            sz + cz * (cz + 1.0) / 2.0
                        ) * self.positive_array[i, j, k - 1]
                        positive_temp_entry += (
                            sz + cz * (cz - 1.0) / 2.0
                        ) * self.positive_array[i, j, k + 1]

                        positive_temp_entry += (
                            sy + cy * (cy + 1.0) / 2.0
                        ) * self.positive_array[i, j - 1, k]
                        positive_temp_entry += (
                            sy + cy * (cy - 1.0) / 2.0
                        ) * self.positive_array[i, j + 1, k]

                        positive_temp_entry += (
                            sx + cx * (cx + 1.0) / 2.0
                        ) * self.positive_array[i - 1, j, k]
                        positive_temp_entry += (
                            sx + cx * (cx - 1.0) / 2.0
                        ) * self.positive_array[i + 1, j, k]

                        positive_temp_entry += (
                            1.0 - cx * cx - cy * cy - cz * cz - 2.0 * (sx + sy + sz)
                        ) * self.positive_array[i, j, k]

                        # same for the negative charge carriers
                        negative_temp_entry = (
                            sz + cz * (cz + 1.0) / 2.0
                        ) * self.negative_array[i, j, k + 1]
                        negative_temp_entry += (
                            sz + cz * (cz - 1.0) / 2.0
                        ) * self.negative_array[i, j, k - 1]

                        negative_temp_entry += (
                            sy + cy * (cy + 1.0) / 2.0
                        ) * self.negative_array[i, j + 1, k]
                        negative_temp_entry += (
                            sy + cy * (cy - 1.0) / 2.0
                        ) * self.negative_array[i, j - 1, k]

                        negative_temp_entry += (
                            sx + cx * (cx + 1.0) / 2.0
                        ) * self.negative_array[i + 1, j, k]
                        negative_temp_entry += (
                            sx + cx * (cx - 1.0) / 2.0
                        ) * self.negative_array[i - 1, j, k]

                        negative_temp_entry += (
                            1.0 - cx * cx - cy * cy - cz * cz - 2.0 * (sx + sy + sz)
                        ) * self.negative_array[i, j, k]

                        # the recombination part
                        recomb_temp = (
                            self.alpha
                            * self.positive_array[i, j, k]
                            * self.negative_array[i, j, k]
                            * dt
                        )
                        self.positive_array_temp[i, j, k] = (
                            positive_temp_entry - recomb_temp
                        )
                        self.negative_array_temp[i, j, k] = (
                            negative_temp_entry - recomb_temp
                        )

                        if k > self.no_z_electrode and k < (
                            self.no_z + self.no_z_electrode
                        ):
                            # sum over the recombination between the virtual electrodes
                            no_recombined_charge_carriers += recomb_temp

            # update the positive and negative arrays
            for i in range(1, self.no_xy - 1):
                for j in range(1, self.no_xy - 1):
                    for k in range(1, self.no_z_with_buffer - 1):
                        self.positive_array[i, j, k] = self.positive_array_temp[i, j, k]
                        self.negative_array[i, j, k] = self.negative_array_temp[i, j, k]

        # return the collection efficiency
        f = (
            no_initialised_charge_carriers - no_recombined_charge_carriers
        ) / no_initialised_charge_carriers
        return f
