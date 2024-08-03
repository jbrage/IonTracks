from __future__ import division

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def create_sc_gradients(
    s: NDArray,
    c: NDArray,
) -> Tuple[NDArray, NDArray, float]:
    sc_pos = np.array(
        [
            s[0] + c[0] * (c[0] + 1.0) / 2.0,
            s[1] + c[1] * (c[1] + 1.0) / 2.0,
            s[2] + c[2] * (c[2] + 1.0) / 2.0,
        ]
    )

    sc_neg = np.array(
        [
            s[0] + c[0] * (c[0] - 1.0) / 2.0,
            s[1] + c[1] * (c[1] - 1.0) / 2.0,
            s[2] + c[2] * (c[2] - 1.0) / 2.0,
        ]
    )

    sc_center = (
        1.0 - c[0] * c[0] - c[1] * c[1] - c[2] * c[2] - 2.0 * (s[0] + s[1] + s[2])
    )

    return sc_pos, sc_neg, sc_center


@dataclass
class GenericElectronSolver(ABC):
    # Simulation parameters
    electron_density_per_cm3: float  # fluence-rate [/cm^2/s]
    voltage: float  # [V/cm] magnitude of the electric field
    electrode_gap: float  # [cm]
    grid_spacing: float
    ion_mobility: float = 1.73  # cm s^-1 V^-1, averaged for positive and negative ions
    ion_diff: float = 3.7e-2  # cm^2/s, averaged for positive and negative ions
    alpha: float = 1.60e-6  # cm^3/s, recombination constant
    r_cm: float = 0.002  # radius of the sampled cylinder
    buffer_radius: int = 5
    no_z_electrode: int = (
        5  # length of the electrode-buffer to ensure no ions drift through the array in one time step
    )
    dt: float = 1.0

    # Derived properties
    @property
    def no_xy(self) -> int:
        """Number of voxels om the xy-directions"""

        no_xy = int(2 * self.r_cm / self.grid_spacing)
        no_xy += 2 * self.buffer_radius
        return no_xy

    @property
    def no_z(self) -> int:
        """Number of voxels om the z-direction"""

        return int(self.electrode_gap / self.grid_spacing)

    @property
    def no_z_with_buffer(self) -> int:
        return 2 * self.no_z_electrode + self.no_z

    @property
    def electric_field(self) -> float:
        return self.voltage / self.electrode_gap

    @property
    def separation_time_steps(self) -> int:
        return int(
            self.electrode_gap
            / (2.0 * self.ion_mobility * self.electric_field * self.dt)
        )

    @property
    def computation_time_steps(self) -> int:

        return self.separation_time_steps * 3

    @property
    def delta_border(self) -> int:
        return 2

    def von_neumann_expression(self) -> Tuple[float, NDArray, NDArray]:
        """
        Finds a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        decreases and eventually damps out
        """

        von_neumann_expression = False
        dt = 1.0

        # initialize the coefficients
        sx = sy = sz = cx = cy = cz = 0.0

        while not von_neumann_expression:
            dt /= 1.01
            # as defined in the Deghan (2004) paper

            # we leave the coeficents as 0 in the xy plane
            # sx = ion_diff*dt/(self.grid_spacing_cm**2)
            # sy = ion_diff*dt/(self.grid_spacing_cm**2)

            sz = self.ion_diff * dt / (self.grid_spacing**2)
            cz = self.ion_mobility * self.electric_field * dt / self.grid_spacing
            # check von Neumann's criterion
            criterion_1 = 2 * (sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1

            criterion_2 = cx**2 * cy**2 * cz**2 <= 8 * sx * sy * sz

            von_neumann_expression = criterion_1 and criterion_2

        return dt, np.array([sx, sy, sz]), np.array([cx, cy, cz])

    def __post_init__(
        self,
    ):
        # depending on the cluster/computer, the upper limit may be changed
        if (self.no_xy * self.no_xy * self.no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (self.no_xy * self.no_xy * self.no_z_with_buffer)
            )

        self.dt, self.s, self.c = self.von_neumann_expression()

    @abstractmethod
    def should_simulate_beam_for_time_step(self, time_step: int) -> bool:
        pass

    @abstractmethod
    def update_electron_density_arrays_after_beam(
        self, positive_array: NDArray, negative_array: NDArray
    ) -> None:
        pass

    @abstractmethod
    def get_initialised_charge_carriers_after_beam() -> float:
        pass

    def simulate_beam(self, positive_array: NDArray, negative_array: NDArray):
        self.update_electron_density_arrays_after_beam(positive_array, negative_array)

        return self.get_initialised_charge_carriers_after_beam()

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

        f_steps_list = np.zeros(self.computation_time_steps)

        sc_pos, sc_neg, sc_center = create_sc_gradients(self.s, self.c)

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
                        positive_temp_entry = 0

                        positive_temp_entry += sc_pos[0] * positive_array[i - 1, j, k]
                        positive_temp_entry += sc_neg[0] * positive_array[i + 1, j, k]

                        positive_temp_entry += sc_pos[1] * positive_array[i, j - 1, k]
                        positive_temp_entry += sc_neg[1] * positive_array[i, j + 1, k]

                        positive_temp_entry += sc_pos[2] * positive_array[i, j, k - 1]
                        positive_temp_entry += sc_neg[2] * positive_array[i, j, k + 1]

                        positive_temp_entry += sc_center * positive_array[i, j, k]

                        # same for the negative charge carriers
                        negative_temp_entry = 0

                        negative_temp_entry += sc_pos[0] * negative_array[i + 1, j, k]
                        negative_temp_entry += sc_neg[0] * negative_array[i - 1, j, k]

                        negative_temp_entry += sc_pos[1] * negative_array[i, j + 1, k]
                        negative_temp_entry += sc_neg[1] * negative_array[i, j - 1, k]

                        negative_temp_entry += sc_pos[2] * negative_array[i, j, k + 1]
                        negative_temp_entry += sc_neg[2] * negative_array[i, j, k - 1]

                        negative_temp_entry += sc_center * negative_array[i, j, k]

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
            positive_array[:] = positive_array_temp
            negative_array[:] = negative_array_temp

        return f_steps_list
