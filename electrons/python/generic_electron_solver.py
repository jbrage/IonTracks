from __future__ import division
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Tuple
import numpy as np


def von_neumann_expression(
    dt, ion_diff, grid_spacing_cm, ion_mobility, Efield_V_cm
) -> None:
    """
    Finds a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
    decreases and eventually damps out
    """
    von_neumann_expression = False

    while not von_neumann_expression:
        dt /= 1.01
        # as defined in the Deghan (2004) paper
        # sx = ion_diff*dt/(self.grid_spacing_cm**2)
        # sy = ion_diff*dt/(self.grid_spacing_cm**2)
        sx = sy = 0.0  # uniform charge => no gradient driven diffusion in the xy plane

        sz = ion_diff * dt / (grid_spacing_cm**2)
        cx = cy = 0.0
        cz = ion_mobility * Efield_V_cm * dt / grid_spacing_cm
        # check von Neumann's criterion
        criterion_1 = 2 * (sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1

        criterion_2 = cx**2 * cy**2 * cz**2 <= 8 * sx * sy * sz

        von_neumann_expression = criterion_1 and criterion_2

    return dt, sx, sy, sz, cx, cy, cz


@dataclass
class GenericElectronSolver(ABC):
    # Simulation parameters
    electron_density_per_cm3: float  # fluence-rate [/cm^2/s]
    voltage_V: float  # [V/cm] magnitude of the electric field
    electrode_gap: float  # [cm] # electrode gap
    grid_spacing_cm: float
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

        no_xy = int(2 * self.r_cm / self.grid_spacing_cm)
        no_xy += 2 * self.buffer_radius
        return no_xy

    @property
    def no_z(self) -> int:
        """Number of voxels om the z-direction"""

        return int(self.electrode_gap / self.grid_spacing_cm)

    @property
    def no_z_with_buffer(self) -> int:
        return 2 * self.no_z_electrode + self.no_z

    @property
    def Efield_V_cm(self) -> float:
        return self.voltage_V / self.electrode_gap

    @property
    def separation_time_steps(self) -> int:
        """
        Number of step required to drag the two charge carrier distributions apart
        along with the number of tracks to be uniformly distributed over the domain
        """
        return int(
            self.electrode_gap / (2.0 * self.ion_mobility * self.Efield_V_cm * self.dt)
        )

    @property
    def computation_time_steps(self) -> int:
        return self.separation_time_steps * 3

    @property
    def delta_border(self) -> int:
        return 2

    def __post_init__(
        self,
    ):
        # depending on the cluster/computer, the upper limit may be changed
        if (self.no_xy * self.no_xy * self.no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (self.no_xy * self.no_xy * self.no_z_with_buffer)
            )

        self.dt, self.sx, self.sy, self.sz, self.cx, self.cy, self.cz = (
            von_neumann_expression(
                self.dt,
                self.ion_diff,
                self.grid_spacing_cm,
                self.ion_mobility,
                self.Efield_V_cm,
            )
        )

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
