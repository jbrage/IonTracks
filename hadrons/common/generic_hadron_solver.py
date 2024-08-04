from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import cos, exp, log, pi, sin
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from hadrons.utils.common import calculate_track_radius, get_LET_per_um

from ..common_properties import (
    W,
    air_density_g_cm3,
    alpha,
    ion_diff,
    ion_mobility,
    water_density_g_cm3,
)
from ..geiss_utils import Geiss_r_max, Geiss_RRD_cm


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
class GenericHadronSolver(ABC):
    voltage: float  # [V/cm] magnitude of the electric field
    IC_angle: float  # [rad]
    electrode_gap: float  # [cm]
    energy: float  # [MeV/u]
    RDD_model: Literal["Gauss", "Geiss"] = "Gauss"
    grid_spacing: float = 3e-4  # [cm]
    # TODO: Narrow this type down
    particle: str = "proton"
    no_z_electrode: int = (
        4  # length of the electrode-buffer to ensure no ions drift through the array in one time step
    )
    n_track_radii: int = 6  # scaling factor to determine the grid width
    dt: float = 1.0

    @property
    def LET_per_um(self) -> float:
        return get_LET_per_um(self.energy, self.particle)

    @property
    def LET_per_cm(self) -> float:
        return self.LET_per_um * 1e7

    @property
    def a0(self) -> float:
        density_ratio = water_density_g_cm3 / air_density_g_cm3
        return 8.0 * 1e-7 * density_ratio

    @property
    def r_max(self) -> float:
        return Geiss_r_max(self.energy, air_density_g_cm3)

    @property
    def track_radius(self) -> float:
        return calculate_track_radius(self.LET_per_um)

    @property
    def track_area(self) -> float:
        return pi * self.track_radius**2

    @property
    def Gaussian_factor(self) -> float:
        N0 = self.LET_per_cm / W
        return N0 / self.track_area

    @property
    def no_xy(self) -> int:
        return int(self.track_radius * self.n_track_radii / self.grid_spacing)

    @property
    def no_z(self) -> int:
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
            self.electrode_gap / (2.0 * ion_mobility * self.electric_field * self.dt)
        )

    @property
    def computation_time_steps(self) -> int:

        return self.separation_time_steps * 2

    @property
    def RDD_function(self):
        if self.RDD_model == "Gauss":
            return lambda r_cm: self.Gaussian_factor * exp(
                -(r_cm**2) / self.track_radius**2
            )
        elif self.RDD_model == "Geiss":
            c = self.LET_per_cm / (pi * W) * (1 / (1 + 2 * log(self.r_max / self.a0)))
            return lambda r_cm: Geiss_RRD_cm(r_cm, c, self.a0, self.r_max)
        else:
            raise ValueError(
                f"Invalid RDD model: {self.RDD_model}. Must be 'Gauss' or 'Geiss'."
            )

    def von_neumann_expression(self) -> Tuple[float, NDArray, NDArray]:
        """
        Finds a time step dt which fulfils the von Neumann criterion, i.e. ensures the numericl error does not increase but
        decreases and eventually damps out
        """

        von_neumann_expression = False
        dt = 1.0

        while not von_neumann_expression:
            dt /= 1.01
            # as defined in the Deghan (2004) paper
            sx = ion_diff * dt / (self.grid_spacing**2)
            sy = sz = sx

            cx = (
                ion_mobility
                * self.electric_field
                * dt
                / self.grid_spacing
                * sin(self.IC_angle)
            )
            cy = 0
            cz = (
                ion_mobility
                * self.electric_field
                * dt
                / self.grid_spacing
                * cos(self.IC_angle)
            )

            criterion_1 = 2 * (sx + sy + sz) + cx**2 + cy**2 + cz**2 <= 1

            criterion_2 = cx**2 * cy**2 * cz**2 <= 8 * sx * sy * sz

            von_neumann_expression = criterion_1 and criterion_2

        return dt, np.array([sx, sy, sz]), np.array([cx, cy, cz])

    def __post_init__(self):
        # depending on the cluster/computer, the upper limit may be changed
        if (self.no_xy * self.no_xy * self.no_z) > 1e8:
            raise ValueError(
                "Too many elements in the array: %i"
                % (self.no_xy * self.no_xy * self.no_z_with_buffer)
            )

        self.dt, self.s, self.c = self.von_neumann_expression()

    @abstractmethod
    def get_number_of_tracks(self, time_step: int) -> int:
        pass

    @abstractmethod
    def get_track_for_time_step(
        self, time_step: int
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        pass

    def calculate(self):
        positive_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        negative_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))

        positive_array_temp = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        negative_array_temp = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))

        no_recombined_charge_carriers = 0.0
        no_initialised_charge_carriers = 0.0

        f_steps_list = np.zeros(self.computation_time_steps)

        sc_pos, sc_neg, sc_center = create_sc_gradients(self.s, self.c)

        for time_step in tqdm(
            range(self.computation_time_steps), desc="Calculating..."
        ):
            # calculate the new densities and store them in temporary arrays

            for _ in range(self.get_number_of_tracks(time_step)):
                positive_track, negative_track, initialized_carriers = (
                    self.get_track_for_time_step(time_step)
                )

                positive_array += positive_track
                negative_array += negative_track

                no_initialised_charge_carriers += initialized_carriers

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
                            alpha
                            * positive_array[i, j, k]
                            * negative_array[i, j, k]
                            * self.dt
                        )

                        positive_array_temp[i, j, k] = positive_temp_entry - recomb_temp
                        negative_array_temp[i, j, k] = negative_temp_entry - recomb_temp

                        no_recombined_charge_carriers += recomb_temp

            # update the positive and negative arrays

            # TODO: check if copying the first element in each dimension is necessary
            # positive_array[1:-1, 1:-1, 1:-1] = positive_array_temp[1:-1, 1:-1, 1:-1]
            # negative_array[1:-1, 1:-1, 1:-1] = negative_array_temp[1:-1, 1:-1, 1:-1]

            positive_array[:] = positive_array_temp[:]
            negative_array[:] = negative_array_temp[:]

            print(no_initialised_charge_carriers, no_recombined_charge_carriers)

            # calculate the fraction of charge carriers that have not recombined, if no charge carriers have been initialised, set the fraction to 1
            if no_initialised_charge_carriers != 0:
                f_steps_list[time_step] = (
                    no_initialised_charge_carriers - no_recombined_charge_carriers
                ) / no_initialised_charge_carriers
            else:
                f_steps_list[time_step] = 1

        return f_steps_list
