from dataclasses import dataclass
from math import sqrt
from typing import NDArray, Tuple

import numpy as np

from hadrons.python.generic_hadron_solver import GenericHadronSolver


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
class InitialHadronSolver(GenericHadronSolver):

    def should_insert_track(self, time_step: int) -> bool:
        return time_step == 0

    def get_track_for_time_step(self):
        positive_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        negative_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
        no_initialised_charge_carriers = 0.0

        mid_xy_array = int(self.no_xy / 2.0)

        for i in range(self.no_xy):
            for j in range(self.no_xy):
                distance_from_center_cm = (
                    sqrt((i - mid_xy_array) ** 2 + (j - mid_xy_array) ** 2)
                    * self.grid_spacing
                )
                ion_density = self.RDD_function(distance_from_center_cm)
                no_initialised_charge_carriers += self.no_z * ion_density
                positive_array[
                    i, j, self.no_z_electrode : (self.no_z + self.no_z_electrode)
                ] += ion_density
                negative_array[
                    i, j, self.no_z_electrode : (self.no_z + self.no_z_electrode)
                ] += ion_density

        return positive_array, negative_array, no_initialised_charge_carriers
