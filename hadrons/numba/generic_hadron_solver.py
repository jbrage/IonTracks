from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from hadrons.common.generic_hadron_solver import (
    GenericHadronSolver,
    create_sc_gradients,
)


def numba_calculate(
    computation_time_steps: int,
    get_number_of_tracks: Callable[[int], bool],
    get_track_for_time_step: Callable[[NDArray, NDArray], float],
    no_xy: int,
    no_z_with_buffer: int,
    s: NDArray,
    c: NDArray,
    alpha: float,
    dt: float,
):
    positive_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    negative_array = np.zeros((no_xy, no_xy, no_z_with_buffer))

    positive_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
    negative_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))

    no_recombined_charge_carriers = 0.0
    no_initialised_charge_carriers = 0.0

    f_steps_list = np.zeros(computation_time_steps)

    sc_pos, sc_neg, sc_center = create_sc_gradients(s, c)

    for time_step in range(computation_time_steps):
        # calculate the new densities and store them in temporary arrays

        for _ in range(get_number_of_tracks(time_step)):
            positive_track, negative_track, initialized_carriers = (
                get_track_for_time_step(time_step)
            )

            positive_array += positive_track
            negative_array += negative_track

            no_initialised_charge_carriers += initialized_carriers

        for i in range(1, no_xy - 1):
            for j in range(1, no_xy - 1):
                for k in range(1, no_z_with_buffer - 1):
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
                        alpha * positive_array[i, j, k] * negative_array[i, j, k] * dt
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

    f_steps_list[time_step] = (
        no_initialised_charge_carriers - no_recombined_charge_carriers
    ) / no_initialised_charge_carriers

    return f_steps_list


class NumbaHadronSolver(GenericHadronSolver, ABC):
    @abstractmethod
    def get_track_inserting_functions(
        self,
    ) -> Tuple[Callable[[int], bool], Callable[[NDArray, NDArray], float]]:
        pass

    def calculate(self):
        get_number_of_tracks, get_track_for_time_step = (
            self.get_track_inserting_functions()
        )

        f_steps_list = numba_calculate(
            self.computation_time_steps,
            get_number_of_tracks,
            get_track_for_time_step,
            self.no_xy,
            self.no_z_with_buffer,
            self.s,
            self.c,
            self.alpha,
            self.dt,
        )

        return f_steps_list
