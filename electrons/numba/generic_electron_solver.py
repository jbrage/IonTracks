from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from electrons.common.generic_electron_solver import (
    GenericElectronSolver as PythonGenericElectronSolver,
)
from electrons.common.generic_electron_solver import create_sc_gradients

numba_create_sc_gradients = njit(create_sc_gradients)


@njit
def numba_calculate(
    computation_time_steps: int,
    should_simulate_beam_for_time_step: Callable[[int], bool],
    simulate_beam: Callable[[NDArray, NDArray], float],
    no_xy: int,
    no_z_with_buffer: int,
    s: NDArray,
    c: NDArray,
    alpha: float,
    dt: float,
    no_z: int,
    no_z_electrode: int,
) -> NDArray:
    positive_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    negative_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
    positive_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
    negative_array_temp = np.zeros((no_xy, no_xy, no_z_with_buffer))
    no_recombined_charge_carriers = 0.0
    no_initialised_charge_carriers = 0.0

    f_steps_list = np.zeros(computation_time_steps)

    sc_neg, sc_pos, sc_center = numba_create_sc_gradients(s, c)

    for time_step in range(computation_time_steps):

        """
        Refill the array with the electron density each time step
        """
        if should_simulate_beam_for_time_step(time_step):
            no_initialised_step = simulate_beam(
                positive_array,
                negative_array,
            )
        else:
            no_initialised_step = 0.0

        no_initialised_charge_carriers += no_initialised_step

        # calculate the new densities and store them in temporary arrays
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

                    if k > no_z_electrode and k < (no_z + no_z_electrode):
                        # sum over the recombination between the virtual electrodes
                        no_recombined_charge_carriers += recomb_temp

        f_steps_list[time_step] = (
            no_initialised_charge_carriers - no_recombined_charge_carriers
        ) / no_initialised_charge_carriers
        positive_array[:] = positive_array_temp
        negative_array[:] = negative_array_temp

    return f_steps_list


class NumbaGenericElectronSolver(PythonGenericElectronSolver, ABC):

    @abstractmethod
    def get_beam_simulation_functions(
        self,
    ) -> Tuple[Callable[[int], bool], Callable[[NDArray, NDArray], float]]:
        pass

    def calculate(self):
        should_simulate_beam_for_time_step, simulate_beam = (
            self.get_beam_simulation_functions()
        )

        f_steps_list = numba_calculate(
            self.computation_time_steps,
            should_simulate_beam_for_time_step,
            simulate_beam,
            self.no_xy,
            self.no_z_with_buffer,
            self.s,
            self.c,
            self.alpha,
            self.dt,
            self.no_z,
            self.no_z_electrode,
        )

        return f_steps_list
