from math import exp, sqrt

import numpy as np
from numba import njit

from hadrons.common.continous_beam import get_continous_beam_pde_solver
from hadrons.numba.generic_hadron_solver import NumbaHadronSolver

BaseContinuousBeamPDESolver = get_continous_beam_pde_solver(NumbaHadronSolver)


class NumbaContinousPDESolver(BaseContinuousBeamPDESolver):
    def get_track_inserting_functions(self):
        # closure to give numba access to the class properties without using `self`
        track_distribution = self.track_distribution
        no_xy = self.no_xy
        no_z_with_buffer = self.no_z_with_buffer
        no_z_electrode = self.no_z_electrode
        no_z = self.no_z
        grid_spacing = self.grid_spacing
        Gaussian_factor = self.Gaussian_factor
        track_radius = self.track_radius
        xy_middle_idx = self.xy_middle_idx
        inner_radius = self.inner_radius
        separation_time_steps = self.separation_time_steps
        random_generator = self.random_generator

        # TODO: probably could be an util
        @njit
        def get_random_xy_coordinate():
            return random_generator.random() * no_xy

        @njit
        def get_random_coordinates():
            x = get_random_xy_coordinate()
            y = get_random_xy_coordinate()

            # check whether the point is inside the circle.
            # it is too time comsuming simply to set x=rand(0,1)*no_xy
            while (
                sqrt((x - xy_middle_idx) ** 2 + (y - xy_middle_idx) ** 2) > inner_radius
            ):
                x = get_random_xy_coordinate()
                y = get_random_xy_coordinate()

            return x, y

        @njit
        def get_number_of_tracks(time_step: int) -> int:
            return track_distribution[time_step]

        @njit
        def get_track_for_time_step(time_step: int):
            positive_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
            negative_array = np.zeros((no_xy, no_xy, no_z_with_buffer))
            no_initialised_charge_carriers = 0.0

            x, y = get_random_coordinates()

            for k in range(no_z_electrode, no_z + no_z_electrode):
                for i in range(no_xy):
                    for j in range(no_xy):
                        distance_from_center = (
                            sqrt((i - x) ** 2 + (j - y) ** 2) * grid_spacing
                        )
                        ion_density = Gaussian_factor * exp(
                            -(distance_from_center**2) / track_radius**2
                        )
                        positive_array[i, j, k] += ion_density
                        negative_array[i, j, k] += ion_density
                        # calculate the recombination only for charge carriers inside the circle
                        if (
                            sqrt((i - xy_middle_idx) ** 2 + (j - xy_middle_idx) ** 2)
                            < inner_radius
                        ):
                            if time_step > separation_time_steps:
                                no_initialised_charge_carriers += ion_density

            return positive_array, negative_array, no_initialised_charge_carriers

        return get_number_of_tracks, get_track_for_time_step
