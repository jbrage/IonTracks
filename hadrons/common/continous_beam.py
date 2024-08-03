from dataclasses import dataclass
from math import exp, sqrt
from random import Random

import numpy as np

from hadrons.common.generic_hadron_solver import GenericHadronSolver
from hadrons.utils.track_distribution import create_track_distribution


def get_continous_beam_pde_solver(base_solver_class: GenericHadronSolver):

    @dataclass
    class ContinousHadronSolver(base_solver_class):
        # TODO: dataclasses do not allow for non-default properties if the inherided class has default properties - not sure how to fix this yet
        fluence_rate: float = None  # [cm-^2/s]
        # TODO: add a seed param
        _random_generator: Random = Random(2137)

        @property
        def buffer_radius(self):
            return 10

        @property
        def no_xy(self) -> int:
            return (
                int(self.track_radius * self.n_track_radii / self.grid_spacing)
                + 2 * self.buffer_radius
            )

        @property
        def random_generator(self):
            return self._random_generator

        @property
        def xy_middle_idx(self):
            return int(self.no_xy / 2.0)

        @property
        def inner_radius(self):
            outer_radius = self.no_xy / 2.0

            return outer_radius - self.buffer_radius

        def __post_init__(self):
            super().__post_init__()
            self.simulation_time = self.computation_time_steps * self.dt
            self.number_of_tracks = int(
                self.fluence_rate * self.simulation_time * self.track_area
            )
            self.number_of_tracks = max(1, self.number_of_tracks)

            self.track_distribution = create_track_distribution(
                self.computation_time_steps,
                self.dt,
                self.number_of_tracks,
                self.separation_time_steps,
                self._random_generator,
            )

        def get_number_of_tracks(self, time_step: int) -> bool:
            return self.track_distribution[time_step]

        def get_random_xy_coordinate(self):
            return self.random_generator.random() * self.no_xy

        def get_random_coordinates(self):
            x = self.get_random_xy_coordinate()
            y = self.get_random_xy_coordinate()

            # check whether the point is inside the circle.
            # it is too time comsuming simply to set x=rand(0,1)*no_xy
            while (
                sqrt((x - self.xy_middle_idx) ** 2 + (y - self.xy_middle_idx) ** 2)
                > self.inner_radius
            ):
                x = self.get_random_xy_coordinate()
                y = self.get_random_xy_coordinate()

            return x, y

        def get_track_for_time_step(self, time_step: int):
            positive_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
            negative_array = np.zeros((self.no_xy, self.no_xy, self.no_z_with_buffer))
            no_initialised_charge_carriers = 0.0

            x, y = self.get_random_coordinates()

            for k in range(self.no_z_electrode, self.no_z + self.no_z_electrode):
                for i in range(self.no_xy):
                    for j in range(self.no_xy):
                        distance_from_center = (
                            sqrt((i - x) ** 2 + (j - y) ** 2) * self.grid_spacing
                        )
                        ion_density = self.Gaussian_factor * exp(
                            -(distance_from_center**2) / self.track_radius**2
                        )
                        positive_array[i, j, k] += ion_density
                        negative_array[i, j, k] += ion_density
                        # calculate the recombination only for charge carriers inside the circle
                        if (
                            sqrt(
                                (i - self.xy_middle_idx) ** 2
                                + (j - self.xy_middle_idx) ** 2
                            )
                            < self.inner_radius
                        ):
                            if time_step > self.separation_time_steps:
                                no_initialised_charge_carriers += ion_density

            return positive_array, negative_array, no_initialised_charge_carriers

    return ContinousHadronSolver
