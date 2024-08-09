from dataclasses import dataclass
from math import sqrt

import numpy as np

from hadrons.common.generic_hadron_solver import GenericHadronSolver


def get_initial_recombination_pde_solver(base_solver_class: GenericHadronSolver):
    @dataclass
    class InitialHadronSolver(base_solver_class):

        def get_number_of_tracks(self, time_step: int) -> bool:
            if time_step == 0:
                return 1
            return 0

        def get_track_for_time_step(self, _):
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

        def should_count_recombined_charge_carriers(
            self, time_step: int, x: float, y: float, z: float
        ) -> bool:
            # For initial recombination we can always count the charge carriers
            return True

    return InitialHadronSolver
