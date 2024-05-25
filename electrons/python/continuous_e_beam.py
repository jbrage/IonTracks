from __future__ import division
import numpy as np

from generic_electron_solver import GenericElectronSolver


class ContinousBeamPDEsolver(GenericElectronSolver):
    def get_electron_density_after_beam(
        self,
        positive_array,
        negative_array,
        time_step,
    ):
        delta_border = 2
        electron_density_per_cm3_s = self.electron_density_per_cm3 * self.dt
        initialised_charge_carriers = 0.0

        for k in range(self.no_z_electrode, self.no_z + self.no_z_electrode):
            for i in range(delta_border, self.no_xy - delta_border):
                for j in range(delta_border, self.no_xy - delta_border):
                    positive_array[i, j, k] += electron_density_per_cm3_s
                    negative_array[i, j, k] += electron_density_per_cm3_s
                    initialised_charge_carriers += electron_density_per_cm3_s

        return positive_array, negative_array, initialised_charge_carriers
