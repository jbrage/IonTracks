from __future__ import division

from generic_electron_solver import GenericElectronSolver


class ContinousBeamPDEsolver(GenericElectronSolver):
    def get_initialised_charge_carriers_after_beam(self):
        electron_density_per_cm3_s = self.electron_density_per_cm3 * self.dt

        initialised_charge_carriers = (
            electron_density_per_cm3_s
            * (self.no_xy - 2 * self.delta_border) ** 2
            * self.no_z
        )

        return initialised_charge_carriers

    def update_electron_density_arrays_after_beam(
        self,
        positive_array,
        negative_array,
    ):
        electron_density_per_cm3_s = self.electron_density_per_cm3 * self.dt

        positive_array[
            self.delta_border : (self.no_xy - self.delta_border),
            self.delta_border : (self.no_xy - self.delta_border),
            self.no_z_electrode : (self.no_z + self.no_z_electrode),
        ] += electron_density_per_cm3_s

        negative_array[
            self.delta_border : (self.no_xy - self.delta_border),
            self.delta_border : (self.no_xy - self.delta_border),
            self.no_z_electrode : (self.no_z + self.no_z_electrode),
        ] += electron_density_per_cm3_s

    def should_simulate_beam_for_time_step(self, time_step):
        return True
