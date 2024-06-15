from __future__ import division

from electrons.common.generic_electron_solver import GenericElectronSolver


def get_pulsed_beam_pde_solver(base_solver_class: GenericElectronSolver):

    class PulsedBeamPDEsolver(base_solver_class):

        def get_initialised_charge_carriers_after_beam(self):
            initialised_charge_carriers = (
                self.electron_density_per_cm3
                * (self.no_xy - 2 * self.delta_border) ** 2
                * self.no_z
            )

            return initialised_charge_carriers

        def update_electron_density_arrays_after_beam(
            self,
            positive_array,
            negative_array,
        ):
            positive_array[
                self.delta_border : (self.no_xy - self.delta_border),
                self.delta_border : (self.no_xy - self.delta_border),
                self.no_z_electrode : (self.no_z + self.no_z_electrode),
            ] += self.electron_density_per_cm3

            negative_array[
                self.delta_border : (self.no_xy - self.delta_border),
                self.delta_border : (self.no_xy - self.delta_border),
                self.no_z_electrode : (self.no_z + self.no_z_electrode),
            ] += self.electron_density_per_cm3

        def should_simulate_beam_for_time_step(self, time_step):
            return time_step == 0

    return PulsedBeamPDEsolver
