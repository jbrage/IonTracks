from __future__ import division


from generic_electron_solver import GenericElectronSolver


class PulsedBeamPDEsolver(GenericElectronSolver):
    def get_electron_density_after_beam(
        self, positive_array, negative_array, time_step
    ):
        delta_border = 2
        initialised_charge_carriers = 0.0

        # in the pulsed scenario there is only one beam at time 0
        if time_step == 0:
            for k in range(self.no_z_electrode, self.no_z + self.no_z_electrode):
                for i in range(delta_border, self.no_xy - delta_border):
                    for j in range(delta_border, self.no_xy - delta_border):
                        positive_array[i, j, k] += self.electron_density_per_cm3
                        negative_array[i, j, k] += self.electron_density_per_cm3
                        initialised_charge_carriers += self.electron_density_per_cm3

        return positive_array, negative_array, initialised_charge_carriers
