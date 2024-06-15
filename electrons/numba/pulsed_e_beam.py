from numba import njit
from numpy.typing import NDArray

from electrons.common.pulsed_e_beam import get_pulsed_beam_pde_solver
from electrons.numba.generic_electron_solver import NumbaGenericElectronSolver

BasePulsedBeamPDEsolver = get_pulsed_beam_pde_solver(NumbaGenericElectronSolver)


class NumbaPulsedBeamPDEsolver(BasePulsedBeamPDEsolver, NumbaGenericElectronSolver):

    def get_beam_simulation_functions(self):

        # Make the variables local so numba.jit doesn't get mad
        electron_density_per_cm3 = self.electron_density_per_cm3
        delta_border = self.delta_border
        no_xy = self.no_xy
        no_z_electrode = self.no_z_electrode
        no_z = self.no_z

        @njit
        def simulate_beam(positive_array: NDArray, negative_array: NDArray):
            positive_array[
                delta_border : (no_xy - delta_border),
                delta_border : (no_xy - delta_border),
                no_z_electrode : (no_z + no_z_electrode),
            ] += electron_density_per_cm3

            negative_array[
                delta_border : (no_xy - delta_border),
                delta_border : (no_xy - delta_border),
                no_z_electrode : (no_z + no_z_electrode),
            ] += electron_density_per_cm3

            initialised_charge_carriers = (
                electron_density_per_cm3 * (no_xy - 2 * delta_border) ** 2 * no_z
            )

            return initialised_charge_carriers

        @njit
        def should_simulate_beam_for_time_step(time_step: int):
            return time_step == 0

        return (
            should_simulate_beam_for_time_step,
            simulate_beam,
        )
