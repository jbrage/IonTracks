from numba import njit
from numpy.typing import NDArray

from electrons.common.continous_e_beam import get_continous_beam_pde_solver
from electrons.numba.generic_electron_solver import NumbaGenericElectronSolver

BaseContinousBeamPDEsolver = get_continous_beam_pde_solver(NumbaGenericElectronSolver)


class NumbaContinousBeamPDEsolver(
    BaseContinousBeamPDEsolver, NumbaGenericElectronSolver
):

    def get_beam_simulation_functions(self):

        # Make the variables local so numba.jit doesn't get mad
        electron_density_per_cm3 = self.electron_density_per_cm3
        dt = self.dt
        delta_border = self.delta_border
        no_xy = self.no_xy
        no_z_electrode = self.no_z_electrode
        no_z = self.no_z

        @njit
        def simulate_beam(positive_array: NDArray, negative_array: NDArray):
            electron_density_per_cm3_s = electron_density_per_cm3 * dt

            positive_array[
                delta_border : (no_xy - delta_border),
                delta_border : (no_xy - delta_border),
                no_z_electrode : (no_z + no_z_electrode),
            ] += electron_density_per_cm3_s

            negative_array[
                delta_border : (no_xy - delta_border),
                delta_border : (no_xy - delta_border),
                no_z_electrode : (no_z + no_z_electrode),
            ] += electron_density_per_cm3_s

            electron_density_per_cm3_s = electron_density_per_cm3 * dt

            initialised_charge_carriers = (
                electron_density_per_cm3_s * (no_xy - 2 * delta_border) ** 2 * no_z
            )

            return initialised_charge_carriers

        @njit
        def should_simulate_beam_for_time_step(time_step: int):
            return True

        return (
            should_simulate_beam_for_time_step,
            simulate_beam,
        )
