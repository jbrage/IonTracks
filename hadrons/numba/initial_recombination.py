import numba as np

from hadrons.common.initial_recombination import get_initial_recombination_pde_solver
from hadrons.numba.generic_hadron_solver import NumbaHadronSolver

BaseInitialRecombinationPDESolver = get_initial_recombination_pde_solver(
    NumbaHadronSolver
)


class NumbaInitialRecombinationPDESolver(BaseInitialRecombinationPDESolver):
    def get_track_inserting_functions(self):
        def should_insert_track(time_step: int) -> bool:
            return time_step % 100 == 0

        def get_track_for_time_step(
            positive_array: np.ndarray, negative_array: np.ndarray
        ) -> float:
            return np.sum(positive_array + negative_array)

        return should_insert_track, get_track_for_time_step
