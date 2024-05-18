from enum import Enum, auto
from hadrons.cython_files.initial_recombination import (
    single_track_PDEsolver as cython_single_track_PDEsolver,
)
from hadrons.python.initial_recombination import (
    single_track_PDEsolver as python_single_track_PDEsolver,
)
from hadrons.numba.initial_recombination_numba import (
    single_track_PDEsolver as numba_single_track_PDEsolver,
)
from hadrons.parallel.initial_recombination_numba_parallel import (
    single_track_PDEsolver as numba_parallel_single_track_PDEsolver,
)


class SolverType(Enum):
    CYTHON = auto()
    PYTHON = auto()
    NUMBA = auto()
    NUMBA_PARALLEL = auto()


def solvePDE(input, type: SolverType):
    if type == SolverType.CYTHON:
        return cython_single_track_PDEsolver(**input)
    elif type == SolverType.PYTHON:
        PDE_solver = python_single_track_PDEsolver(**input)
        return PDE_solver.solve()
    elif type == SolverType.NUMBA:
        PDE_solver = numba_single_track_PDEsolver(**input)
        return PDE_solver.solve()
    elif type == SolverType.NUMBA_PARALLEL:
        PDE_solver = numba_parallel_single_track_PDEsolver(**input)
        return PDE_solver.solve()
    else:
        raise ValueError(f"Unsupported solver type: {type}")
