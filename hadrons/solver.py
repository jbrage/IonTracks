from enum import Enum, auto
from hadrons.cython_files.initial_recombination import single_track_PDEsolver as cython_single_track_PDEsolver
from hadrons.initial_recombination import single_track_PDEsolver as python_single_track_PDEsolver

class SolverType(Enum):
    CYTHON = auto()
    PYTHON = auto()

def solvePDE(input, type: SolverType):
    if type == SolverType.CYTHON:
        return cython_single_track_PDEsolver(*input)
    elif type == SolverType.PYTHON:
        return python_single_track_PDEsolver(**input)
    else:
        raise ValueError(f"Unsupported solver type: {type}")