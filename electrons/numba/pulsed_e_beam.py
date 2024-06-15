from electrons.common.pulsed_e_beam import get_pulsed_beam_pde_solver
from electrons.numba.generic_electron_solver import NumbaGenericElectronSolver

NumbaPulsedBeamPDEsolver = get_pulsed_beam_pde_solver(NumbaGenericElectronSolver)
