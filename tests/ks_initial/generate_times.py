import time
from hadrons.solver import solvePDE
from tests.ks_initial.testing_parameters import SLOW_MATRIX_DF
from tests.utils import get_PDEsolver_input
from hadrons.solver import SolverType
from pathlib import Path

def time_solver(row, solver_type):
    input_dict = get_PDEsolver_input(
        row['E_MeV_u'],
        row['voltage_V'],
        row['electrode_gap_cm'],
        row['particle'],
        row['grid_size_um'],
    )

    start_time = time.time()
    
    solvePDE(input_dict, type=solver_type)

    return time.time() - start_time

def main():
    result_df = SLOW_MATRIX_DF.copy()

    # compile numba so the compilation time is not included in timing
    time_solver(SLOW_MATRIX_DF.iloc[0], SolverType.NUMBA)

    # for solver_type in SolverType:
    for solver_type in [SolverType.NUMBA, SolverType.CYTHON]:
        result_df[f'{solver_type.name.lower()}_time'] = SLOW_MATRIX_DF.apply(lambda row: time_solver(row, solver_type), axis=1)

    filepath = Path('times_result.csv')

    print(result_df)
    
    result_df.to_csv(filepath, )

    return 0

if __name__ == '__main__':
    main()