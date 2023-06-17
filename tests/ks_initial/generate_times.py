import time
from hadrons.solver import solvePDE
from tests.ks_initial.testing_parameters import TEST_DATA_FRAMES
from tests.utils import get_PDEsolver_input
from hadrons.solver import SolverType
from pathlib import Path

DEBUG = True

def time_solver(row, solver_type, verbose=DEBUG):
    input_dict = get_PDEsolver_input(
        row['E_MeV_u'],
        row['voltage_V'],
        row['electrode_gap_cm'],
        row['particle'],
        row['grid_size_um'],
    )

    start_time = time.time()
    
    solvePDE(input_dict, type=solver_type)

    run_time = time.time() - start_time
    
    if verbose:
        print(f'{solver_type} ran for {run_time} seconds.')

    return run_time

def main():
    # compile numba so the compilation time is not included in timing
    for solver_type in [SolverType.NUMBA_PARALLEL, SolverType.NUMBA]:
        time_solver(TEST_DATA_FRAMES[0].iloc[0], solver_type, verbose=False)

    for idx, MATRIX_DF in enumerate(TEST_DATA_FRAMES):
        result_df = MATRIX_DF.copy()

        print(f"CALCULATING DF {idx}")

        for solver_type in [SolverType.NUMBA_PARALLEL, SolverType.CYTHON]:
            result_df[f'{solver_type.name.lower()}_time'] = MATRIX_DF.apply(lambda row: time_solver(row, solver_type), axis=1)

        filepath = Path(f'times_result_{idx}.csv')

        print(result_df)
        
        result_df.to_csv(filepath)

    return 0

if __name__ == '__main__':
    main()