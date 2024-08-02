import argparse

from electrons.numba.continous_e_beam import (
    NumbaContinousBeamPDEsolver as ContinousBeamPDEsolver,
)
from electrons.numba.pulsed_e_beam import (
    NumbaPulsedBeamPDEsolver as PulsedBeamPDEsolver,
)

SOLVER_MAP = {"continous": ContinousBeamPDEsolver, "pulsed": PulsedBeamPDEsolver}


def run_simulation(
    solver_name="continous",
    voltage=300,
    electrode_gap=0.1,
    electron_density_per_cm3=1e9,
    verbose=True,
):
    # select a solver based on string name, defaukt to ContinousBeamPDEsolver if the name is invalid
    Solver = SOLVER_MAP.get(solver_name, ContinousBeamPDEsolver)

    if solver_name not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_name}", defaulting to Continous solver.')
        solver_name = "continous"

    if verbose:
        print(f"Running the simulation using the {solver_name} solver.")
        print(f"Voltage: {voltage} [V]")
        print(f"Electrode gap: {electrode_gap} [cm]")
        print(f"Electron density per cm3: {electron_density_per_cm3}")

    solver = Solver(
        electron_density_per_cm3=electron_density_per_cm3,
        voltage=voltage,
        electrode_gap=electrode_gap,
        grid_spacing=5e-4,
    )

    return solver.calculate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "solver_name",
        type=str,
        default="continous",
        help="The type of the solver to use",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    result = run_simulation(**vars(args))

    print(f"calculated f is {result}")
