import argparse

from hadrons.python.continous_beam import ContinousHadronSolver
from hadrons.python.initial_recombination import InitialHadronSolver

SOLVER_MAP = {"continous": ContinousHadronSolver, "initial": InitialHadronSolver}


def run_simulation(
    solver_name="continous",
    LET=1,
    voltage=300,
    IC_angle=1,
    electrode_gap=0.1,
    energy=200,
    electron_density_per_cm3=1e9,
    verbose=False,
):
    if solver_name not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_name}", defaulting to Continous solver.')
        solver_name = "continous"

    # select a solver based on string name, defaukt to ContinousBeamPDEsolver if the name is invalid
    Solver = SOLVER_MAP.get(solver_name, InitialHadronSolver)

    if verbose:
        print(f"Running the simulation using the {solver_name} solver.")
        print(f"Voltage: {voltage} [V]")
        print(f"Electrode gap: {electrode_gap} [cm]")
        print(f"Electron density per cm3: {electron_density_per_cm3}")

    solver = Solver(
        # LET=LET,
        voltage=voltage,  # [V/cm] magnitude of the electric field
        IC_angle=IC_angle,
        electrode_gap=electrode_gap,
        energy=energy,
    )

    return solver.calculate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "solver_name",
        default="continous",
        help="The type of the solver to use",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )

    args = parser.parse_args()

    result = run_simulation(**vars(args))

    print(f"calculated f is {result}")
