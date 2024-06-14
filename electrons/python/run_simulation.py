import argparse

from continuous_e_beam import ContinousBeamPDEsolver
from pulsed_e_beam import PulsedBeamPDEsolver

SOLVER_MAP = {"continous": ContinousBeamPDEsolver, "pulsed": PulsedBeamPDEsolver}


def run_simulation(
    solver_name="continous",
    voltage_V=300,
    electrode_gap=0.1,
    electron_density_per_cm3=1e9,
    verbose=True,
):
    Solver = (
        SOLVER_MAP[solver_name]
        if solver_name in SOLVER_MAP.keys()
        else ContinousBeamPDEsolver
    )

    if solver_name not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_name}", defaulting to Continous solver.')
        solver_name = "continous"

    if verbose:
        print(f"Running the simulation using the {solver_name} solver.")
        print(f"Voltage: {voltage_V} [V]")
        print(f"Electrode gap: {electrode_gap} [cm]")
        print(f"Electron density per cm3: {electron_density_per_cm3}")

    solver = Solver(
        electron_density_per_cm3=electron_density_per_cm3,
        voltage_V=voltage_V,
        electrode_gap=electrode_gap,
        grid_spacing_cm=5e-4,
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
