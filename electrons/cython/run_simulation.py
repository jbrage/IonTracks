import argparse

from electrons.cython.continuous_e_beam import continuous_beam_PDEsolver
from electrons.cython.pulsed_e_beam import pulsed_beam_PDEsolver

SOLVER_MAP = {"continous": continuous_beam_PDEsolver, "pulsed": pulsed_beam_PDEsolver}


def run_simulation(
    solver_name="continous",
    voltage_V=300,
    electrode_gap=0.1,
    electron_density_per_cm3=1e9,
    verbose=True,
):
    parameters = {
        "voltage_V": voltage_V,
        "d_cm": electrode_gap,
        "elec_per_cm3": electron_density_per_cm3,
        "show_plot": False,
        "print_parameters": False,
    }

    if solver_name not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_name}", defaulting to Continous solver.')
        solver_name = "continous"

    if verbose:
        print(f"Running the simulation using the {solver_name} solver.")
        print(f"Voltage: {voltage_V} [V]")
        print(f"Electrode gap: {electrode_gap} [cm]")
        print(f"Electron density per cm3: {electron_density_per_cm3}")

    solver = SOLVER_MAP[solver_name]

    # return the collection efficiency
    result = solver(parameters)

    if solver_name == "continous":
        f = result[1]["f"]
    else:
        f = result

    return f


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

    print("calculated f is", result)
