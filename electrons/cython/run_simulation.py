from electrons.cython.pulsed_e_beam import pulsed_beam_PDEsolver
from electrons.cython.continuous_e_beam import continuous_beam_PDEsolver

# from .continuous_e_beam import continuous_beam_PDEsolver
import sys

SOLVER_MAP = {"continous": continuous_beam_PDEsolver, "pulsed": pulsed_beam_PDEsolver}


def run_simulation(
    voltage_V=300,
    electrode_gap=0.1,
    electron_density_per_cm3=1e9,
):
    parameters = {
        "voltage_V": voltage_V,
        "d_cm": electrode_gap,
        "elec_per_cm3": electron_density_per_cm3,
        "show_plot": False,
        "print_parameters": False,
    }

    solver_arg = sys.argv[1] if len(sys.argv) > 1 else ""

    solver_type = solver_arg if solver_arg in SOLVER_MAP.keys() else "continous"

    solver = SOLVER_MAP[solver_type]

    if solver_type not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_type}", defaulting to Continous solver.')

    # return the collection efficiency
    result = solver(parameters)

    if solver_type == "continous":
        f = result[1]["f"]
    else:
        f = result

    return f


if __name__ == "__main__":

    result = run_simulation(
        electron_density_per_cm3=1e9,
        voltage_V=300,
        electrode_gap=0.1,
    )

    print("calculated f is", result)
