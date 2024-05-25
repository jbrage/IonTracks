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

    solver_name = sys.argv[1] if len(sys.argv) > 1 else ""

    solver = (
        SOLVER_MAP[solver_name]
        if solver_name in SOLVER_MAP.keys()
        else continuous_beam_PDEsolver
    )

    if solver_name not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_name}", defaulting to Continous solver.')

    # return the collection efficiency
    f = solver(parameters)

    return f


if __name__ == "__main__":

    result = run_simulation(
        electron_density_per_cm3=1e9,
        voltage_V=300,
        electrode_gap=0.1,
    )

    print("calculated f is", result)
