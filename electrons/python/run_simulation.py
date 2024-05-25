from continuous_e_beam import ContinousBeamPDEsolver
from pulsed_e_beam import PulsedBeamPDEsolver
import sys

SOLVER_MAP = {"continous": ContinousBeamPDEsolver, "pulsed": PulsedBeamPDEsolver}


def run_simulation(
    voltage_V=300,
    electrode_gap=0.1,
    electron_density_per_cm3=1e9,
):
    solver_name = sys.argv[1] if len(sys.argv) > 0 else ""

    Solver = (
        SOLVER_MAP[solver_name]
        if solver_name in SOLVER_MAP.keys()
        else ContinousBeamPDEsolver
    )

    if solver_name not in SOLVER_MAP.keys():
        print(f'Invalid solver type "{solver_name}", defaulting to Continous solver.')

    solver = Solver(
        electron_density_per_cm3=electron_density_per_cm3,
        voltage_V=voltage_V,
        electrode_gap=electrode_gap,
        unit_length_cm=5e-4,
    )

    return solver.calculate()


if __name__ == "__main__":
    result = run_simulation()

    print(f"calculated f is {result}")
