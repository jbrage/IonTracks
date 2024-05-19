from continuous_e_beam import ContinousBeamPDEsolver
from pulsed_e_beam import PulsedBeamPDEsolver


def get_solver(solver_name: str):
    if solver_name == "continous":
        return ContinousBeamPDEsolver

    if solver_name == "pulsed":
        return PulsedBeamPDEsolver

    raise TypeError(f"Invalid electron solver name {solver_name}")


def run_simulation(
    voltage_V=300,
    electrode_gap=0.1,
    electron_density_per_cm3=1e9,
):
    solver_name = "pulsed"

    Solver = get_solver(solver_name)

    solver = Solver(
        electron_density_per_cm3=electron_density_per_cm3,
        voltage_V=voltage_V,
        electrode_gap=electrode_gap,
        unit_length_cm=4e-4,
    )

    return solver.calculate()


if __name__ == "__main__":
    result = run_simulation()

    print(f"calculated f is {result}")
