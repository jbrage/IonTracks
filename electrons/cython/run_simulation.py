from electrons.cython.pulsed_e_beam import pulsed_beam_PDEsolver


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

    # return the collection efficiency
    f = pulsed_beam_PDEsolver(parameters)

    return 1 / f


if __name__ == "__main__":

    result = run_simulation(
        electron_density_per_cm3=1e9,
        voltage_V=300,
        electrode_gap=0.1,
    )

    print(f"calculated f is {result}")
