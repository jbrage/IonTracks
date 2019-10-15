import sys
import numpy as np
from recombination_cythonized import pulsed_beam_PDEsolver
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from Boag_theory import Boag_pulsed

sys.path.append('./cython')


def IonTracks_pulsed(voltage_V, d_cm, elec_per_cm3):

    '''
    Solve the partial differential equation:
        - SHOW_PLOT = True shows a few pictures of the charge carrier movements
        - PRINT_parameters = True prints the simulation parameters
            (grid size, time steps, useful for debugging)
    '''

    SHOW_PLOT = False
    PRINT_parameters = False
    simulation_parameters = [
                                elec_per_cm3,
                                voltage_V,
                                d_cm,
                                SHOW_PLOT,
                                PRINT_parameters
                            ]
    return pulsed_beam_PDEsolver(simulation_parameters)


if __name__ == "__main__":

    '''
    The Boag theory is validated against measurements, and IonTracks is here validated
    against the Boag theory.
    The Boag theory was validated in
        R E Ellis and L R Read 1969 Phys. Med. Biol. 14 293
    where the charges are given in esu units:
    '''

    d_cm = 0.1  # electrode gap
    voltages_V = np.asarray([1181, 901.3, 621.3, 434.8, 270, 186.1, 124.4, 75, 50, 25, 12.4])
    # preallocate arrays
    voltages_LS = np.linspace(min(voltages_V), max(voltages_V), 100)
    f_result_IonTracks = np.empty(len(voltages_V))
    f_result_Boag = np.empty(len(voltages_LS))

    # convert esu to coulomb
    e_charge = 1.60217662e-19  # [C]
    esu_list = np.asarray([2.98, 1.90, 1.31])
    esu_to_C = 3.33564e-10
    charge_density_C_cm3 = esu_list * esu_to_C

    # prepare plot
    clist = [i for i in colors.get_named_colors_mapping()]
    Boag_style = {'ls': '-'}
    IT_style = {'ls': '', 'marker': 'o', 'markerfacecolor': 'none'}
    plt.figure()

    for i, Q in enumerate(charge_density_C_cm3):

        # IonTracks uses the number of electrons per ccm rather than charge per ccm
        elec_per_cm3 = Q/e_charge

        for idx, V in enumerate(voltages_LS):
            f_result_Boag[idx] = Boag_pulsed(Q, d_cm, V)

        for idx, V in enumerate(voltages_V):
            f_result_IonTracks[idx] = IonTracks_pulsed(V, d_cm, elec_per_cm3)

        # plot results
        Boag_style['c'] = clist[i]
        Boag_style['label'] = r"$Q$={:.2e} C/cm3, Boag theory".format(Q)
        IT_style['c'] = clist[i]
        IT_style['label'] = r"$Q$={:.2e} C/cm3, IonTracks".format(Q)

        plt.plot(1./voltages_V, f_result_IonTracks, **IT_style)
        plt.plot(1./voltages_LS, f_result_Boag, **Boag_style)

    plt.title("Plane-parallel chamber with $d = {:0.2g}$ cm air gap".format(d_cm))
    plt.xlabel("Inverse voltage [1/V]")
    plt.ylabel("Collection efficiency")
    plt.legend(loc='best', frameon=False)
    plt.savefig("collection_efficiencies.pdf")
