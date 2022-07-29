import numpy as np
from electrons.cython.continuous_e_beam import continuous_beam_PDEsolver
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from Boag_theory import Boag_Continuous


def IonTracks_continuous(voltage_V, d_cm, elec_per_cm3):

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
    f, f_steps, dt, collection_time_steps = continuous_beam_PDEsolver(simulation_parameters)

    time_s = np.arange(0, len(f_steps)*dt, dt)
    sep_time_s = collection_time_steps*dt*2

    f = f_steps[-1]
    # plt.figure()
    # plt.plot(time_s, f_steps)
    # plt.plot([sep_time_s, sep_time_s], [min(f_steps), max(f_steps)])
    #
    # plt.xlabel("Time [s]")
    # plt.ylabel("Collection efficiency $f$")
    # plt.savefig("recomb_cont.pdf")

    return f


if __name__ == "__main__":


    d_cm = 0.2 # electrode gap
    voltages_V = np.asarray([60, 120,  200])

    # preallocate arrays
    voltages_LS = np.linspace(min(voltages_V), max(voltages_V), 100)
    f_result_IonTracks = np.empty(len(voltages_V))
    f_result_Boag = np.empty((3, len(voltages_LS)))

    # charge densities (convert from dose-rate in air?)
    e_charge = 1.60217662e-19 # [C]
    charge_density_C_cm3 = [0.02*1e-6, 0.08*1e-6]

    # prepare plot
    clist = [i for i in colors.get_named_colors_mapping()]
    Boag_style = {'alpha' : 0.25 }
    IT_style = {'ls' : '', 'marker' : 'o', 'markerfacecolor' : 'none'}
    plt.figure()

    for i, Q in enumerate(charge_density_C_cm3):

        # IonTracks uses the number of electrons per ccm rather than charge per ccm
        elec_per_cm3 = Q/e_charge

        for idx, V in enumerate(voltages_LS):
            f_result_Boag[:, idx] = Boag_Continuous(Q, d_cm, V)

        for idx, V in enumerate(voltages_V):
            f_result_IonTracks[idx] = IonTracks_continuous(V, d_cm, elec_per_cm3)

        # plot results
        Boag_style['color'] = clist[i]
        Boag_style['label'] = r"$Q$={:.2e} C/cm3/s, Boag theory".format(Q)
        IT_style['color'] = clist[i]
        IT_style['label'] = r"$Q$={:.2e} C/cm3/s, IonTracks".format(Q)

        plt.plot(1./voltages_V, f_result_IonTracks, **IT_style)

        # plot Boag with 1 std uncertainties; Boag should be avoided below f = 0.7
        Boag_low = f_result_Boag[1,:]
        Boag_high = f_result_Boag[2,:]
        Boag_mean = f_result_Boag[0,:]

        plt.fill_between(1./voltages_LS, Boag_high, Boag_low, **Boag_style)
        plt.plot(1./voltages_LS, Boag_mean, c=clist[i], ls = '--')

    plt.title("Plane-parallel chamber with $d = {:0.2g}$ cm air gap".format(d_cm))
    plt.xlabel("Inverse voltage [1/V]")
    plt.ylabel("Collection efficiency")
    plt.legend(loc = 'best', frameon=False)
    plt.savefig("collection_efficiencies.pdf", bbox_inches='tight')
