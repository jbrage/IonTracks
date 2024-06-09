import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Boag_theory import Boag_continuous, e_charge

from electrons.cython.continuous_e_beam import continuous_beam_PDEsolver


def IonTracks_continuous(
    voltage_V=300,
    electrode_gap_cm=0.1,
    elec_per_cm3=1e9,
    show_plot=False,
    print_parameters=False,
):
    # convert to a dict
    parameters = {
        "voltage_V": voltage_V,
        "d_cm": electrode_gap_cm,
        "elec_per_cm3": elec_per_cm3,
        "show_plot": show_plot,
        "print_parameters": print_parameters,
    }

    result_df, time_variation_df = continuous_beam_PDEsolver(parameters)

    return result_df, time_variation_df


if __name__ == "__main__":
    """
    # calculate an example for default parameters
    result_df, time_variation_df = IonTracks_continuous()
    print(result_df)

    # plot the result to show how the charge collection converges after the time it takes for
    # a partcile to move between the two electrodes. This is the recombination in the continuous situaiton
    fig, ax = plt.subplots()
    ax.set_title("Charge collection versus time")
    sns.lineplot(ax=ax, data=time_variation_df, x="time_us", y="f", label="Charge collection")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("Collection efficiency")
    ax.axvline(x=result_df["convergence_time_s"].values * 1e6, label="Drift time b/w gap", c="r", ls=":")
    ax.axhline(y=result_df["f"].values, label="Converged efficiency", c="r", ls="--")
    ax.ticklabel_format(style="plain")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig_charge_collection_versus_time.pdf", bbox_inches="tight")"""

    # compare the IonTracks results to the Boag theory for a continuous beam

    # set parameters for the IonTracks calculation
    d_cm = 0.1
    charge_density_C_cm3_s = np.asarray([0.02, 0.08]) * 1e-6
    data_dict = dict(
        electrode_gap_cm=[d_cm],
        elec_per_cm3_s=charge_density_C_cm3_s / e_charge,
        voltage_V=np.asarray([60, 120, 200]),
    )

    # create a data frame with all the variables
    data_df = pd.DataFrame.from_records(
        data=itertools.product(*data_dict.values()), columns=data_dict.keys()
    )

    # Calculate the recombination using IonTracks
    IonTracks_df = pd.DataFrame()
    for idx, data in data_df.iterrows():
        result_df, _ = IonTracks_continuous(
            voltage_V=data.voltage_V,
            electrode_gap_cm=data.electrode_gap_cm,
            elec_per_cm3=data.elec_per_cm3_s,
        )
        IonTracks_df = pd.concat([IonTracks_df, result_df], ignore_index=True)
    print(IonTracks_df)

    # rename for plot
    IonTracks_df["electron_density"] = IonTracks_df["elec_per_cm3"].map(
        "IonTracks: {:0.2E} e$^{{-}}$/cm$^3$".format
    )

    # generate the data for the Boag theory
    data_dict[
        "charge_density_C_cm3_s"
    ] = charge_density_C_cm3_s  # replace by the charge density, not electron
    data_dict["voltage_V"] = 10 ** np.linspace(
        np.log10(IonTracks_df.voltage_V.min()),
        np.log10(IonTracks_df.voltage_V.max()),
        100,
    )
    Boag_df = pd.DataFrame.from_records(
        data=itertools.product(*data_dict.values()), columns=data_dict.keys()
    )

    # calculate the Boag collection efficiency for each point
    for idx, data in Boag_df.iterrows():
        f, _, _ = Boag_continuous(
            Qdensity_C_cm3_s=data.charge_density_C_cm3_s,
            d_cm=data.electrode_gap_cm,
            V=data.voltage_V,
        )
        Boag_df.loc[idx, "f"] = f
    Boag_df["ks"] = 1 / Boag_df["f"]
    Boag_df["charge_density_C_cm3_s"] = Boag_df["charge_density_C_cm3_s"].map(
        "Boag: {:0.2E} C/cm$^3$".format
    )

    # plot the results
    fig, ax = plt.subplots()

    sns.lineplot(
        ax=ax, data=Boag_df, x="voltage_V", y="ks", hue="charge_density_C_cm3_s"
    )
    sns.scatterplot(
        ax=ax, data=IonTracks_df, x="voltage_V", y="ks", hue="electron_density"
    )

    ax.set_title("Continuous beam: $d = {:0.2g}$ cm air gap".format(d_cm))
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Recombination correction factor ($k_s$)")
    ax.set_xscale("log")
    ax.legend(loc="best", frameon=False)
    fig.savefig("fig_example_continuous.pdf")
