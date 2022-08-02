import numpy as np
from electrons.cython.pulsed_e_beam import pulsed_beam_PDEsolver
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from Boag_theory import Boag_pulsed, e_charge
import pandas as pd
import seaborn as sns
import itertools


def IonTracks_pulsed(voltage_V=300, electrode_gap_cm=0.1, elec_per_cm3=1e9, show_plot=False, print_parameters=False):

    parameters = {"voltage_V": voltage_V,
    	          "d_cm": electrode_gap_cm, 
    	          "elec_per_cm3": elec_per_cm3,
    	          "show_plot": show_plot,  
                  "print_parameters" : print_parameters,
                  }                                      
    
    # return the collection efficiency                        
    f = pulsed_beam_PDEsolver(parameters)
    
    parameters["ks"] = 1/f
    return pd.DataFrame([parameters])


if __name__ == "__main__":

    '''
    The Boag theory is validated against measurements, and IonTracks is here validated
    against the Boag theory.
    
    The Boag theory was validated in
        R E Ellis and L R Read 1969 Phys. Med. Biol. 14 293
    where the charges are given in esu units:
    '''

    # convert esu to coulomb, data from the validation paper
    data_esu_cm3 = np.asarray([2.98, 1.90, 1.31])
    esu_to_C = 3.33564e-10
    charge_density_C_cm3 = data_esu_cm3 * esu_to_C 

    d_cm = 0.1  # electrode gap

    # set parameters for the IonTracks calculation
    data_dict = dict(
        electrode_gap_cm=[d_cm],
        elec_per_cm3 = charge_density_C_cm3/ e_charge,
        voltage_V = [1181, 901.3, 621.3, 434.8, 270, 186.1, 124.4, 75, 50, 25, 12.4],
    )

    # create a data frame with all the variables
    data_df = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), columns=data_dict.keys())

    # Calculate the recombination using IonTracks
    IonTracks_df = pd.DataFrame()
    for idx, data in data_df.iterrows():
        result_df = IonTracks_pulsed(voltage_V=data.voltage_V, electrode_gap_cm=data.electrode_gap_cm, elec_per_cm3=data.elec_per_cm3)
        IonTracks_df = pd.concat([IonTracks_df, result_df], ignore_index=True)    
    print(IonTracks_df)
    
    # rename for plot
    IonTracks_df["electron_density"] = IonTracks_df["elec_per_cm3"].map("IonTracks: {:0.2E} e$^{{-}}$/cm$^3$".format)
    
    # set parameters for the Boag theorys
    data_dict = dict(
        electrode_gap_cm=[d_cm],
        charge_density_C_cm3 = charge_density_C_cm3,
        # logscaled points
        voltage_V = 10 **np.linspace(np.log10(IonTracks_df.voltage_V.min()), np.log10(IonTracks_df.voltage_V.max()), 100),
    )

    # create a data frame with all the variables
    Boag_df = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), columns=data_dict.keys())
    # calculate the recombination using the Boag model
    for idx, data in Boag_df.iterrows():
        Boag_df.loc[idx, "f"] = Boag_pulsed(data.charge_density_C_cm3, data.electrode_gap_cm, data.voltage_V)
        
    Boag_df["ks"] = 1/Boag_df["f"]
    print(Boag_df)
    # rename for plot
    Boag_df["charge_density_C_cm3"] = Boag_df["charge_density_C_cm3"].map("Boag: {:0.2E} C/cm$^3$".format)
    
    # plot the results
    fig, ax = plt.subplots()
    
    sns.lineplot(ax=ax, data=Boag_df, x="voltage_V", y="ks", hue="charge_density_C_cm3")
    sns.scatterplot(data=IonTracks_df, ax=ax, x="voltage_V", y="ks", hue="electron_density")    

    ax.set_title("Plane-parallel chamber with $d = {:0.2g}$ cm air gap".format(d_cm))
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Recombination correction factor ($k_s$)")
    ax.set_xscale("log")
    ax.legend(loc='best', frameon=False)
    fig.savefig("fig_example_pulsed.pdf")
