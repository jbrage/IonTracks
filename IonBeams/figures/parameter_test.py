import sys
sys.path.append("./cython")
import pyximport
pyximport.install()
import numpy as np
from general_and_initial_recombination import total_recombination_PDEsolver
from functions import E_MeV_to_LET_keV_um, calc_b_cm, doserate_to_fluence
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from scipy.interpolate import interp1d


def IonTracks_total_recombination(parameters, PRINT_parameters=False):
    """
    Calculate the stopping power, fluence-rate, and track radius as a
    function of proton energy and dose-rate.

    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    """
    
    # unpack parameters (TO DO: use a dictionary)
    (
        voltage_V,
        energy_MeV,
        doserate_Gy_min,
        electrode_gap_cm,
        unit_length_cm,
        r_cm,
        buffer_radius,
    ) = parameters

    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    track_radius_cm = calc_b_cm(LET_keV_um)
    fluence_cm2_s = doserate_to_fluence(doserate_Gy_min, energy_MeV)

    SHOW_PLOT = False
    myseed = int(np.random.randint(1, 1e7))
    simulation_parameters = [
        fluence_cm2_s,
        LET_keV_um,
        track_radius_cm,
        voltage_V,
        electrode_gap_cm,
        SHOW_PLOT,
        myseed,
        PRINT_parameters,
        unit_length_cm,
        r_cm,
        buffer_radius,
    ]

    f = total_recombination_PDEsolver(simulation_parameters)
    return 1.0 / f


if __name__ == "__main__":

    # old settings
    voltage_V = 200
    energy_MeV = 226
    buffer_radius = 10 # sufficient

    df = pd.DataFrame()

    for doserate_Gy_min in [10000, 1000]:
        for electrode_gap_cm in [0.15, 0.1]:
            
            previous_parameters = [voltage_V, energy_MeV, doserate_Gy_min, electrode_gap_cm]

            for unit_length_um in [10]: # smaller grid length => more voxels => longer computation time
                for r_um in np.arange(20, 220, 20): # larger radius => more voxels
                    
                    # convert from um to cm
                    r_cm = r_um * 1e-4
                    unit_length_cm = unit_length_um * 1e-4
    
                    # these are varied
                    variables = [unit_length_cm, r_cm, buffer_radius]
    
                    # combine with the "normal"
                    input_parameters = np.concatenate((previous_parameters, variables))
    
                    # solve
                    ks_IonTracks = IonTracks_total_recombination(input_parameters)
    
                    # save results to a data frame
                    result = {
                        "doserate_Gy_min": doserate_Gy_min,
                        "electrode_gap_cm": electrode_gap_cm,
                        "buffer_radius": buffer_radius,
                        "unit_length_um": unit_length_um,
                        "r_um": r_um,
                        "k_s": ks_IonTracks,
                    }
                    df = df.append(result, ignore_index=True)
    
                    # print some progress ...
                    print(result)
                
    df.to_csv("convergence_data.csv", index=False)

    fig_diff, ax_diff = plt.subplots()
    
    nrows = df.doserate_Gy_min.unique().size
    ncols = df.electrode_gap_cm.unique().size
    
    fig, ax_grid = plt.subplots(nrows=nrows, 
                                ncols=ncols, 
                                sharex=True, 
                                figsize=(3*nrows, 2*ncols)
                                )
    axes = ax_grid.ravel()
    
    # define the old and the new simulation cylinder radius (in microns)
    radii_um_dic = {"Old radius": 60, "New radius": 120}
    
    # group the data by dose rate and electrode gap height
    for ax_no, (pars, group) in enumerate(df.groupby(["doserate_Gy_min", "electrode_gap_cm"])):
        
        ID = ax_no + 1
        ax = axes[ax_no]
        ax.set_title("ID {}: $\dot{{D}}$ = {:.0f} kGy/min".format(ID, pars[0]/1000))
        
        
        # interpolate to estimate k_s (group multiple unit_length_um values)
        g = group.groupby("r_um").mean().reset_index()
        interpol_obj = interp1d(x=g.r_um, y=g.k_s)
        xls = np.linspace(group.r_um.min(), group.r_um.max(), 100)
        
        # the highest calculated
        final_k_s = interpol_obj(group.r_um.max())
        
        # plot the underestimation
        underestimation_per = 100*(interpol_obj(xls) / final_k_s - 1)
        label = "ID {}".format(ID)
        ax_diff.plot(xls, underestimation_per, label=label)
        
        sns.lineplot(
            data=group,
            ax=ax,
            x="r_um",
            y="k_s",
            hue="unit_length_um",
            style="unit_length_um",
            markers=True,
            legend=False,
        )

        # add the new and old default radius to the fig
        for name in radii_um_dic.keys():
            r_um = radii_um_dic[name]
            ax.text(radii_um_dic[name], 1, name, rotation=90, va="bottom", ha="right")
    
            k_s_radius = interpol_obj(radii_um_dic[name])
            ax.plot([0, r_um, r_um], [k_s_radius, k_s_radius, 1])        
            
            
            ax_diff.axvline(x=r_um, c="k", lw=1) 

    
        ax.set_xlabel("Simulation cylinder radius (um)")
        ax.set_ylabel("$k_s$   ($d$ = {:0.2} cm)".format(pars[1]))
        # ax.legend(title="Unit length (um)", loc="lower right")
        ax.set_xlim(xmin=0)
        
        # python does the silly +1 addition on the y ticks; avoid this
        if group["k_s"].max() < 1.002:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        
    fig.tight_layout()
    fig.savefig("convergence_plot.pdf", bbox_inches="tight") 
    fig.savefig("convergence_plot.png", dpi=250, bbox_inches="tight")
    
    
    for name in radii_um_dic.keys():
        r_um = radii_um_dic[name]
        ax_diff.text(radii_um_dic[name], min(ax_diff.get_ylim()), name, rotation=90, va="bottom", ha="right")
        
    ax_diff.axhline(y=0, ls=":", c="k", zorder=-1)
    ax_diff.legend()
    ax_diff.set_xlabel("Simulation cylinder radius (um)")
    ax_diff.set_ylabel("Underestimation of $k_s$ (%)")
    fig_diff.savefig("Underestimation_plot.pdf", bbox_inches="tight")
    fig_diff.savefig("Underestimation_plot.png", dpi=250, bbox_inches="tight")
