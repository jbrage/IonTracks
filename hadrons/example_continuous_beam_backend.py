import itertools
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functions import IonTracks_continuous_beam

def main():
    # Parse backend argument
    if len(sys.argv) > 1:
        backend = sys.argv[1].lower()
    else:
        backend = "cython"

    # set parameters
    data_dict = dict(
        electrode_gap_cm=[0.1],
        particle=["proton", "carbon"],
        voltage_V=[300],
        E_MeV_u=[250],
        doserate_Gy_min=10 ** np.arange(3),
    )

    # create a data frame with all the variables
    data_df = pd.DataFrame.from_records(
        data=itertools.product(*data_dict.values()), columns=data_dict.keys()
    )

    # Prepare output file for stepwise results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_dir = "../artifacts"
    import os
    os.makedirs(artifacts_dir, exist_ok=True)
    filename = os.path.join(artifacts_dir, f"IonTracks_results_{backend}_{timestamp}.txt")
    IonTracks_df = pd.DataFrame()
    start_time = time.time()
    results_str = ""
    for idx, data in data_df.iterrows():
        result_df = IonTracks_continuous_beam(
            E_MeV_u=data.E_MeV_u,
            voltage_V=data.voltage_V,
            particle=data.particle,
            doserate_Gy_min=data.doserate_Gy_min,
            backend=backend,
        )
        IonTracks_df = pd.concat([IonTracks_df, result_df], ignore_index=True)
        results_str += f"Step {idx}\n"
        results_str += IonTracks_df.to_csv(sep="\t", index=False)
        results_str += "\n"
    elapsed = time.time() - start_time
    with open(filename, "w") as f:
        f.write(results_str)
        f.write(f"Elapsed time: {elapsed:.3f} s\n")
    print(f"Results saved to {filename}")

    # plot
    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        data=IonTracks_df,
        x="doserate_Gy_min",
        y="ks_IonTracks",
        hue="voltage_V",
        style="particle",
        markers=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Dose rate (Gy/s)")
    ax.set_ylabel("$k_s$ (IonTracks)")
    fig.savefig(os.path.join(artifacts_dir, f"IonTracks_continuous_beam_{backend}.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    main()
