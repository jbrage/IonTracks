from functions import IonTracks_continuous_beam
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import seaborn as sns


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

# calcualte the ion recombination correction factor using IonTracks
IonTracks_df = pd.DataFrame()
for idx, data in data_df.iterrows():
    result_df = IonTracks_continuous_beam(
        E_MeV_u=data.E_MeV_u,
        voltage_V=data.voltage_V,
        particle=data.particle,
        doserate_Gy_min=data.doserate_Gy_min,
    )

    IonTracks_df = pd.concat([IonTracks_df, result_df], ignore_index=True)
    print(idx, IonTracks_df)

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
fig.savefig("IonTracks_continuous_beam.pdf", bbox_inches="tight")
