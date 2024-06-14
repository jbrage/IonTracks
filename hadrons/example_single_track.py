import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functions import Jaffe_theory, ks_initial_IonTracks

# set parameters
data_dict = dict(
    electrode_gap_cm=[0.1],
    particle=["proton", "carbon"],  # proton, helium, carbon, argon, iron
    voltage_V=[200, 300],
    E_MeV_u=np.linspace(1, 250, 100),
)

# create a data frame with all the variables
data_df = pd.DataFrame.from_records(
    data=itertools.product(*data_dict.values()), columns=data_dict.keys()
)

# use the Jaffe theory for initial recombination for these parameters
result_df = pd.DataFrame()
for idx, data in data_df.iterrows():
    Jaffe_df = Jaffe_theory(
        data.E_MeV_u,
        data.voltage_V,
        data.electrode_gap_cm,
        particle=data.particle,
        input_is_LET=False,
    )
    result_df = pd.concat([result_df, Jaffe_df], ignore_index=True)

# plot the results
fig, ax = plt.subplots()
sns.lineplot(
    ax=ax, data=result_df, x="E_MeV_u", y="ks_Jaffe", hue="voltage_V", style="particle"
)
ax.set_xlabel("Energy (MeV/u)")
ax.set_ylabel("$k_s$ Jaffe")
fig.savefig("Jaffe_example.pdf", bbox_inches="tight")

print("... Jaffe theory finished")

# APPLY IONTRACKS:
# reduce the number of parameters
data_dict["E_MeV_u"] = [60, 250]
data_df_shorter = pd.DataFrame.from_records(
    data=itertools.product(*data_dict.values()), columns=data_dict.keys()
)

# calculate the recombination with the IonTracks code
IonTracks_df = pd.DataFrame()
for idx, data in data_df_shorter.iterrows():
    temp_df = ks_initial_IonTracks(
        E_MeV_u=data.E_MeV_u,
        voltage_V=data.voltage_V,
        electrode_gap_cm=data.electrode_gap_cm,
        particle=data.particle,
        RDD_model="Gauss",
    )

    IonTracks_df = pd.concat([IonTracks_df, temp_df], ignore_index=True)
    print(IonTracks_df)

# add to the plot
sns.scatterplot(data=IonTracks_df, ax=ax, x="E_MeV_u", y="ks", label="IonTracks")
ax.set_ylabel("$k_s$")
fig.savefig("Jaffe_theory_and_IonTracks.pdf", bbox_inches="tight")

# save ion tracks results
IonTracks_df.to_csv("IonTracks.csv")
