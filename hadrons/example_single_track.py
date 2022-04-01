import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions import Jaffe_theory, ks_initial_IonTracks
import itertools


# set parameters
data_dict = dict(
    electrode_gap_cm = [0.2],
    particle = ["proton", "carbon"], # proton, helium, argon, iron
    voltage_V = [100, 200, 300],
    E_MeV_u = np.linspace(1, 250, 100),
    )

# create a data frame with all the variables
data_df = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), 
                                    columns=data_dict.keys())

# use the Jaffe theory for initial recombination for these parameters
Jaffe_df = pd.DataFrame()
for idx, data in data_df.iterrows():
    result = Jaffe_theory(data.E_MeV_u, data.voltage_V, data.electrode_gap_cm, particle=data.particle, input_is_LET=False)
    Jaffe_df = Jaffe_df.append(result, ignore_index=True)

print(Jaffe_df.head())
print(Jaffe_df["ks_Jaffe"])

# plot the results
fig, ax = plt.subplots()
ax.set_title("{} ion, electrode gap = {} cm".format(particle, electrode_gap_cm))
sns.lineplot(data=Jaffe_df, x="E_MeV_u", y="ks_Jaffe", hue="voltage_V", ax=ax)
fig.savefig("Jaffe_example.pdf", bbox_inches="tight")

# use IonTracks to calculate the same
IonTracks_result = ks_initial_IonTracks(E_MeV_u,
                                        voltage_V,
                                        electrode_gap_cm,
                                        particle=particle,
                                        RDD_model="Gauss")
print(IonTracks_result)
