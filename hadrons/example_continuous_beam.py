from functions import IonTracks_continuous_beam
import matplotlib.pyplot as plt
import pandas as pd
import itertools


# set parameters
data_dict = dict(
    electrode_gap_cm = [0.2],
    particle = ["proton", "carbon"], 
    voltage_V = [200],
    E_MeV_u = [250],
    doserate_Gy_min = [1, 10, 100, 1000]
    )

# create a data frame with all the variables
data_df = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()), 
                                    columns=data_dict.keys())

IonTracks_df = pd.DataFrame()
for idx, data in data_df.iterrows():
    result_dic = IonTracks_continuous_beam(E_MeV_u=data.E_MeV_u,
                                           voltage_V=data.voltage_V,
                                           particle=data.particle,
                                           doserate_Gy_min=data.doserate_Gy_min
                                           )
    print(result_dic)
    IonTracks_df = IonTracks_df.append(result_dic, ignore_index=True)

# plot
fig, ax = plt.subplots()
ax.plot(IonTracks_df["doserate_Gy_min"], IonTracks_df["ks_IonTracks"], "ro")
ax.set_xscale("log")
ax.set_xlabel("Inverse voltage (1/V)")
ax.set_ylabel("$k_s$ (IonTracks)")

fig.savefig("IonTracks_continuous_beam.pdf", bbox_inches="tight")
