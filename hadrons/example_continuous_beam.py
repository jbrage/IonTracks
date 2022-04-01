import numpy as np
from functions import IonTracks_continuous_beam
import matplotlib.pyplot as plt
import pandas as pd

E_MeV_u = 250
particle = "proton"
doserate_Gy_min = 100
voltage_V = 2000

IonTracks_df = pd.DataFrame()
for doserate_Gy_min in [10, 100, 1000]:
    result_dic = IonTracks_continuous_beam(E_MeV_u=E_MeV_u,
                                           voltage_V=voltage_V,
                                           particle=particle,
                                           doserate_Gy_min=doserate_Gy_min
                                           )
    print(result_dic)
    IonTracks_df = IonTracks_df.append(result_dic, ignore_index=True)

# plot
fig, ax = plt.subplots()
ax.plot(IonTracks_df["doserate_Gy_min"], IonTracks_df["ks_IonTracks"], "ro")
ax.set_xlim("log")

ax.set_title("{} ions, at {} Gy/mn".format(particle, doserate_Gy_min))
ax.set_xlabel("Inverse voltage (1/V)")
ax.set_ylabel("$k_s$ (IonTracks)")

fig.savefig("IonTracks_continuous_beam.pdf", bbox_inches="tight")
