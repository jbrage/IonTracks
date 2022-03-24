import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lmfit.models import LinearModel
from scipy.interpolate import interp1d

Geiss_df = pd.read_csv("data/result.csv")
Kanai_df = pd.read_csv("data/Kanai_data_gradients.csv")
Jaffe_df = pd.read_csv("data/data_Jaffe.csv")



voltages = Geiss_df.voltage_V.unique()

fig, axes = plt.subplots(nrows=1, ncols=len(voltages), sharey=True)
axes = axes.flatten()

for idx, (ax, VOLTAGE) in enumerate(zip(axes, voltages)):

    ax.set_title("{:0.0f} V".format(VOLTAGE))

    data = Jaffe_df[Jaffe_df.voltage_V == VOLTAGE]
    sns.lineplot(data=data, x="LET_keV_um", y="ks_Jaffe", hue="particle", ax=ax)

    Jaffe_interpol = interp1d(data["LET_keV_um"], data["ks_Jaffe"])

    data = Geiss_df[(Geiss_df.voltage_V == VOLTAGE)  & (Geiss_df.scale == 3) ]
    sns.scatterplot(data=data, x="LET_keV_um", y="ks_Geiss", hue="a0_nm", ax=ax)

    for a0 in Geiss_df["a0_nm"].unique():
        Geiss = data.loc[data["a0_nm"] == a0, "ks_Geiss"]
        Jaffe = Jaffe_interpol(Geiss)

        rsq = sum((Geiss - Jaffe)**2 / Jaffe)
        print(VOLTAGE, a0, rsq)


    data = Geiss_df[(Geiss_df.voltage_V == VOLTAGE) & (Geiss_df.a0_nm == 8.5) & (Geiss_df.scale == 3)]
    sns.scatterplot(data=data, x="LET_keV_um", y="ks_Geiss", hue="a0_nm", ax=ax)        

    ax.set_xscale("log")
ax.set_ylim(ymin=0.98)
fig.savefig("plot.pdf")


# # %% 
# for a0_nm, group in Geiss_df.groupby("a0_nm"):
#     sub_Geiss_df = group [(group.beta == 0) 
#                           # & (group.a0_nm == 1.5)
#                           & (group.scale == 3.0)]
    
#     fig, ax = plt.subplots()    
#     ax.set_title("a0_nm = {}".format(a0_nm))
    
#     sns.lineplot(x="LET_keV_um", y="ks_IT", hue="voltage_V", data=sub_Geiss_df, ax=ax)
#     sns.scatterplot(data=Jaffe_df, x="LET_keV_um", y="ks_Jaffe", hue="voltage_V")
    
#     ax.set_xlim(xmax=0.005, xmin=0)
#     ax.set_ylim(ymax=1.003, ymin=0.999)
        
        
