import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lmfit.models import LinearModel


Geiss_df = pd.read_csv("data/result.csv")


Kanai_df = pd.read_csv("data/Kanai_data_gradients.csv")

Jaffe_df = pd.read_csv("data/data_Jaffe.csv")
Jaffe_df["water_LET_keV_um"] = Jaffe_df["LET_keV_um"] * 1000 # 1225
# %%

d_cm = 0.2
Jaffe_df["inv_E_cm_V"] = d_cm / Jaffe_df["voltage_V"]

for particle_name, group in Jaffe_df.groupby("particle"):
    gradient = (group["ks_Jaffe"] - group["ks_Jaffe"].min()) / (group["inv_E_cm_V"] - group["inv_E_cm_V"].min())
    Jaffe_df.loc[Jaffe_df.particle==particle_name, "gradient"] = gradient

for particle_name in Jaffe_df.particle.unique():
    for E_MeV in Jaffe_df.E_MeV_u.unique():
        condition = (Jaffe_df.E_MeV_u == E_MeV) & (Jaffe_df.particle == particle_name)
        data = Jaffe_df.loc[condition]
        
        model = LinearModel()
        pars = model.guess(data["ks_Jaffe"], x=data["inv_E_cm_V"])
        fit_result = model.fit(data["ks_Jaffe"], pars, x=data["inv_E_cm_V"])
        
        Jaffe_df.loc[condition, "slope"] = fit_result.params["slope"].value

Jaffe_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# %%
data = Jaffe_df[Jaffe_df.E_MeV_u == 11]
fig, ax = plt.subplots()
sns.lineplot(data=data, x="inv_E_cm_V", y="ks_Jaffe", hue="particle", ax=ax)


# %%

data = Jaffe_df[Jaffe_df.voltage_V == 250].dropna()
fig, ax = plt.subplots()

# data["LET_keV_um"] = data["LET_keV_um"] / 2

# sns.lineplot(data=data, x="water_LET_keV_um", y="gradient", hue="particle", ax=ax, lw=1, style="particle")
sns.lineplot(data=data, x="water_LET_keV_um", y="slope", hue="particle", ax=ax, lw=1, style="particle")
sns.scatterplot(data=Kanai_df, x="LET_keV_um", y="gradient", hue="particle", ax=ax, s=15, style="particle")


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"LET$_\mathrm{water}$ (keV/$\mathrm{\mu}$m)")
ax.set_ylabel(r"Slope $\Delta k_s / \Delta E ^{-1}$  (V/cm)")


ax.set_xlim([10, 1e4])
ax.set_ylim([1, 1e3])
fig.savefig("plot.pdf")


# %%
fig, ax = plt.subplots()

data = Jaffe_df[Jaffe_df.voltage_V == 200]

sns.lineplot(data=data, x="E_MeV_u", y="ks_Jaffe", hue="particle")


# data = Geiss_df[(Geiss_df.voltage_V == 200) & (Geiss_df.a0_nm == 1.0)]
# sns.scatterplot(data=data, x="LET_keV_um", y="ks_Gauss", hue="particle")
# # sns.scatterplot(data=data, x="LET_keV_um", y="ks_Geiss", hue="particle")
ax.set_xscale("log")

ax.set_ylim(ymin=1)

# # %%
# data = Jaffe_df[Jaffe_df.voltage_V == 200]
# fig, ax = plt.subplots()
# sns.lineplot(data=data, x="LET_keV_um", y="ks_Jaffe", hue="particle")


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
        
        
