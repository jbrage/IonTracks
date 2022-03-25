import pandas as pd
import matplotlib.pyplot as plt
from scanned_beam import gaussian_density
import seaborn as sns
from lmfit.models import GaussianModel
import numpy as np
from scipy.integrate import simps, quad
from glob import glob

# find all results
datafiles = glob("data/full_beam*.csv")

# combine all results into a single df
df = pd.DataFrame()
print("\nFiles:")
for filename in datafiles:
    print("\t", filename)
    df_temp = pd.read_csv(filename)
    df = pd.concat([df_temp, df], ignore_index=True)

df.to_csv("data/full_beam_df.csv", index=False)

print("\n")
fig, ax = plt.subplots()

sns.scatterplot(data=df, x="r_cm", y="ks", hue="doserate_Gy_s", ax=ax)
# sns.lineplot(data=df, x="r_cm", y="density_r", hue="doserate_Gy_s", ax=ax)

x = np.linspace(0, df["r_cm"].max(), 500)

res_df = pd.DataFrame()

for (dr, d_cm), group in df.groupby(["doserate_Gy_s", "d_cm"]):

    # prepare model fit
    model = GaussianModel()
    pars = model.make_params(center=0, sigma=df.sigma_cm.unique()[0])
    fit = model.fit(group["ks"]-1, pars, x=group["r_cm"])

    def get_recombination(r):
        return fit.eval(x=r) + 1

    ax.plot(x, get_recombination(x))
    ax.plot(x, gaussian_density(x), "--")

    # func = lambda s: gaussian_density(s) * get_recombination(s)
  
    rr = gaussian_density(x) * get_recombination(x)

    average_ks = simps(rr, x)    
    weight = simps(gaussian_density(x), x) 
    ks = average_ks / weight

    print("d = {:0.0f} mm,\tGy/s = {:0.0f},\tks = {:0.8f}".format(d_cm*10, dr, ks))
    row = {"doserate_Gy_s": dr, "ks": ks, "d_cm": d_cm}
    res_df = res_df.append(row, ignore_index=True)

fig.savefig("r_plot.pdf")


# =========================================================

df = pd.read_csv("data/recombination_data_Robert.csv")

currennt_nA_to_Gy_s = 3600 / 800 # 800 nA = 3600 Gy/s
df["Doserate_Gy_s"] = df["current_nA"] * currennt_nA_to_Gy_s

MU_to_Gy = 140 # 175 MU/Gy for high dose-rates, 130 MU/Gy for lower
df["Dose_Gy"] = df["MU"] / MU_to_Gy
df["Dose_Gy"] = df["Dose_Gy"].astype(int)


x = "Doserate_Gy_s"
# x = "current_nA"
fig, ax = plt.subplots()
for D_Gy, group in df.groupby("Dose_Gy"):

    if D_Gy < 100: c="gray"
    else: c = "darkred"
     
    label = r"Experiment: $\approx${:0.0f} Gy".format(D_Gy)
    ax.plot(group[x], group["M1"], ".-", c=c, label=label, lw=1)
    ax.plot(group[x], group["M2"], ".-", c=c, lw=1)

#sns.lineplot(data=df, x=x, y="M1", hue="Dose_Gy", ax=ax, legend=False)
#sns.lineplot(data=df, x=x, y="M2", hue="Dose_Gy", ax=ax, legend=True)

ax.text(2000, 0.97, "$d=5.0\,$mm", ha="center", va="center")
ax.text(1000, 0.55, "$d=10\,$mm", ha="center", va="center")

ax.set_xlabel("Dose-rate (Gy/s)")
ax.set_ylabel("Collection efficiency")

ax.set_xscale("log")
ax.grid(True)


for d_cm, group in res_df.groupby("d_cm"):
    label = "Theory (IonTracks), d = {:0.1f} cm".format(d_cm)
    # ax.errorbar(group["doserate_Gy_s"], 1./group["ks"],ms=10, zorder=10, label=label, fmt=".")
    label = r"Theory (IonTracks)"
    ax.errorbar(group["doserate_Gy_s"], 1./group["ks"],ms=10, zorder=10, fmt=".", c="k", mfc="w", label=label)


handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

#ax.legend() # title="Dose (Gy)")
# ax.set_yscale("log")

fig.savefig("plot_doserates.pdf", bbox_inches="tight")

