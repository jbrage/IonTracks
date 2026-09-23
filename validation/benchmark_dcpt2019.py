"""Benchmark against the 2019 DCPT measurements (Christensen et al 2020, Phys. Med. Biol. 65 115003).

Measurement: Roos-type plane-parallel chamber (2 mm gap), 2 cm water depth, Varian ProBeam
(72.8 MHz isochronous cyclotron), 10 x 10 cm^2 single-energy layers of 41 x 41 spots (2.5 mm
spacing) at 70, 150, 226 and 244 MeV, dose-rate factors (DRF) 0.2, 0.5 and 1.0, charge collected
three times at 50, 67, 83, 125 and 200 V. Data: validation/data/dcpt2019/ (from branch
origin/backend-benchmark, commit e3c2721).

This is a blind prediction: no model input is adjusted to the measured charges. Each input comes
from the delivery records, from independent literature, or is a stated assumption.
* Delivery: the spot map, the dose per layer (``dose_Gy``) and the spot duration (``time_s``,
  machine logs), with 2 ms dead time between spots (assumed; not in the data).
* Spot size sigma(E) at the chamber: ProBeam in-air sigma at isocentre, 5.6 mm at 70 MeV to
  3.8 mm at 244 MeV (Lee et al 2020, J. Appl. Clin. Med. Phys.), log-interpolated in energy. The
  prediction is repeated at sigma -/+ 10 %, the site-to-site agreement of ProBeam in-air spot
  sizes (Langner et al 2017, J. Appl. Clin. Med. Phys.).
* Gas: Gas.air(particle="proton") at each measurement's T and p (Boag mobilities,
  alpha = 1.6e-6 cm^3/s, W_air = 34.44 eV from ICRU 90, free electrons after Goetz et al.).
  The Roos sensitive area (r = 7.8 mm) is sampled at 32 points.
* Initial (in-track) recombination: one proton track per energy and voltage in the 3D track
  model (Tile3D; N0 from the stopping power in air, Gaussian radius from the IonTracks v3 LET
  table). The surviving fraction f_init of every track feeds the general recombination:
  f = f_init * f_general(q f_init), the coupling validated in validation/VALIDATION_TILE3D.md.
* Physics: the prediction uses the default model (free electrons and space charge). The classic
  model (neither) is shown for comparison.

Because every spot is followed by a dead time longer than the chamber memory, each spot at each
point is an independent episode whose loss depends only on its local rate q. The fast evaluator
tabulates that loss with the v4 solver over a grid of q (per voltage, spot duration and gas), and
is checked against the full solve_field.

Run from the repository root:  python validation/benchmark_dcpt2019.py   (~1 min)
"""

import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotstyle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plotstyle import AXIS, INK2, S1, S2, S3  # noqa: E402

from iontracks import (  # noqa: E402
    Field,
    Gas,
    GaussianTrack,
    Grid1D,
    Grid3D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Tile3D,
    Uniform1D,
    solve_field,
    solve_point,
)
from iontracks.constants import MEV_TO_J  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation" / "data" / "dcpt2019"
FIG = ROOT / "validation" / "figures"
VOLTS = np.array([50.0, 67.0, 83.0, 125.0, 200.0])
GAP_CM, ROOS_R_CM = 0.2, 0.78
SPACING_CM, N_SIDE, DEAD_S = 0.25, 41, 2e-3
ENERGIES = (70, 150, 226, 244)
SIGMA_REL = 0.10  # site-to-site agreement of ProBeam in-air spot sigma (Langner et al 2017)
SIGMAS = {"nominal": 0.0, "small": -SIGMA_REL, "large": SIGMA_REL}
# IonTracks v3 single-track k_s - 1 at 200 V, 2 mm (origin/backend-benchmark comparison_initial.csv):
# the values used before v4 had a track model, kept only for comparison
LEGACY_INITIAL_200V = {70: 3.62e-4, 150: 2.07e-4, 226: 1.58e-4, 244: 1.51e-4}
CLASSIC = {"free_electrons": False, "space_charge": False}
PHYSICS = {
    "full": {},
    "classic": CLASSIC,
    "fe": {"space_charge": False},
    "sc": {"free_electrons": False},
}
MODELS = {k: Uniform1D(Grid1D(n_cells=100), **v) for k, v in PHYSICS.items()}
MODEL = MODELS["full"]
TRACK_GRID = Grid3D(n_xy=40, n_z=80)
POINTS = ParallelPlateChamber(GAP_CM, 200.0, sensitive_radius_cm=ROOS_R_CM).sample_points(4, 8)

plotstyle.apply()
DRF_COLOR = {0.2: S3, 0.5: S2, 1.0: S1}
E_MARKER = {70: "o", 150: "s", 226: "^", 244: "D"}


# ------------------------------------------------------------------ data
def load():
    ks = {
        (int(r["Energy_MeV"]), float(r["dose_rate_factor"])): r
        for r in csv.DictReader(open(DATA / "ks_200V.csv"))
    }
    charges = {}
    for r in csv.DictReader(open(DATA / "charges.csv", encoding="utf-8")):
        key = (int(r["Energy_MeV"]), float(r["DRF"]))
        q = np.array([float(r[k]) for k in ("Q1_nC", "Q2_nC", "Q3_nC")])
        q = q[~np.isnan(q)]
        # standard error of the repeat readings, floored at the 1e-4 reading resolution
        charges.setdefault(key, {})[float(r["voltage_V"])] = (
            q.mean(), max(q.std(ddof=1) / math.sqrt(q.size), 1e-4 * q.mean()),
            float(r["T_C"]), float(r["P_kPa"]),
        )  # fmt: skip
    settings = []
    for key in sorted(ks):
        c = charges[key]
        q200, s200, T, P = c[200.0]
        ratio = np.array([c[v][0] / q200 for v in VOLTS])
        err = np.array(
            [ratio[i] * math.hypot(c[v][1] / c[v][0], s200 / q200) for i, v in enumerate(VOLTS)]
        )
        err[-1] = 0.0
        settings.append(
            {"E": key[0], "drf": key[1], "row": ks[key], "ratio": ratio, "err": err, "T": T, "P": P}
        )
    return settings


def stopping_power(E, medium):
    rho = 1.20479e-3 if medium == "air" else 1.0
    d = np.genfromtxt(ROOT / "validation" / "data" / "stopping_power" / f"stopping_power_{medium}.csv",
                      delimiter=",", skip_header=3, names=True)  # fmt: skip
    return float(np.interp(E, d["E_MeV_u"], d["proton_LET_keV_um"])) * 10 / rho  # MeV cm^2/g


def sigma_literature_cm(E):
    """ProBeam in-air spot sigma at isocentre (Lee et al 2020), 5.6 mm (70 MeV) to 3.8 mm (244 MeV)."""
    frac = math.log(E / 70) / math.log(244 / 70)
    return (5.6 + (3.8 - 5.6) * frac) / 10


def gas_for(s):
    return Gas.air(T_K=273.15 + s["T"], p_hPa=10 * s["P"], particle="proton")


def raster(s, sigma_cm):
    """The 41 x 41 layer: serpentine order, spot duration from the logs, DEAD_S between spots."""
    E, row = s["E"], s["row"]
    fluence = float(row["dose_Gy"]) / (stopping_power(E, "water") * MEV_TO_J * 1e3)
    n_spot = fluence * SPACING_CM**2
    coords = (np.arange(N_SIDE) - (N_SIDE - 1) / 2) * SPACING_CM
    xs, ys = [], []
    for i, y in enumerate(coords):
        for x in coords if i % 2 == 0 else coords[::-1]:
            xs.append(x)
            ys.append(y)
    dwell = float(row["time_s"])
    t_on = np.arange(len(xs)) * (dwell + DEAD_S)
    return Field.from_spots(t_on, t_on + dwell, np.array(xs), np.array(ys), n_spot, sigma_cm,
                            stopping_power(E, "air"))  # fmt: skip


# ------------------------------------------------------------------ initial recombination
_INITIAL: dict = {}


def f_initial(E, V, gas, physics="full"):
    """Surviving fraction of one proton track (3D track model) at energy E and voltage V."""
    key = (E, V, physics)
    if key not in _INITIAL:
        track = GaussianTrack.from_stopping_power(stopping_power(E, "air"), gas)
        model = Tile3D(track, TRACK_GRID, **PHYSICS[physics])
        _INITIAL[key] = model.single_track(ParallelPlateChamber(GAP_CM, V), gas).f_recombination
    return _INITIAL[key]


# ------------------------------------------------------------------ fast evaluator
_TABLES: dict = {}


def loss_table(V, dwell, gas, physics="full"):
    """Loss fraction of one constant-rate spot episode, tabulated over the local rate q."""
    key = (V, round(dwell, 9), gas, physics)
    if key not in _TABLES:
        chamber = ParallelPlateChamber(GAP_CM, V)
        q = np.logspace(9, 15, 49)
        loss = []
        for qi in q:
            ep = MODELS[physics].run_episode(PiecewiseConstant.constant(qi, 0, dwell), chamber, gas)
            loss.append(ep.recombined_cm2 / ep.produced_cm2)
        _TABLES[key] = (np.log(q), np.log(np.array(loss) / q))  # loss / q is smooth in log q
    return _TABLES[key]


def f_general(s, sigma_cm, V, physics="full", scale=1.0):
    """Chamber collection efficiency for general recombination (charge-weighted over points).

    ``scale`` multiplies the ionisation (the fraction surviving initial recombination).
    """
    gas = gas_for(s)
    field = raster(s, sigma_cm)
    dwell = float(s["row"]["time_s"])
    lq, lh = loss_table(V, dwell, gas, physics)
    produced = recombined = 0.0
    for x, y, w in zip(POINTS.x_cm, POINTS.y_cm, POINTS.weight_cm2, strict=True):
        q = (
            field.fluence_per_spot_cm2(x, y)
            * gas.ion_pairs_per_cm(field.stopping_power_MeV_cm2_g)
            / dwell
            * scale
        )
        q = q[q > 0]
        loss = q * np.exp(np.interp(np.log(q), lq, lh))
        produced += w * np.sum(q)
        recombined += w * np.sum(q * loss)
    return 1.0 - recombined / produced


def predicted_ratio(s, sigma_cm, physics="full"):
    """Model Q(V)/Q(200 V): initial recombination (3D track) feeding general recombination."""
    gas = gas_for(s)
    f = []
    for V in VOLTS:
        f_init = f_initial(s["E"], V, gas, physics)
        f.append(f_init * f_general(s, sigma_cm, V, physics, f_init))
    f = np.array(f)
    return f / f[-1], f


def extrapolated_ks(ratio):
    """Linear fit of Q(200)/Q(V) vs 1/V^2 over all five voltages, k_s = 1/intercept."""
    slope, intercept = np.polyfit(1.0 / VOLTS**2, 1.0 / ratio, 1)
    return 1.0 / intercept


def tvm_ks(ratio):
    """Two-voltage method (TRS-398, continuous) with V1 = 200 V, V2 = 67 V, as in the paper's Fig. 5."""
    n2 = (VOLTS[-1] / VOLTS[1]) ** 2
    m1_over_m2 = 1.0 / ratio[1]  # ratio = Q(V)/Q(200 V); TRS-398 uses M(V1)/M(V2) with V1 > V2
    return (n2 - 1) / (n2 - m1_over_m2)


# ------------------------------------------------------------------ run
def main():
    t0 = time.perf_counter()
    settings = load()
    print("check: 1/V^2 extrapolation of the measured charges reproduces the dataset k_s")
    for s in settings:
        s["ks_meas"] = float(s["row"]["k_s"])
        s["ks_meas_refit"] = extrapolated_ks(s["ratio"])
    dev = max(abs(s["ks_meas_refit"] - s["ks_meas"]) for s in settings)
    print(f"  max |refit - dataset| = {dev:.1e}")

    print("check: fast evaluator vs full solve_field (244 MeV, DRF 1.0)")
    s244 = next(s for s in settings if s["E"] == 244 and s["drf"] == 1.0)
    sig = sigma_literature_cm(244)
    for V in (200.0, 50.0):
        fast = f_general(s244, sig, V)
        full = solve_field(raster(s244, sig), ParallelPlateChamber(GAP_CM, V, sensitive_radius_cm=ROOS_R_CM),
                           gas_for(s244), MODEL, points=POINTS).f_recombination  # fmt: skip
        print(
            f"  {V:4.0f} V: 1-f fast {1 - fast:.5e}  full {1 - full:.5e}  rel {abs((1 - fast) / (1 - full) - 1):.1e}"
        )
    s244["fast_check"] = abs((1 - fast) / (1 - full) - 1)

    print(f"blind prediction: literature spot size, and sigma -/+ {SIGMA_REL:.0%}")
    for s in settings:
        pred = {
            k: predicted_ratio(s, sigma_literature_cm(s["E"]) * (1 + d)) for k, d in SIGMAS.items()
        }
        s["pred"] = pred["nominal"][0]
        s["pred_classic"], f_classic = predicted_ratio(s, sigma_literature_cm(s["E"]), "classic")
        s["pred_fe"], _ = predicted_ratio(s, sigma_literature_cm(s["E"]), "fe")
        s["pred_sc"], _ = predicted_ratio(s, sigma_literature_cm(s["E"]), "sc")
        s["ks_true_classic"] = 1 / f_classic[-1]
        s["band"] = (
            np.minimum(pred["small"][0], pred["large"][0]),
            np.maximum(pred["small"][0], pred["large"][0]),
        )
        s["ks_true"] = {k: 1 / f[-1] for k, (_, f) in pred.items()}
        s["tvm_model"] = {k: tvm_ks(r) for k, (r, _) in pred.items()}
        s["tvm_meas"] = tvm_ks(s["ratio"])
        # a ratio is consistent if it lies within the sigma band, widened by 2x its repeat scatter
        lo, hi = s["band"]
        outside = np.maximum(0.0, np.maximum(lo - s["ratio"], s["ratio"] - hi))
        s["consistent"] = (outside <= 2 * s["err"])[:-1]
        # the earlier approach: continuous beam at the dataset's dose rate (classic physics)
        gas = gas_for(s)
        q = float(s["row"]["dose_rate_air_Gy_s"]) * gas.ion_pairs_per_cm3_per_Gy()
        f_c = []
        for V in VOLTS:
            ch = ParallelPlateChamber(GAP_CM, V)
            f_init = f_initial(s["E"], V, gas, "classic")
            fg = solve_point(
                PiecewiseConstant.constant(q * f_init, 0, 0.05 * 200 / V), ch, gas,
                MODELS["classic"],
            ).f_recombination  # fmt: skip
            f_c.append(f_init * fg)
        f_c = np.array(f_c)
        s["pred_cont"] = f_c / f_c[-1]
        print(f"  {s['E']:3d} MeV DRF {s['drf']:.1f}: max |dev| {np.max(np.abs(s['pred'] - s['ratio'])) * 100:.2f} %, "
              f"consistent {int(s['consistent'].sum())}/{s['consistent'].size}")  # fmt: skip

    make_figure(settings)
    write_report(settings, dev, time.perf_counter() - t0)
    print(f"done in {time.perf_counter() - t0:.0f} s")


def make_figure(settings):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.2))
    x = 1e6 / VOLTS**2  # 1/kV^2
    for a, E in zip(ax.ravel()[:4], ENERGIES, strict=True):
        for s in (s for s in settings if s["E"] == E):
            c = DRF_COLOR[s["drf"]]
            lo, hi = s["band"]
            a.fill_between(x, 1 / hi, 1 / lo, color=c, alpha=0.18, lw=0)
            a.plot(x, 1 / s["pred"], color=c, lw=1.2)
            a.plot(x, 1 / s["pred_classic"], color=c, lw=1.0, ls=":")
            a.errorbar(x, 1 / s["ratio"], yerr=s["err"] / s["ratio"] ** 2, fmt="o", color=c, ms=5,
                       capsize=2, label=f"DRF {s['drf']:g}, measured")  # fmt: skip
        a.plot(
            [],
            [],
            color=INK2,
            lw=1.2,
            label=f"v4 prediction, σ = {sigma_literature_cm(E) * 10:.1f} mm",
        )
        a.fill_between([], [], [], color=INK2, alpha=0.18, lw=0, label=f"σ ± {SIGMA_REL:.0%}")
        a.plot(
            [], [], color=INK2, lw=1.0, ls=":", label="classic model (no free e⁻, no space charge)"
        )
        a.set(
            xlabel="1 / V²  (1/kV²)",
            ylabel="Q(200 V) / Q(V)",
            title=f"({'abcd'[ENERGIES.index(E)]}) {E} MeV",
        )
        a.legend(loc="upper left")

    # (e) k_s at 200 V: TVM(200/67) from measured charges vs from predicted charges, and the true k_s
    a = ax[1, 1]
    tvm_meas = np.array([s["tvm_meas"] - 1 for s in settings]) * 100

    def model(key):
        nom = np.array([s[key]["nominal"] - 1 for s in settings]) * 100
        lo = np.array([min(s[key].values()) - 1 for s in settings]) * 100
        hi = np.array([max(s[key].values()) - 1 for s in settings]) * 100
        return nom, np.vstack([nom - lo, hi - nom])

    tvm_nom, tvm_err = model("tvm_model")
    true_nom, true_err = model("ks_true")
    top = max((true_nom + true_err[1]).max(), (tvm_nom + tvm_err[1]).max(), tvm_meas.max())
    lim = [0, top * 1.08]
    a.plot(lim, lim, color=AXIS, lw=0.8)
    a.errorbar(tvm_meas, tvm_nom, yerr=tvm_err, fmt="s", color=S2, ms=5, capsize=2,
               label="v4 predicted charges, same TVM (like with like)")  # fmt: skip
    a.errorbar(tvm_meas, true_nom, yerr=true_err, fmt="o", color=S1, ms=5, capsize=2,
               label="v4 true k_s = P / (P − R)")  # fmt: skip
    a.set(xlabel="measured k_s − 1 at 200 V, TVM 200/67 V (%)", ylabel="predicted k_s − 1 at 200 V (%)",
          title="(e) k_s at 200 V: extraction method vs true value", xlim=lim, ylim=lim)  # fmt: skip
    a.legend(loc="upper left")

    # (f) residuals of every ratio against the prediction, with the sigma band behind each point
    a = ax[1, 2]
    a.axhline(0, color=AXIS, lw=0.8)
    for s in settings:
        dx = (ENERGIES.index(s["E"]) - 1.5) * 2.2 + (s["drf"] - 0.6) * 4
        xv = VOLTS[:-1] + dx
        lo, hi = s["band"]
        c = DRF_COLOR[s["drf"]]
        a.vlines(
            xv, (lo - s["pred"])[:-1] * 100, (hi - s["pred"])[:-1] * 100, color=c, alpha=0.3, lw=4
        )
        a.errorbar(xv, (s["ratio"] - s["pred"])[:-1] * 100, yerr=2 * s["err"][:-1] * 100,
                   fmt=E_MARKER[s["E"]], color=c, ms=4.5, capsize=0, lw=1.0)  # fmt: skip
    handles = [
        Line2D([], [], color=DRF_COLOR[d], lw=4, alpha=0.5, label=f"DRF {d:g}")
        for d in (0.2, 0.5, 1.0)
    ]
    handles += [
        Line2D([], [], color=INK2, marker=E_MARKER[E], ls="", ms=4.5, label=f"{E} MeV")
        for E in ENERGIES
    ]
    a.legend(handles=handles, loc="lower right", ncol=2)
    a.set(xlabel="voltage (V)", ylabel="measured − predicted Q(V)/Q(200 V) (%)",
          title="(f) Residuals: bars = 2 × repeat scatter, shade = σ ± 10 %")  # fmt: skip
    fig.suptitle("Figure 6 · Blind benchmark: DCPT 2019, Roos chamber, 2 mm, spot-scanned 41 × 41 layers",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig6_dcpt2019_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(settings, refit_dev, runtime):
    def rng(d, fmt=".4f"):
        return f"{d['nominal']:{fmt}} ({min(d.values()):{fmt}}–{max(d.values()):{fmt}})"

    rows = "\n".join(
        f"| {s['E']} | {s['drf']:g} | {float(s['row']['time_s']) * 1e3:.1f} | {s['tvm_meas']:.4f} "
        f"| {rng(s['tvm_model'])} | {rng(s['ks_true'])} | {s['ks_meas']:.4f} "
        f"| {np.max(np.abs(s['pred'] - s['ratio'])) * 100:.2f} "
        f"| {int(s['consistent'].sum())}/{s['consistent'].size} "
        f"| {np.max(np.abs(s['pred_cont'] - s['ratio'])) * 100:.2f} |"
        for s in settings
    )
    n_cons = sum(int(s["consistent"].sum()) for s in settings)
    n_all = sum(s["consistent"].size for s in settings)
    tvm_dev = np.array(
        [(s["tvm_model"]["nominal"] - 1) / (s["tvm_meas"] - 1) - 1 for s in settings]
    )
    true_over_tvm = np.array(
        [(s["ks_true"]["nominal"] - 1) / (s["tvm_model"]["nominal"] - 1) for s in settings]
    )
    sens_small = np.array(
        [(s["ks_true"]["small"] - 1) / (s["ks_true"]["nominal"] - 1) - 1 for s in settings]
    )
    sens_large = np.array(
        [(s["ks_true"]["large"] - 1) / (s["ks_true"]["nominal"] - 1) - 1 for s in settings]
    )
    pred_dev = max(np.max(np.abs(s["pred"] - s["ratio"])) for s in settings) * 100
    classic_dev = max(np.max(np.abs(s["pred_classic"] - s["ratio"])) for s in settings) * 100
    rms = (
        np.sqrt(np.mean(np.concatenate([(s["pred"] - s["ratio"])[:-1] for s in settings]) ** 2))
        * 100
    )
    rms_classic = (
        np.sqrt(
            np.mean(np.concatenate([(s["pred_classic"] - s["ratio"])[:-1] for s in settings]) ** 2)
        )
        * 100
    )

    def stats(key):
        res = np.concatenate([(s[key] - s["ratio"])[:-1] for s in settings])
        return np.max(np.abs(res)) * 100, np.sqrt(np.mean(res**2)) * 100

    decomp = "\n".join(
        f"| {name} | {stats(key)[0]:.2f} % | {stats(key)[1]:.2f} % |"
        for name, key in (
            ("default (free electrons + space charge)", "pred"),
            ("free electrons only", "pred_fe"),
            ("space charge only", "pred_sc"),
            ("classic (neither)", "pred_classic"),
        )
    )
    decomp_indented = "\n".join("   " + line for line in decomp.splitlines())
    gas0 = gas_for(settings[0])
    init_rows = "\n".join(
        f"| {E} | {GaussianTrack.from_stopping_power(stopping_power(E, 'air'), gas0).radius_cm * 1e4:.0f} | "
        + " | ".join(f"{1 / _INITIAL[(E, V, 'full')] - 1:.2e}" for V in VOLTS)
        + f" | {1 / _INITIAL[(E, 200.0, 'classic')] - 1:.2e} | {LEGACY_INITIAL_200V[E]:.2e} |"
        for E in ENERGIES
    )
    cont_dev = max(np.max(np.abs(s["pred_cont"] - s["ratio"])) for s in settings) * 100
    fast_check = next(s for s in settings if "fast_check" in s)["fast_check"]
    sig_rows = " · ".join(f"{E} MeV: {sigma_literature_cm(E) * 10:.2f} mm" for E in ENERGIES)
    (ROOT / "validation" / "BENCHMARK_DCPT2019.md").write_text(
        f"""# Benchmark: DCPT 2019 recombination measurements

Generated by `python validation/benchmark_dcpt2019.py` ({runtime:.0f} s). The measurement set-up
is a Roos chamber (2 mm gap, r = 7.8 mm) at 2 cm water depth in Varian ProBeam 10 × 10 cm² layers
of 41 × 41 spots. Charges were collected at 5 voltages × 3 dose-rate factors (DRF) × 4 energies.
Reference: Christensen et al 2020, Phys. Med. Biol. 65 115003.

**This is a blind prediction. No model input is adjusted to the measured charges.** The table
below lists every input and where it comes from.

| Input | Value | Source |
|---|---|---|
| Spot map, dose per layer, spot duration | 41 × 41 at 2.5 mm; `dose_Gy`; `time_s` | Delivery records in the dataset |
| Dead time between spots | 2 ms | Assumed (not in the data) |
| Spot σ at the chamber | {sig_rows} | ProBeam in-air σ at isocentre, 5.6 mm (70 MeV) to 3.8 mm (244 MeV), [Lee et al 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7170285/), log-interpolated |
| Spot σ uncertainty | ± {SIGMA_REL:.0%} | Site-to-site agreement of ProBeam in-air σ, [Langner et al 2017](https://doi.org/10.1002/acm2.12078) |
| µ⁺, µ⁻, α, D± | 1.36, 2.1 cm²/Vs; 1.6e-6 cm³/s; Einstein | Boag; `Gas.air(particle="proton")` at each measurement's T and p |
| W_air | 34.44 eV | ICRU 90 (protons) |
| Free electrons | mobility 830 cm²/Vs, attachment rate γ(E) | Götz et al. 2013, as tabulated by Fenwick & Kumar 2023 |
| Space charge | on | Gauss's law for the ion charge (the default model) |
| Initial recombination | one track per energy and voltage in the 3D track model | N0 from the stopping power in air; Gaussian radius from the IonTracks v3 LET table (20 µm floor for protons) |

The dose per layer agrees with the chamber's own 200 V reading × s_w,air ≈ 1.13, so it is
probably that reading. It sets the ionisation density, which the voltage dependence cannot
constrain, so it isn't a tuned parameter.

![Figure 6](figures/fig6_dcpt2019_benchmark.png)

## Results

1. **Voltage dependence of the charge.** The prediction reproduces Q(V)/Q(200 V) for all 10
   settings to within {pred_dev:.2f} % (RMS {rms:.2f} %). Of the {n_all} ratios below 200 V,
   {n_cons} lie inside the σ ± {SIGMA_REL:.0%} band widened by twice their repeat scatter. The
   classic model (no free electrons, uniform field) deviates by up to {classic_dev:.2f} % (RMS
   {rms_classic:.2f} %). The earlier approach (a continuous beam at the dataset's dose rate)
   deviates by up to {cont_dev:.2f} %.
2. **k_s at 200 V, extracted like for like.** Applying the same TVM (200/67 V) to measured and
   predicted charges, the mean relative difference in k_s − 1 is {np.mean(tvm_dev) * 100:+.0f} %
   (spread {np.std(tvm_dev) * 100:.0f} %, nominal σ).
3. **True k_s versus TVM.** The model's true k_s − 1, from P / (P − R), is on average
   {np.mean(true_over_tvm):.2f} times the value the TVM extracts from the same predicted charges
   (range {true_over_tvm.min():.2f}–{true_over_tvm.max():.2f}). The TVM assumes that 1/Q falls
   as 1/V², but three parts of the model do not:
   - initial recombination goes as 1/V;
   - the free-electron fraction grows with the field;
   - the spot structure varies the local dose rate across the chamber and in time.

   This is a model result, not a measurement, and it needs an independent check: a
   higher-voltage series or a calorimetric reference.
4. **Which physics does what.** Deviation of Q(V)/Q(200 V) over the 40 ratios below 200 V:

   | Model | max | RMS |
   |---|---|---|
{decomp_indented}

   Free electrons alone change little. Space charge makes the agreement worse. At 50 V and the
   highest dose rate the space-charge parameter η = e q d⁴ / (ε₀ µ⁺ V²) reaches about 3 at a spot centre,
   so the model adds loss at low voltage, and the predicted voltage dependence steepens beyond
   what was measured. Two things set the size of that term:
   - the ion parameters: η ∝ 1/µ⁺, and Boag's µ⁺ = 1.36 cm²/Vs is at the low end of published
     values (the free-electron models of Götz, Kranzer and Fenwick & Kumar use 1.87 cm²/Vs);
   - the laterally uniform (1D) field, while a spot is only about twice as wide as the gap.

   The Risø benchmark shows the same space-charge excess in its field ratios. Choosing between
   these needs independent data, not a fit to these charges. It is an open item.

## Initial recombination from the 3D track model

k_s − 1 of one track (default physics) at each voltage, the classic-physics value at 200 V, and
the IonTracks v3 values that were used before v4 had a track model. b is the Gaussian track radius.

| E (MeV) | b (µm) | 50 V | 67 V | 83 V | 125 V | 200 V | 200 V, classic | 200 V, IonTracks v3 |
|---|---|---|---|---|---|---|---|---|
{init_rows}

## Uncertainties

* **Measurement.** Error bars are the standard error of the repeat readings (two or three per
  voltage), floored at the 1e-4 reading resolution. The charges are not monitor-normalised and
  each voltage is a separate delivery, so the true uncertainty also contains the
  delivery-to-delivery reproducibility. That term is not in the data, and none is added.
* **Spot size.** A 10 % smaller σ raises the predicted k_s − 1 by
  {sens_small.min() * 100:.0f}–{sens_small.max() * 100:.0f} %, and a 10 % larger σ lowers it by
  {-sens_large.max() * 100:.0f}–{-sens_large.min() * 100:.0f} %. The effect is smallest at low dose
  rates, where initial recombination dominates. General recombination scales as α/σ², so a 10 %
  error in α acts like a 5 % error in σ. The band covers σ only.
* **Not propagated.** The spot sizes are from another ProBeam site, in air at isocentre. They are
  interpolated in energy between 70 and 244 MeV. Multiple scattering over 2 cm of water adds
  ≲ 0.3 mm in quadrature, which is negligible. The dead time matters only if it is shorter than
  the chamber memory (about 0.8 ms at 50 V).

## Checks on the set-up

* The dataset's `k_s` column is not reproduced by a 5-point linear fit in 1/V² (max difference
  {refit_dev:.1e}), so it was derived some other way. TVM with 200/67 V applied to the raw charges
  reproduces the TVM point in Fig. 5(a) of the paper (for example 1.0054 at 244 MeV, DRF 1). Model
  and measurement are therefore compared with TVM on both sides.
* The fast evaluator (tabulated single-spot episodes) against the full `solve_field` at 244 MeV,
  DRF 1, 50 V: relative difference in 1 − f of {fast_check:.1e}.

## Per setting

Values in brackets span σ ± {SIGMA_REL:.0%}. "Max Δ" is the largest absolute deviation of
Q(V)/Q(200 V) over the five voltages, in percent. "Consistent" counts the ratios inside the band
widened by 2 × their repeat scatter.

| E (MeV) | DRF | spot (ms) | measured k_s (TVM) | predicted k_s (TVM) | predicted true k_s | dataset k_s | max Δ (%) | consistent | max Δ, continuous (%) |
|---|---|---|---|---|---|---|---|---|---|
{rows}

## Notes and open questions

* **Chamber.** The paper states a Roos-type chamber with 2 mm electrode spacing. The dataset README
  on `origin/backend-benchmark` says "Advanced Markus, 2 mm", but the PTW Advanced Markus has a
  1 mm gap. That label should be corrected.
* **Dose rates.** The dataset's `dose_rate_*` columns equal D_field / (≈ 38 × t_spot) for every
  setting. That is higher than the effective dose rates in Fig. 3(a) of the paper by 1.5× (244 MeV)
  to 2.7× (70 MeV). The spot model uses neither: it uses the spot map, dose, spot duration and σ
  directly.
* **Would tighten the prediction.** DCPT's own spot sizes at 2 cm depth, the dead time between
  spots from the logs, and the delivery-to-delivery reproducibility of the charge readings.
* **Model inputs without independent data here.** The track radius (v3 LET table, floor 20 µm)
  and the free-electron attachment rate below 1000 V/cm (the Götz parameterisation is stated
  for ≥ 1000 V/cm and is extrapolated to 250 V/cm at 50 V).
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
