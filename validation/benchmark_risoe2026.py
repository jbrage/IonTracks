"""Benchmark against pulsed-beam extrapolation-chamber measurements (Varian linac, Risø, 31.03.2026).

Measurement: extrapolation chamber (collecting electrode r = 15 mm) 71.4 cm from the virtual
source, 20 x 20 cm^2 field, 50 MU per reading, 1014.38 hPa. 16 MeV electrons at 100 and
1000 MU/min, 10 MV FFF photons at 400 and 2400 MU/min. Gaps 0.5, 1, 2, 5, 8 and 10 mm at 10, 20
and 40 V/mm, each read at both polarities. Pulse counts per 50 MU come from a separate session
(09.04.2026). Data: validation/data/risoe2026/ (from IonTracks-AMM, commit 538874c).

This is a blind prediction: no model input is adjusted to the recombination data.
* Charge per reading: the mean of the two polarities, which removes the polarity effect to first
  order.
* Ionisation per MU: the saturation charge per unit gap at the LOW dose rate of each beam. It uses
  only the two smallest gaps at 40 V/mm (the extrapolation-chamber slope, which cancels a gap
  offset), corrected for the model's own small loss there, and divided by e x area. The
  high-dose-rate settings reuse it (same ionisation per MU), so their charges are not used as
  input at all. Sensitivity band: ionisation -/+ 10 %.
* Pulse train: pulses per 50 MU and the period from pulses_per_MU.csv, with 4 us rectangular pulses
  (assumed; checked at 2 and 6 us). The high-dose-rate records are read two ways:
  - H1: the counts as recorded;
  - H2: constant dose per pulse, i.e. the repetition rate scales with the dose rate. H2 is
    motivated by the counter's apparent ~120 Hz ceiling, which shows in the same file.
* Gas: Gas.air(particle="electron") at 20 C (assumed; not in the data) and 1014.38 hPa.
* Physics: the prediction uses the default model (free electrons after Goetz et al., space
  charge). The classic model (neither) is shown for comparison under H2.

Observables (no gap offset or inverse-square correction needed):
* the field ratio Q(d, E) / Q(d, 40 V/mm) for E = 20 and 10 V/mm, at fixed beam, dose rate and gap;
* the dose-rate ratio Q_high / Q_low at the same gap and voltage (both 50 MU).

Run from the repository root:  python validation/benchmark_risoe2026.py   (~1 min)
"""

import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotstyle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plotstyle import AXIS, INK2, S1, S2, S3, SURFACE  # noqa: E402

from iontracks import (  # noqa: E402
    Gas,
    Grid1D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Uniform1D,
    solve_point,
)
from iontracks.constants import ELEMENTARY_CHARGE_C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation" / "data" / "risoe2026"
FIG = ROOT / "validation" / "figures"
AREA_CM2 = math.pi * 1.5**2  # r = 15 mm, implied by the dataset's electrons_s_cm3 column
SOURCE_MM = 714.0  # chamber distance from the virtual source
T_K, P_HPA = 293.15, 1014.38
MU = 50.0
TAU_S = 4e-6
N0_REL = 0.10
GAPS = (0.5, 1.0, 2.0, 5.0, 8.0, 10.0)
FIELDS = (40, 20, 10)  # V/mm; 40 V/mm is the reference of the field ratio
READING_NC = 0.05  # resolution of the recorded charges
BEAMS = {"16 MeV": (100, 1000), "10 FFF": (400, 2400)}
BEAM_NAME = {"16 MeV": "16 MeV electrons", "10 FFF": "10 MV FFF photons"}
PULSE_FILES = {
    ("16 MeV", 100): ("10413",),
    ("16 MeV", 1000): ("10422", "10444"),
    ("10 FFF", 400): ("10475",),
    ("10 FFF", 2400): ("10477",),
}
MODELS = {
    "full": Uniform1D(Grid1D(n_cells=200)),
    "classic": Uniform1D(Grid1D(n_cells=200), free_electrons=False, space_charge=False),
    "fe": Uniform1D(Grid1D(n_cells=200), space_charge=False),
    "sc": Uniform1D(Grid1D(n_cells=200), free_electrons=False),
}
GAS = Gas.air(T_K=T_K, p_hPa=P_HPA, particle="electron")

plotstyle.apply()
FIELD_COLOR = {40: S1, 20: S2, 10: S3}
SETTING_MARKER = {
    ("16 MeV", 100): "o",
    ("16 MeV", 1000): "s",
    ("10 FFF", 400): "^",
    ("10 FFF", 2400): "D",
}


# ------------------------------------------------------------------ data
def load_charges():
    """{(beam, MU/min, d_mm, V/mm): (polarity mean, Q+, Q-)} in nC."""
    out = {}
    with open(DATA / "data_extrapolationChamber.csv", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            r = {k.strip(): v for k, v in raw.items()}
            d, V = float(r["d_mm"]), float(r["Voltage_pm_V"])
            qp, qn = float(r["Charge_positive_nC"]), float(r["Charge_negative_nC"])
            out[(r["label"].strip(), int(r["Dose_rate MU_min"]), d, round(V / d))] = (
                0.5 * (qp + qn),
                qp,
                qn,
            )
    return out


def load_pulses():
    """Pulses per 50 MU and pulse period as recorded, per setting."""
    rows = {r["FileNumber"]: r for r in csv.DictReader(open(DATA / "pulses_per_MU.csv"))}
    counts = {}
    for key, files in PULSE_FILES.items():
        n = np.mean([float(rows[f]["NumPulses"]) * MU / float(rows[f]["MUs"]) for f in files])
        period = np.mean([float(rows[f]["AvgTimeBetweenPulses_s"]) for f in files])
        counts[key] = (float(n), float(period))
    return counts, rows


def pulse_structures(counts):
    """H1: counts as recorded. H2: dose per pulse of the low dose rate, rate scaled with MU/min."""
    h2 = {}
    for beam, (low, high) in BEAMS.items():
        n, period = counts[(beam, low)]
        h2[(beam, low)] = (n, period)
        h2[(beam, high)] = (n, period * low / high)
    return {"H1": dict(counts), "H2": h2}


def k_di(d_mm):
    """Inverse-square correction to the front electrode (fluence at the gap mid-plane)."""
    return 1.0 + d_mm / SOURCE_MM


# ------------------------------------------------------------------ model
_CACHE: dict = {}


def efficiency(d_mm, field_V_mm, n0, n_pulses, period_s, tau_s=TAU_S, physics="full"):
    """Collection efficiency (general recombination) of the whole 50 MU pulse train."""
    key = (d_mm, field_V_mm, round(n0, -3), round(n_pulses), round(period_s, 9), tau_s, physics)
    if key not in _CACHE:
        chamber = ParallelPlateChamber(d_mm / 10, field_V_mm * d_mm)
        source = PiecewiseConstant.pulse_train(n0, tau_s, period_s, round(n_pulses))
        _CACHE[key] = solve_point(source, chamber, GAS, MODELS[physics]).f_recombination
    return _CACHE[key]


def ionisation_per_mu(q, beam, pulses, physics="full"):
    """Saturation ion pairs per cm^3 per MU from the low-dose-rate slope at 0.5-1 mm, 40 V/mm."""
    low = BEAMS[beam][0]
    n_pulses, period = pulses[(beam, low)]
    d1, d2 = GAPS[0], GAPS[1]
    q1 = q[(beam, low, d1, 40)][0] * k_di(d1)
    q2 = q[(beam, low, d2, 40)][0] * k_di(d2)
    f1 = f2 = 1.0
    for _ in range(4):
        per_cm3 = (q2 / f2 - q1 / f1) * 1e-9 / (ELEMENTARY_CHARGE_C * AREA_CM2 * (d2 - d1) / 10)
        n0 = per_cm3 / n_pulses
        f1 = efficiency(d1, 40, n0, n_pulses, period, physics=physics)
        f2 = efficiency(d2, 40, n0, n_pulses, period, physics=physics)
    return per_cm3 / MU, (1 - f1, 1 - f2)


def boag_u(d_mm, field_V_mm, n0):
    """Boag's pulsed parameter u = alpha n0 d^2 / ((mu+ + mu-) V)."""
    mu = GAS.mu_pos_cm2_Vs + GAS.mu_neg_cm2_Vs
    return GAS.alpha_cm3_s * n0 * (d_mm / 10) / (mu * field_V_mm * 10)


def overlap(d_mm, field_V_mm, period_s):
    """Positive-ion transit time over the pulse period (> 1: consecutive pulses overlap)."""
    return (d_mm / 10) / (GAS.mu_pos_cm2_Vs * field_V_mm * 10) / period_s


# ------------------------------------------------------------------ run
def main():
    t0 = time.perf_counter()
    q = load_charges()
    counts, pulse_rows = load_pulses()
    structures = pulse_structures(counts)
    per_mu, cal_loss = {}, {}
    ion = {k: {} for k in ("classic", "fe", "sc")}
    for beam in BEAMS:
        per_mu[beam], cal_loss[beam] = ionisation_per_mu(q, beam, structures["H2"])
        for k in ion:
            ion[k][beam], _ = ionisation_per_mu(q, beam, structures["H2"], k)
        print(f"{beam}: saturation ionisation {per_mu[beam]:.3e} ion pairs/cm^3 per MU "
              f"(model loss at the calibration gaps {cal_loss[beam][0]:.2%}, {cal_loss[beam][1]:.2%})")  # fmt: skip

    settings = []
    for beam, rates in BEAMS.items():
        for rate in rates:
            s = {
                "beam": beam,
                "rate": rate,
                "key": (beam, rate),
                "pred": {},
                "n0": {},
                "period": {},
            }
            s["meas"] = {(d, E): q[(beam, rate, d, E)][0] for d in GAPS for E in FIELDS}
            s["pol"] = {
                (d, E): q[(beam, rate, d, E)][1] / q[(beam, rate, d, E)][2]
                for d in GAPS
                for E in FIELDS
            }
            # H1, H2: default physics; under H2 also classic, free electrons only, space charge only
            for h, struct, physics, calib in (
                ("H1", structures["H1"], "full", per_mu),
                ("H2", structures["H2"], "full", per_mu),
                ("H2c", structures["H2"], "classic", ion["classic"]),
                ("H2fe", structures["H2"], "fe", ion["fe"]),
                ("H2sc", structures["H2"], "sc", ion["sc"]),
            ):
                n_pulses, period = struct[(beam, rate)]
                n0 = calib[beam] * MU / n_pulses
                s["n0"][h], s["period"][h] = n0, period
                s["pred"][h] = {
                    (d, E, v): efficiency(d, E, n0 * (1 + v * N0_REL), n_pulses, period,
                                          physics=physics)
                    for d in GAPS
                    for E in FIELDS
                    for v in (-1, 0, 1)
                }  # fmt: skip
            settings.append(s)
            print(
                f"  {beam} {rate} MU/min: n0 per pulse H1 {s['n0']['H1']:.2e}, H2 {s['n0']['H2']:.2e} /cm^3"
            )

    # observables
    sig_q = (
        READING_NC / math.sqrt(12) / math.sqrt(2)
    )  # polarity mean of two resolution-limited readings
    for s in settings:
        s["ratio"] = {}
        for d in GAPS:
            for E in FIELDS[1:]:
                m = s["meas"][(d, E)] / s["meas"][(d, 40)]
                err = m * math.hypot(sig_q / s["meas"][(d, E)], sig_q / s["meas"][(d, 40)])
                pred = {
                    h: [s["pred"][h][(d, E, v)] / s["pred"][h][(d, 40, v)] for v in (-1, 0, 1)]
                    for h in s["pred"]
                }  # fmt: skip
                s["ratio"][(d, E)] = {"meas": m, "err": err, "pred": pred}
    dr = []
    for beam, (low, high) in BEAMS.items():
        s_lo = next(s for s in settings if s["key"] == (beam, low))
        s_hi = next(s for s in settings if s["key"] == (beam, high))
        for d in GAPS:
            for E in FIELDS:
                m = s_hi["meas"][(d, E)] / s_lo["meas"][(d, E)]
                err = m * math.hypot(sig_q / s_hi["meas"][(d, E)], sig_q / s_lo["meas"][(d, E)])
                pred = {
                    h: s_hi["pred"][h][(d, E, 0)] / s_lo["pred"][h][(d, E, 0)] for h in s_hi["pred"]
                }
                dr.append({"beam": beam, "d": d, "E": E, "meas": m, "err": err, "pred": pred})

    # sensitivity to the pulse width (H2, nominal ionisation)
    tau_dev = 0.0
    for s in settings:
        n_pulses, period = structures["H2"][s["key"]]
        for d in GAPS:
            for E in FIELDS[1:]:
                nom = s["ratio"][(d, E)]["pred"]["H2"][1]
                for tau in (2e-6, 6e-6):
                    alt = efficiency(d, E, s["n0"]["H2"], n_pulses, period, tau) / efficiency(
                        d, 40, s["n0"]["H2"], n_pulses, period, tau
                    )
                    tau_dev = max(tau_dev, abs(alt - nom))
    print(f"pulse width 2-6 us: max change of a field ratio {tau_dev:.1e}")

    # grid convergence at the two extremes (smallest and largest loss, H2)
    s_hi = next(s for s in settings if s["key"] == ("10 FFF", 2400))
    n_pulses, period = structures["H2"][s_hi["key"]]
    fine = Uniform1D(Grid1D(n_cells=400))
    grid_dev = 0.0
    for d, E in ((0.5, 10), (10.0, 10)):
        ch = ParallelPlateChamber(d / 10, E * d)
        src = PiecewiseConstant.pulse_train(s_hi["n0"]["H2"], TAU_S, period, round(n_pulses))
        f_fine = solve_point(src, ch, GAS, fine).f_recombination
        grid_dev = max(grid_dev, abs((1 - s_hi["pred"]["H2"][(d, E, 0)]) / (1 - f_fine) - 1))
    print(f"grid 200 vs 400 cells: max relative change in 1 - f {grid_dev:.1e}")

    make_figure(settings, dr)
    write_report(settings, dr, structures, counts, pulse_rows, per_mu, cal_loss, tau_dev, grid_dev,
                 time.perf_counter() - t0)  # fmt: skip
    print(f"done in {time.perf_counter() - t0:.0f} s")


def summary(settings, h):
    res = np.array([
        r["meas"] - r["pred"][h][1] for s in settings for r in s["ratio"].values()
        if h != "H1" or s["rate"] == BEAMS[s["beam"]][1]
    ])  # fmt: skip
    return res


def make_figure(settings, dr):
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.2))
    xg = np.array(GAPS)
    for a, s in zip(ax.ravel()[:4], settings, strict=True):
        high = s["rate"] == BEAMS[s["beam"]][1]
        for E in FIELDS[1:]:
            c = FIELD_COLOR[E]
            r = [s["ratio"][(d, E)] for d in GAPS]
            pred = np.array([x["pred"]["H2"] for x in r])
            a.fill_between(xg, pred.min(axis=1), pred.max(axis=1), color=c, alpha=0.18, lw=0)
            a.plot(xg, pred[:, 1], color=c, lw=1.2)
            a.plot(xg, [x["pred"]["H2c"][1] for x in r], color=c, lw=1.0, ls=":")
            if high:
                a.plot(xg, [x["pred"]["H1"][1] for x in r], color=c, lw=1.0, ls=(0, (4, 3)))
            a.errorbar(xg, [x["meas"] for x in r], yerr=[2 * x["err"] for x in r], fmt="o", color=c,
                       ms=5, capsize=2, label=f"{E} V/mm, measured")  # fmt: skip
        a.plot([], [], color=INK2, lw=1.2, label="v4 prediction" + (", H2" if high else ""))
        a.fill_between([], [], [], color=INK2, alpha=0.18, lw=0, label=f"ionisation ± {N0_REL:.0%}")
        a.plot([], [], color=INK2, lw=1.0, ls=":", label="classic model (H2)")
        if high:
            a.plot(
                [],
                [],
                color=INK2,
                lw=1.0,
                ls=(0, (4, 3)),
                label="v4, H1 (pulse counts as recorded)",
            )
        a.set(xscale="log", xlabel="electrode gap d (mm)", ylabel="Q(d, E) / Q(d, 40 V/mm)",
              title=f"({'abcd'[settings.index(s)]}) {BEAM_NAME[s['beam']]}, {s['rate']} MU/min")  # fmt: skip
        a.set_xticks(GAPS, [f"{d:g}" for d in GAPS])
        a.minorticks_off()
        a.set_ylim(0.45, 1.02)
        a.legend(loc="lower left")

    # (e) dose-rate ratio: measured vs predicted for both readings of the pulse records
    a = ax[1, 1]
    lo = min(min(r["pred"]["H1"] for r in dr), min(r["meas"] for r in dr)) - 0.02
    a.plot([lo, 1.02], [lo, 1.02], color=AXIS, lw=0.8)
    for r in dr:
        m = "o" if r["beam"] == "16 MeV" else "s"
        c = FIELD_COLOR[r["E"]]
        a.plot(r["pred"]["H1"], r["meas"], m, mfc=SURFACE, mec=c, mew=1.3, ms=5.5)
        a.plot(r["pred"]["H2"], r["meas"], m, color=c, ms=5.5)
    handles = [
        Line2D([], [], color=FIELD_COLOR[E], marker="o", ls="", label=f"{E} V/mm") for E in FIELDS
    ]
    handles += [
        Line2D([], [], color=INK2, marker="o", ls="", label="16 MeV, 1000 / 100 MU/min"),
        Line2D([], [], color=INK2, marker="s", ls="", label="10 FFF, 2400 / 400 MU/min"),
        Line2D([], [], color=INK2, marker="o", ls="", label="filled: H2 (constant dose per pulse)"),
        Line2D([], [], color=INK2, marker="o", ls="", mfc=SURFACE, label="hollow: H1 (counts as recorded)"),
    ]  # fmt: skip
    a.legend(handles=handles, loc="lower right")
    a.set(xlabel="predicted Q_high / Q_low (same gap and voltage)", ylabel="measured Q_high / Q_low",
          title="(e) Dose-rate ratio at 50 MU: which pulse structure?", xlim=(lo, 1.02), ylim=(lo, 1.02))  # fmt: skip

    # (f) residuals of the field ratios (H2), with the ionisation band behind each point
    a = ax[1, 2]
    a.axhline(0, color=AXIS, lw=0.8)
    for s in settings:
        dx = 1 + 0.035 * (list(SETTING_MARKER).index(s["key"]) - 1.5)
        period = s["period"]["H2"]
        for E in FIELDS[1:]:
            c = FIELD_COLOR[E]
            for d in GAPS:
                r = s["ratio"][(d, E)]
                p = r["pred"]["H2"]
                x = d * dx * (1.06 if E == 10 else 0.94)
                a.vlines(x, (min(p) - p[1]) * 100, (max(p) - p[1]) * 100, color=c, alpha=0.3, lw=4)
                ov = overlap(d, E, period) > 1
                a.errorbar(x, (r["meas"] - p[1]) * 100, yerr=2 * r["err"] * 100, fmt=SETTING_MARKER[s["key"]],
                           color=c, mfc=SURFACE if ov else c, mec=c, mew=1.2, ms=5, capsize=0, lw=1.0)  # fmt: skip
    handles = [
        Line2D([], [], color=FIELD_COLOR[E], lw=4, alpha=0.5, label=f"{E} V/mm") for E in FIELDS[1:]
    ]
    handles += [
        Line2D([], [], color=INK2, marker=m, ls="", ms=5, label=f"{k[0]}, {k[1]} MU/min")
        for k, m in SETTING_MARKER.items()
    ]
    handles.append(
        Line2D(
            [], [], color=INK2, marker="o", ls="", mfc=SURFACE, ms=5, label="hollow: pulses overlap"
        )
    )
    lo_, hi_ = a.get_ylim()
    a.set_ylim(lo_ - 0.9 * (hi_ - lo_), hi_)
    a.legend(handles=handles, loc="lower left", ncol=2)
    a.set(xscale="log", xlabel="electrode gap d (mm)", ylabel="measured − predicted field ratio (%)",
          title="(f) Residuals (H2): bars = 2σ reading, shade = ionisation ± 10 %")  # fmt: skip
    a.set_xticks(GAPS, [f"{d:g}" for d in GAPS])
    a.minorticks_off()
    fig.suptitle("Figure 7 · Blind benchmark: pulsed linac beams, extrapolation chamber 0.5–10 mm (Risø 2026)",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig7_risoe2026_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(
    settings, dr, structures, counts, pulse_rows, per_mu, cal_loss, tau_dev, grid_dev, runtime
):
    air_gy = GAS.ion_pairs_per_cm3_per_Gy()

    def rng(p):
        return f"{p[1]:.3f} ({min(p):.3f}–{max(p):.3f})"

    setting_rows = "\n".join(
        f"| {BEAM_NAME[s['beam']]} | {s['rate']} | {counts[s['key']][0]:.0f} | {1 / counts[s['key']][1]:.0f} "
        f"| {structures['H2'][s['key']][0]:.0f} | {1 / s['period']['H2']:.0f} "
        f"| {s['n0']['H1'] / air_gy * 1e3:.2f} | {s['n0']['H2'] / air_gy * 1e3:.2f} |"
        for s in settings
    )
    ratio_rows = []
    for s in settings:
        high = s["rate"] == BEAMS[s["beam"]][1]
        for d in GAPS:
            r20, r10 = s["ratio"][(d, 20)], s["ratio"][(d, 10)]
            h1 = f"{r10['pred']['H1'][1]:.3f}" if high else "="
            flag = " ¹" if abs(s["pol"][(d, 20)] - 1) > 0.02 and d >= 5 else ""
            ratio_rows.append(
                f"| {s['beam']}, {s['rate']} | {d:g} | {overlap(d, 10, s['period']['H2']):.2f} "
                f"| {boag_u(d, 10, s['n0']['H2']):.2f} | {r20['meas']:.3f}{flag} | {rng(r20['pred']['H2'])} "
                f"| {r10['meas']:.3f} | {rng(r10['pred']['H2'])} | {h1} |"
            )
    dr_rows = "\n".join(
        f"| {r['beam']} | {r['d']:g} | {r['E']} | {r['meas']:.3f} | {r['pred']['H2']:.3f} | {r['pred']['H1']:.3f} |"
        for r in dr
    )
    res2 = summary(settings, "H2")
    res1 = summary(settings, "H1")
    res2c = summary(settings, "H2c")

    def region(h, keep):
        return np.array([
            s["ratio"][(d, E)]["meas"] - s["ratio"][(d, E)]["pred"][h][1]
            for s in settings for d in GAPS if keep(s, d) for E in FIELDS[1:]
        ])  # fmt: skip

    variants = ("H2", "H2fe", "H2sc", "H2c")
    decomp_rows = []
    for label, keep in (
        ("all 48 ratios, RMS", lambda s, d: True),
        ("gaps ≤ 2 mm, mean", lambda s, d: d <= 2),
        ("gaps 5–10 mm, mean", lambda s, d: d >= 5),
        ("overlap at 10 V/mm (t_c > T), mean", lambda s, d: overlap(d, 10, s["period"]["H2"]) > 1),
    ):
        vals = [region(h, keep) for h in variants]
        if label.endswith("RMS"):
            cells = [f"{np.sqrt(np.mean(v**2)) * 100:.2f} %" for v in vals]
        else:
            cells = [f"{np.mean(v) * 100:+.2f} %" for v in vals]
        decomp_rows.append(f"   | {label} | " + " | ".join(cells) + " |")
    decomp = chr(10).join(decomp_rows)
    captured_v = {}
    within = sum(
        int(min(r["pred"]["H2"]) - 2 * r["err"] <= r["meas"] <= max(r["pred"]["H2"]) + 2 * r["err"])
        for s in settings
        for r in s["ratio"].values()
    )
    n_ratio = sum(len(s["ratio"]) for s in settings)
    dr1 = np.array([r["meas"] - r["pred"]["H1"] for r in dr])
    high_period = {b: structures["H2"][(b, rates[1])][1] for b, rates in BEAMS.items()}
    dr_ov = [overlap(r["d"], r["E"], high_period[r["beam"]]) > 1 for r in dr]
    dr2_iso = [r["meas"] - r["pred"]["H2"] for r, o in zip(dr, dr_ov, strict=True) if not o]
    worst_iso = max(
        (r for r, o in zip(dr, dr_ov, strict=True) if not o),
        key=lambda r: abs(r["meas"] - r["pred"]["H2"]),
    )
    dr_ffF10 = next(r for r in dr if r["beam"] == "10 FFF" and r["d"] == 10.0 and r["E"] == 10)
    captured = (1 - dr_ffF10["pred"]["H2"]) / (1 - dr_ffF10["meas"])
    captured_c = (1 - dr_ffF10["pred"]["H2c"]) / (1 - dr_ffF10["meas"])
    for h in ("H2", "H2fe", "H2sc", "H2c"):
        captured_v[h] = (1 - dr_ffF10["pred"][h]) / (1 - dr_ffF10["meas"])
    dr40 = [r["meas"] for r in dr if r["E"] == 40]
    dr40_h1 = [r["pred"]["H1"] for r in dr if r["E"] == 40]
    u_max = max(boag_u(d, 10, s["n0"]["H2"]) for s in settings for d in GAPS)
    six = {k: pulse_rows[k] for k in ("10406", "10408", "10409", "10412")}
    (ROOT / "validation" / "BENCHMARK_RISOE2026.md").write_text(
        f"""# Benchmark: pulsed linac beams, extrapolation chamber (Risø 2026)

Generated by `python validation/benchmark_risoe2026.py` ({runtime:.0f} s). The data are from a
Varian linac at Risø (31.03.2026), copied from the IonTracks-AMM repository into
`validation/data/risoe2026/` (see its README):
- an extrapolation chamber (collecting electrode r = 15 mm) 71.4 cm from the source in a
  20 × 20 cm² field;
- 16 MeV electrons at 100 and 1000 MU/min, and 10 MV FFF photons at 400 and 2400 MU/min;
- gaps of 0.5–10 mm at 10, 20 and 40 V/mm, 50 MU per reading, both polarities.

Recombination here is large (up to 45 % at 10 mm and 10 V/mm). At the high dose rates and large
gaps, the ions of one pulse are still in the gap when the next pulse arrives. This makes it a
direct test of v4's handling of pulse overlap.

**This is a blind prediction. No model input is adjusted to the recombination data.**

| Input | Value | Source |
|---|---|---|
| Charge per reading | mean of the two polarities | Dataset (raw charges) |
| Ionisation per MU | {per_mu["16 MeV"]:.3e} (16 MeV), {per_mu["10 FFF"]:.3e} (10 FFF) ion pairs/cm³ | Slope of Q(d) at 0.5–1 mm, 40 V/mm, **low dose rate only**, divided by e × area; the model's own loss there ({cal_loss["16 MeV"][0]:.1%}–{cal_loss["10 FFF"][1]:.1%}) is removed |
| Ionisation uncertainty | ± {N0_REL:.0%} | Sensitivity band, not a derived uncertainty |
| Pulses per 50 MU, period | see below | `pulses_per_MU.csv` (09.04.2026, a different day) |
| Pulse width | 4 µs | Assumed; 2 and 6 µs change any field ratio by ≤ {tau_dev:.1e} |
| Gas | 20 °C (assumed), 1014.38 hPa | `Gas.air`: µ⁺ 1.36, µ⁻ 2.1 cm²/Vs, α 1.6e-6 cm³/s |
| Collecting area | π × (15 mm)² | Implied by the dataset's `electrons_s_cm3` column |

The two observables need neither the gap offset nor an inverse-square correction:

1. **The field ratio Q(d, E) / Q(d, 40 V/mm)** for E = 20 and 10 V/mm, at fixed beam, dose rate and
   gap ({n_ratio} values).
2. **The dose-rate ratio Q_high / Q_low** at the same gap and voltage ({len(dr)} values). Both
   readings are 50 MU, so this tests the pulse structure.

![Figure 7](figures/fig7_risoe2026_benchmark.png)

## The pulse records: two readings

The high-dose-rate pulse counts are hard to believe as recorded. In the same file:
- **6 MV:** 36 pulses/MU from 5 to 200 MU/min (3 → 120 Hz); then about a third of that
  (11.5–11.7 pulses/MU) from 300 to 600 MU/min, where constant dose per pulse would need
  180–360 Hz (files {", ".join(six)}).
- **Other beams:** no counted rate in any beam exceeds ~122 Hz. The 4 MV and 10 FFF series show
  the same drop at the dose rate where the implied rate passes that ceiling.

The counter therefore appears to miss pulses above ~120 Hz. For the 16 MeV electrons the count
already drops at 600 MU/min (implied 99 Hz), so the file alone does not settle that beam.

- **H1** takes the counts as recorded. The high dose rates then have 2.3× (16 MeV) and 2.5× (10 FFF)
  the dose per pulse.
- **H2** keeps the low-dose-rate dose per pulse and scales the repetition rate with the dose rate.

H2 was set up from the pulse file before comparing with the charges, and both are reported.

Independent support for H2 comes from [White et al. 2026](https://doi.org/10.1002/acm2.70670),
who counted TrueBeam pulses with a plastic scintillator (500 µs sampling):
- 6 MV: 35.5 pulses per MU at 400 MU/min, against 11.5 in this file at the same setting;
- 10 MV FFF: 7.9 pulses per MU at 400 MU/min, against 7.4 in this file;
- a maximum repetition rate of 360 Hz and pulses of about 3 µs.

At a constant dose per pulse, 2400 MU/min then needs about 300 Hz, which the machine can deliver.
The pulse file's 116 Hz is therefore an undercount.

| Beam | MU/min | recorded pulses per 50 MU | recorded rate (Hz) | H2 pulses per 50 MU | H2 rate (Hz) | dose per pulse in air, H1 (mGy) | H2 (mGy) |
|---|---|---|---|---|---|---|---|
{setting_rows}

## Results

1. **The dose-rate ratio decides between H1 and H2.** Under H2, the {len(dr2_iso)} ratios without
   pulse overlap are predicted to within {np.max(np.abs(dr2_iso)) * 100:.1f} %. The largest of
   these is {worst_iso["beam"]}, {worst_iso["d"]:g} mm, {worst_iso["E"]} V/mm, which involves the
   flagged polarity pair (¹ below); all others are within
   {sorted(abs(x) for x in dr2_iso)[-2] * 100:.1f} %. The {sum(dr_ov)} overlap cases are item 4.
   Under H1 the deviations reach {np.max(np.abs(dr1)) * 100:.0f} %. At 40 V/mm the measured Q_high / Q_low is
   {min(dr40):.3f}–{max(dr40):.3f} at every gap. H1 predicts it would fall to {min(dr40_h1):.3f} at
   10 mm, because the dose per pulse would have changed. The recorded high-rate pulse counts are
   therefore wrong, and H2 is used below.
2. **Field ratios (H2).** All {n_ratio} ratios are reproduced to within
   {np.max(np.abs(res2)) * 100:.1f} % (RMS {np.sqrt(np.mean(res2**2)) * 100:.1f} %), with losses
   from 1 % to 45 %. {within} of {n_ratio} lie inside the ionisation band widened by twice the
   reading resolution. The classic model (no free electrons, no space charge) reaches
   {np.max(np.abs(res2c)) * 100:.1f} % (RMS {np.sqrt(np.mean(res2c**2)) * 100:.1f} %). Under H1 the
   high-dose-rate ratios deviate by up to {np.max(np.abs(res1)) * 100:.0f} %.
3. **Which physics does what.** Residual = measured − predicted field ratio (positive: less loss
   was measured than predicted), for the default model and with each effect switched off:

   | Ratios | default (free e⁻ + space charge) | free electrons only | space charge only | classic |
   |---|---|---|---|---|
{decomp}
   | overlap: share of the measured extra loss at 10 mm, 10 V/mm (10 FFF) | {captured_v["H2"]:.0%} | {captured_v["H2fe"]:.0%} | {captured_v["H2sc"]:.0%} | {captured_v["H2c"]:.0%} |

   - **Free electrons hardly matter here.** At 100–400 V/cm the attachment length is only
     10–40 µm, so few electrons escape. Their columns are close to the classic ones.
   - **Space charge decides both the good and the bad.** Boag's u reaches {u_max:.1f} and the
     space-charge parameter is ≈ 3.9 u, so space charge adds loss wherever the ion density is
     high relative to the field.
     - Good: it reproduces the extra loss from overlapping pulses at fixed dose per pulse (the
       last row).
     - Bad: it also adds about 2 % of loss to the field ratios at 5–10 mm, where the classic
       model agrees better.

     The field model is laterally uniform, 1D Gauss. The ion parameters (Boag's µ± and α) set
     the space-charge strength through u. So this benchmark cannot separate a too-strong
     space-charge model from unsuitable ion parameters.
4. **Pulse overlap.** Under H2 the 10 FFF 2400 MU/min pulses come
   every {structures["H2"][("10 FFF", 2400)][1] * 1e3:.1f} ms, against a positive-ion transit time
   of 7.4 ms at 10 mm and 10 V/mm. The dose per pulse is the same at both dose rates. The field
   ratio there is:
   - predicted: {next(s for s in settings if s["key"] == ("10 FFF", 400))["ratio"][(10.0, 10)]["pred"]["H2"][1]:.3f}
     at 400 MU/min and {next(s for s in settings if s["key"] == ("10 FFF", 2400))["ratio"][(10.0, 10)]["pred"]["H2"][1]:.3f}
     at 2400 MU/min;
   - measured: {next(s for s in settings if s["key"] == ("10 FFF", 400))["ratio"][(10.0, 10)]["meas"]:.3f}
     and {next(s for s in settings if s["key"] == ("10 FFF", 2400))["ratio"][(10.0, 10)]["meas"]:.3f}.

   The default model gives {captured:.0%} of the measured extra loss (dose-rate ratio
   {dr_ffF10["pred"]["H2"]:.3f} against {dr_ffF10["meas"]:.3f}), and the classic model
   {captured_c:.0%}. Treating each pulse in isolation, as the AMM scripts do, predicts none of it.
   The H2 repetition rate is inferred (six times the 400 MU/min count), not measured. A pulse
   count that resolves > 120 Hz would test it directly.

## Validity

- **Free electrons.** The fields here are 100–400 V/cm. The Götz attachment parameterisation is
  stated for ≥ 1000 V/cm (with one value at 500 V/cm), so it is extrapolated at every point of
  this benchmark. Boag's α and mobilities may also partly contain free-electron effects already
  (see the DCPT benchmark), so the free-electron part is the least certain input.
- **Space charge.** For pulses the field of the separated charge relative to the applied field is
  e n₀ d / (ε₀ E) ≈ 3.9 u in air, which exceeds 1 at most 5–10 mm points. The model solves
  Gauss's law for the laterally uniform ion charge.
- **Numerics.** Going from 200 to 400 cells changes 1 − f by ≤ {grid_dev:.1e} (relative) at the
  smallest and largest loss. The pulse width does not matter: pulses of 2–6 µs are much shorter
  than the transit times (≥ 90 µs).

## Uncertainties

- **Measurement.** Each charge is a single reading per polarity, recorded to 0.05 nC. The error
  bars use that resolution only (0.014 nC per reading). There are no repeats, so the
  reproducibility is unknown and not included.
- **Ionisation.** It comes from the low-dose-rate charges at the two smallest gaps. It sets the
  loss scale (roughly ∝ n₀), which the field ratios cannot constrain. The ± 10 % band shows
  that sensitivity.
- **Not propagated.** The pulse records are from another day (09.04.2026). The temperature is
  assumed, which affects α/µ by ~0.3 %/K. The area is taken from the dataset's own column.

## Per gap: field ratios

Values in brackets span the ionisation ± {N0_REL:.0%}. t_c/T is the positive-ion transit time at
10 V/mm over the pulse period (> 1: pulses overlap). u is Boag's parameter at 10 V/mm (H2).

| Setting | d (mm) | t_c/T | u | measured 20/40 | predicted 20/40 | measured 10/40 | predicted 10/40 | H1 10/40 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(ratio_rows)}

¹ This reading's polarity pair (Q⁺/Q⁻ = {q_note(settings)}) is far more asymmetric than any other reading at gaps ≥ 5 mm.
It is kept in the comparison.

## Dose-rate ratios

| Beam | d (mm) | E (V/mm) | measured | H2 | H1 |
|---|---|---|---|---|---|
{dr_rows}

## Notes on the source data

- `k_di` in the dataset is an inverse-square correction. It is 1 + d/890 mm on `main` of
  IonTracks-AMM, and 1 + d/714 mm after the "uniformity correction fix" on
  `test-structued-pulses`, which matches the stated 71.4 cm. It cancels in both observables and is
  used only in the ionisation calibration (≤ 0.14 %).
- The AMM scripts divide by 496 (16 MeV) and 371 (10 FFF) pulses at both dose rates, which amounts
  to H2. They then treat each pulse in isolation, so pulse overlap is missed.
- The dataset's `k_s` (AMM `fit_summary.csv`) comes from a linear fit of Q(d) at 0.5–1 mm and is
  below 1 in places. It is not used here.
- **Would tighten the benchmark.** A pulse count at the high dose rates with a detector that
  resolves > 120 Hz, the temperature, repeat readings, and an independent dose per MU at the
  chamber position.
""",
        encoding="utf-8",
    )


def q_note(settings):
    s = next(s for s in settings if s["key"] == ("16 MeV", 100))
    return f"{s['pol'][(10.0, 20)]:.3f}"


if __name__ == "__main__":
    main()
