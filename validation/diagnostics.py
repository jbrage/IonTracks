"""Diagnostic figures and animations for the two benchmark experiments.

DCPT 2019 (scanned protons, Roos chamber 2 mm), setting 244 MeV, DRF 1:
  diag_dcpt2019.png        time structure at the chamber centre, chamber current, cumulative
                           collection and recombination, k_s map, gap profiles during one spot
  diag_dcpt2019_gap.gif    ions in the gap at the chamber centre while spots pass (50 V)
  diag_dcpt2019_track.gif  one 244 MeV proton track in the 3D track model (200 V)

Risø 2026 (pulsed 10 MV FFF, extrapolation chamber), 10 mm gap, 10 V/mm, 400 and 2400 MU/min:
  diag_risoe2026.png       pulse structure, current, cumulative collection and recombination,
                           per-pulse efficiency, field distortion
  diag_risoe2026_gap.gif   ions and field in the gap: isolated pulses (400 MU/min) against
                           overlapping pulses (2400 MU/min)

All runs use the default model (free electrons and space charge). Writes the files to
validation/figures/ and a caption page validation/DIAGNOSTICS.md.
Run from the repository root:  python validation/diagnostics.py   (~3 min)
"""

import time
from pathlib import Path

import benchmark_dcpt2019 as dcpt
import benchmark_risoe2026 as risoe
import matplotlib.pyplot as plt
import numpy as np
import plotstyle
from matplotlib.animation import FuncAnimation, PillowWriter
from plotstyle import INK2, LIMIT, MUTED, S1, S2, S3, SEQUENTIAL

from iontracks import (
    GaussianTrack,
    Grid1D,
    Grid3D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Tile3D,
    Uniform1D,
    solve_field,
)
from iontracks.constants import ELEMENTARY_CHARGE_C

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
MODEL = Uniform1D(Grid1D(n_cells=150))
FPS = 12
plotstyle.apply()


def cumulative_fractions(ep, t):
    """Produced, collected (induced) and recombined charge up to t, as fractions of the total."""
    total = ep.produced_cm2
    return (
        ep.cumulative("source", t) / total,
        ep.cumulative("induced", t) / total,
        ep.cumulative("recombined", t) / total,
    )


def save_gif(anim, name):
    anim.save(FIG / name, writer=PillowWriter(fps=FPS), dpi=80)


# ================================================================== DCPT 2019
def run_dcpt():
    print("DCPT 2019: 244 MeV, DRF 1")
    s = next(x for x in dcpt.load() if x["E"] == 244 and x["drf"] == 1.0)
    gas = dcpt.gas_for(s)
    sigma = dcpt.sigma_literature_cm(244)
    field = dcpt.raster(s, sigma)
    per_gy = gas.ion_pairs_per_cm3_per_Gy()
    src0 = field.local_source(0.0, 0.0, gas)
    # the spot that passes closest to the centre, and a window of a few spots around it
    peak = int(np.argmax(src0.rates))
    t_peak = src0.edges[peak]
    dwell = float(s["row"]["time_s"])
    t_lo, t_hi = t_peak - 3 * (dwell + dcpt.DEAD_S), t_peak + 4 * (dwell + dcpt.DEAD_S)
    window = src0.window(max(t_lo, src0.t_start_s), t_hi).shifted(-t_lo)

    # whole layer at the chamber level (50 V and 200 V)
    res = {}
    for volts in (50.0, 200.0):
        ch = ParallelPlateChamber(dcpt.GAP_CM, volts, sensitive_radius_cm=dcpt.ROOS_R_CM)
        t0 = time.perf_counter()
        res[volts] = solve_field(field, ch, gas, MODEL, points=dcpt.POINTS)
        print(
            f"  layer at {volts:g} V: k_s = {res[volts].ks:.5f} ({time.perf_counter() - t0:.0f} s)"
        )

    # the centre point over the window, with gap snapshots (50 V)
    ch50 = ParallelPlateChamber(dcpt.GAP_CM, 50.0)
    frames = np.linspace(0, window.t_end_s + 1.5e-3, 90)
    ep = MODEL.run_episode(window, ch50, gas, snapshot_times_s=frames)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    # (a) local dose rate at the centre over the whole layer
    a = ax[0, 0]
    t_line = np.linspace(src0.t_start_s, src0.t_end_s, 20000)
    a.plot(t_line, src0.rate(t_line) / per_gy, color=S1, lw=0.6)
    a.axvspan(t_lo, t_hi, color=S2, alpha=0.2, lw=0, label="window of the gap animation")
    a.set(xlabel="time in the layer (s)", ylabel="local dose rate in air (Gy/s)",
          title="(a) Time structure at the chamber centre")  # fmt: skip
    a.legend(loc="upper right")
    # (b) zoom: the spots passing the centre
    a = ax[0, 1]
    t_zoom = np.linspace(0, window.t_end_s, 4000)
    a.fill_between(t_zoom * 1e3, window.rate(t_zoom) / per_gy, color=S1, alpha=0.3, lw=0)
    a.plot(
        t_zoom * 1e3,
        window.rate(t_zoom) / per_gy,
        color=S1,
        lw=1.0,
        label="ionisation (as dose rate)",
    )
    cur = np.diff(ep.cumulative("induced", t_zoom)) / np.diff(t_zoom) / per_gy
    a.plot(0.5 * (t_zoom[1:] + t_zoom[:-1]) * 1e3, cur, color=S2, lw=1.2,
           label="collected (induced current, same units)")  # fmt: skip
    a.set(xlabel="time (ms)", ylabel="Gy/s in air", title="(b) Spots passing the centre, 50 V")
    a.legend(loc="upper right")
    # (c) chamber current over the layer
    a = ax[0, 2]
    edges = np.linspace(field.t_start_s, field.t_end_s + 5e-3, 3001)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for volts, color in ((50.0, S2), (200.0, S1)):
        a.plot(mid, res[volts].current_A(edges) * 1e9, color=color, lw=0.6, label=f"{volts:g} V")
    a.set(xlabel="time in the layer (s)", ylabel="chamber current (nA, 2.6 ms bins)",
          title="(c) Chamber current")  # fmt: skip
    a.legend(loc="upper right")
    # (d) cumulative collection and recombination at the chamber level
    a = ax[1, 0]
    for volts, color in ((50.0, S2), (200.0, S1)):
        r = res[volts]
        prod = np.zeros(edges.size)
        coll = np.zeros(edges.size)
        reco = np.zeros(edges.size)
        for p in r.points:
            for g, e in zip(p.plan.groups, p.episodes, strict=True):
                for start in g.starts_s:
                    prod += p.weight_cm2 * e.cumulative("source", edges - start)
                    coll += p.weight_cm2 * e.cumulative("induced", edges - start)
                    reco += p.weight_cm2 * e.cumulative("recombined", edges - start)
        a.plot(edges, coll / prod[-1], color=color, lw=1.2, label=f"collected, {volts:g} V")
        a.plot(edges, reco / prod[-1] * 20, color=color, lw=1.0, ls="--",
               label=f"recombined × 20, {volts:g} V")  # fmt: skip
    a.plot(edges, prod / prod[-1], **LIMIT, label="produced")
    a.set(xlabel="time in the layer (s)", ylabel="fraction of all ion pairs",
          title="(d) Cumulative charge, whole chamber")  # fmt: skip
    a.legend(loc="upper left")
    # (e) k_s map at 50 V
    a = ax[1, 1]
    x, y, ks = res[50.0].ks_map()
    ppm = ((ks - 1) / np.mean(ks - 1) - 1) * 1e6
    sc = a.scatter(x * 10, y * 10, c=ppm, cmap=SEQUENTIAL, s=90)
    fig.colorbar(sc, ax=a, label="(k_s − 1) / mean − 1 (ppm)")
    a.set(
        aspect="equal", xlabel="x (mm)", ylabel="y (mm)", title="(e) k_s over the Roos area, 50 V"
    )
    # (f) gap profiles during the closest spot
    a = ax[1, 1 + 1]
    snap = ep.snapshots
    z = snap.z_cm * 10
    t_on = t_peak - t_lo
    pick = np.searchsorted(
        snap.times_s, [t_on + 0.15e-3, t_on + 0.8 * dwell, t_on + dwell + 0.25e-3]
    )
    for k, color in zip(pick, (S1, S2, S3), strict=True):
        a.plot(
            z,
            snap.n_pos_cm3[k] / 1e9,
            color=color,
            lw=1.4,
            label=f"n⁺, t = {snap.times_s[k] * 1e3:.1f} ms",
        )
        a.plot(
            z,
            snap.n_neg_cm3[k] / 1e9,
            color=color,
            lw=1.0,
            ls="--",
            label=f"n⁻, t = {snap.times_s[k] * 1e3:.1f} ms",
        )
    a.set(xlabel="z (mm), anode at 0", ylabel="ion density (10⁹ cm⁻³)",
          title="(f) Gap profiles at the centre, 50 V")  # fmt: skip
    a.legend(loc="upper center", fontsize=7, ncol=2)
    fig.suptitle(
        "DCPT 2019 · 244 MeV, DRF 1 · Roos chamber 2 mm · default model", x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(FIG / "diag_dcpt2019.png")
    plt.close(fig)

    # --- animation: ions in the gap at the centre while spots pass
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a0, a1, a2 = ax
    a0.fill_between(t_zoom * 1e3, window.rate(t_zoom) / per_gy, color=S1, alpha=0.3, lw=0)
    a0.plot(t_zoom * 1e3, window.rate(t_zoom) / per_gy, color=S1, lw=1.0)
    cursor = a0.axvline(0, color=INK2, lw=1.2)
    a0.set(xlabel="time (ms)", ylabel="local dose rate in air (Gy/s)", title="Spots at the centre")
    n_max = max(snap.n_pos_cm3.max(), snap.n_neg_cm3.max()) / 1e9 * 1.1
    (lp,) = a1.plot(z, snap.n_pos_cm3[0] / 1e9, color=S1, lw=1.6, label="positive ions → cathode")
    (ln,) = a1.plot(z, snap.n_neg_cm3[0] / 1e9, color=S2, lw=1.6, label="negative ions → anode")
    a1.set(xlim=(0, z[-1] + z[0]), ylim=(0, n_max), xlabel="z (mm), anode at 0",
           ylabel="ion density (10⁹ cm⁻³)", title="Ions in the 2 mm gap, 50 V")  # fmt: skip
    a1.legend(loc="upper center", fontsize=8)
    t_cum = snap.times_s
    prod, coll, reco = cumulative_fractions(ep, t_cum)
    a2.plot(t_cum * 1e3, prod, **LIMIT, label="produced")
    a2.plot(t_cum * 1e3, coll, color=S1, lw=1.4, label="collected")
    a2.plot(t_cum * 1e3, reco * 10, color=S2, lw=1.4, label="recombined × 10")
    dot = a2.axvline(0, color=INK2, lw=1.2)
    a2.set(xlabel="time (ms)", ylabel="fraction of the window's ion pairs",
           title="Cumulative charge at the centre")  # fmt: skip
    a2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    def update(k):
        t_ms = snap.times_s[k] * 1e3
        cursor.set_xdata([t_ms, t_ms])
        dot.set_xdata([t_ms, t_ms])
        lp.set_ydata(snap.n_pos_cm3[k] / 1e9)
        ln.set_ydata(snap.n_neg_cm3[k] / 1e9)
        return cursor, dot, lp, ln

    save_gif(
        FuncAnimation(fig, update, frames=len(snap.times_s), blit=False), "diag_dcpt2019_gap.gif"
    )
    plt.close(fig)

    # --- animation: one proton track in the 3D model
    ch200 = ParallelPlateChamber(dcpt.GAP_CM, 200.0)
    track = GaussianTrack.from_stopping_power(dcpt.stopping_power(244, "air"), gas)
    model = Tile3D(track, Grid3D(n_xy=48, n_z=100))
    t_c = max(ch200.collection_times_s(gas))
    ep3 = model.single_track(ch200, gas, snapshot_times_s=np.linspace(0, 1.05 * t_c, 70))
    sn = ep3.snapshots
    x_um = (sn.x_cm - sn.x_cm.mean()) * 1e4
    vmax = np.log10(max(sn.n_pos_xz_cm3.max(), 1.0))
    fig, ax = plt.subplots(
        1, 5, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.06, 0.3, 1.3]}
    )
    ax[3].axis("off")
    ims = []
    for a, dens, name in ((ax[0], sn.n_pos_xz_cm3, "positive ions"), (ax[1], sn.n_neg_xz_cm3, "negative ions")):  # fmt: skip
        im = a.imshow(np.log10(np.maximum(dens[0].T, 1e-3)), origin="lower", aspect="auto",
                      cmap=SEQUENTIAL, vmin=vmax - 4, vmax=vmax,
                      extent=(x_um[0], x_um[-1], 0, dcpt.GAP_CM * 10))  # fmt: skip
        a.set(xlabel="x (µm)", ylabel="z (mm), anode at 0", title=name)
        a.grid(False)
        ims.append(im)
    fig.colorbar(ims[1], cax=ax[2], label="log₁₀ density (cm⁻³)")
    zc = sn.z_cm * 10
    (mp,) = ax[4].plot(
        zc, sn.n_pos_mean_cm3[0] / 1e5, color=S1, lw=1.5, label="positive (tile mean)"
    )
    (mn,) = ax[4].plot(
        zc, sn.n_neg_mean_cm3[0] / 1e5, color=S2, lw=1.5, label="negative (tile mean)"
    )
    ax[4].set(xlabel="z (mm)", ylabel="tile-mean density (10⁵ cm⁻³)",
              ylim=(0, 1.1 * sn.n_pos_mean_cm3.max() / 1e5), title="Column sliding apart")  # fmt: skip
    ax[4].legend(loc="lower right", fontsize=8)
    title = fig.suptitle("", x=0.01, ha="left")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.05, right=0.98, wspace=0.35)

    def update3(k):
        ims[0].set_data(np.log10(np.maximum(sn.n_pos_xz_cm3[k].T, 1e-3)))
        ims[1].set_data(np.log10(np.maximum(sn.n_neg_xz_cm3[k].T, 1e-3)))
        mp.set_ydata(sn.n_pos_mean_cm3[k] / 1e5)
        mn.set_ydata(sn.n_neg_mean_cm3[k] / 1e5)
        title.set_text(f"One 244 MeV proton track, 2 mm, 200 V · t = {sn.times_s[k] * 1e6:.0f} µs "
                       f"({sn.times_s[k] / t_c:.2f} t_c) · k_s − 1 of the track = {ep3.ks - 1:.1e}")  # fmt: skip
        return (*ims, mp, mn, title)

    save_gif(
        FuncAnimation(fig, update3, frames=len(sn.times_s), blit=False), "diag_dcpt2019_track.gif"
    )
    plt.close(fig)
    return {"ks50": res[50.0].ks, "ks200": res[200.0].ks, "ks_track": ep3.ks, "t_peak": t_peak}


# ================================================================== Risø 2026
def run_risoe():
    print("Risø 2026: 10 MV FFF, 10 mm, 10 V/mm")
    q = risoe.load_charges()
    counts, _ = risoe.load_pulses()
    structures = risoe.pulse_structures(counts)
    per_mu, _ = risoe.ionisation_per_mu(q, "10 FFF", structures["H2"])
    gas = risoe.GAS
    d_mm, e_mm = 10.0, 10
    ch = ParallelPlateChamber(d_mm / 10, e_mm * d_mm)
    area = risoe.AREA_CM2
    runs = {}
    t_end = 0.1  # s: the first 100 ms of the delivery
    for rate, color in ((400, S1), (2400, S2)):
        n_pulses, period = structures["H2"][("10 FFF", rate)]
        n0 = per_mu * risoe.MU / n_pulses
        n_in = int(t_end / period)
        src = PiecewiseConstant.pulse_train(n0, risoe.TAU_S, period, n_in)
        frames = np.linspace(0, 20e-3, 100)
        ep = MODEL.run_episode(src, ch, gas, snapshot_times_s=frames)
        runs[rate] = {"src": src, "ep": ep, "period": period, "n0": n0, "color": color}
        print(f"  {rate} MU/min: period {period * 1e3:.2f} ms, f = {ep.f_recombination:.4f}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
    per_gy = gas.ion_pairs_per_cm3_per_Gy()
    # (a) pulse structure: dose per pulse as stems
    a = ax[0, 0]
    for rate, r in runs.items():
        starts = r["src"].support()
        t_s = np.array([lo for lo, _ in starts if lo < 0.04])
        a.vlines(t_s * 1e3, 0, r["n0"] / per_gy * 1e3, color=r["color"], lw=1.5,
                 label=f"{rate} MU/min, period {r['period'] * 1e3:.1f} ms")  # fmt: skip
    a.set(xlabel="time (ms)", ylabel="dose per pulse in air (mGy), 4 µs pulses",
          title="(a) Pulse structure (H2: same dose per pulse)", xlim=(-1, 40))  # fmt: skip
    a.legend(loc="upper right")
    # (b) chamber current
    a = ax[0, 1]
    edges = np.linspace(0, 0.04, 2001)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for rate, r in runs.items():
        i = (
            np.diff(r["ep"].cumulative("induced", edges))
            / np.diff(edges)
            * ELEMENTARY_CHARGE_C
            * area
        )
        a.plot(mid * 1e3, i * 1e6, color=r["color"], lw=1.2, label=f"{rate} MU/min")
    a.set(xlabel="time (ms)", ylabel="chamber current (µA)", title="(b) Current: 10 mm gap, 100 V")
    a.legend(loc="upper right")
    # (c) cumulative
    a = ax[0, 2]
    t = np.linspace(0, t_end + 0.02, 600)
    for rate, r in runs.items():
        prod, coll, reco = cumulative_fractions(r["ep"], t)
        a.plot(t * 1e3, coll, color=r["color"], lw=1.4, label=f"collected, {rate} MU/min")
        a.plot(t * 1e3, reco, color=r["color"], lw=1.0, ls="--", label=f"recombined, {rate} MU/min")
    a.plot(t * 1e3, cumulative_fractions(runs[400]["ep"], t)[0], **LIMIT, label="produced (400)")
    a.plot(t * 1e3, cumulative_fractions(runs[2400]["ep"], t)[0], color=MUTED, lw=1.0, ls=":",
           label="produced (2400)")  # fmt: skip
    a.set(xlabel="time (ms)", ylabel="fraction of the ion pairs of the run",
          title="(c) Cumulative charge, first 100 ms")  # fmt: skip
    a.legend(loc="upper left", fontsize=7)
    # (d) efficiency of each period
    a = ax[1, 0]
    for rate, r in runs.items():
        # windows centred between pulses, so that each holds exactly one pulse
        bounds = np.concatenate(([0.0], np.arange(0.5, t_end / r["period"], 1.0) * r["period"]))
        dp = np.diff(r["ep"].cumulative("source", bounds))
        dc = np.diff(r["ep"].cumulative("induced", bounds))
        a.plot(
            np.arange(dp.size) + 1, dc / dp, "o-", color=r["color"], ms=3, label=f"{rate} MU/min"
        )
    a.set(xlabel="pulse number", ylabel="collected / produced in each period",
          title="(d) Build-up to the periodic state")  # fmt: skip
    a.legend(loc="upper right")
    # (e, f) field and ions at a few times (2400 MU/min)
    r = runs[2400]
    snap = r["ep"].snapshots
    z = snap.z_cm * 10
    zf = np.linspace(0, ch.gap_cm * 10, snap.field_V_cm.shape[1])
    cmap = plt.get_cmap(SEQUENTIAL)
    picks = np.searchsorted(snap.times_s, np.array([0.3, 1.5, 3.2, 6.0, 12.0]) * 1e-3)
    for j, k in enumerate(picks):
        c = cmap(j / (len(picks) - 1))
        ax[1, 1].plot(
            zf,
            snap.field_V_cm[k] / ch.field_V_cm,
            color=c,
            label=f"t = {snap.times_s[k] * 1e3:.1f} ms",
        )
        ax[1, 2].plot(z, snap.n_pos_cm3[k] / 1e8, color=c, lw=1.4)
        ax[1, 2].plot(z, snap.n_neg_cm3[k] / 1e8, color=c, lw=1.0, ls="--")
    ax[1, 1].axhline(1.0, **LIMIT)
    ax[1, 1].set(xlabel="z (mm), anode at 0", ylabel="E / (V/d)",
                 title="(e) Space-charge field, 2400 MU/min")  # fmt: skip
    ax[1, 1].legend(loc="upper center", fontsize=7)
    ax[1, 2].set(xlabel="z (mm), anode at 0", ylabel="density (10⁸ cm⁻³): n⁺ solid, n⁻ dashed",
                 title="(f) Ions of overlapping pulses, 2400 MU/min")  # fmt: skip
    fig.suptitle(
        "Risø 2026 · 10 MV FFF · extrapolation chamber, 10 mm, 100 V · default model",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(FIG / "diag_risoe2026.png")
    plt.close(fig)

    # --- animation: isolated against overlapping pulses
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3)
    a_src = fig.add_subplot(gs[0, :2])
    a_cum = fig.add_subplot(gs[0, 2])
    a_400 = fig.add_subplot(gs[1, 0])
    a_2400 = fig.add_subplot(gs[1, 1])
    a_e = fig.add_subplot(gs[1, 2])
    edges = np.linspace(0, 0.02, 1001)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for rate, r in runs.items():
        i = (
            np.diff(r["ep"].cumulative("induced", edges))
            / np.diff(edges)
            * ELEMENTARY_CHARGE_C
            * area
        )
        a_src.plot(mid * 1e3, i * 1e6, color=r["color"], lw=1.2, label=f"current, {rate} MU/min")
        starts = np.array([lo for lo, _ in r["src"].support() if lo < 0.02])
        a_src.vlines(starts * 1e3, 0, -0.08 * max(i.max(), 1e-12) * 1e6, color=r["color"], lw=2)
    a_src.set(xlabel="time (ms)", ylabel="chamber current (µA)",
              title="Pulses (ticks below 0) and chamber current")  # fmt: skip
    a_src.legend(loc="upper right")
    cursor = a_src.axvline(0, color=INK2, lw=1.2)
    t_c = np.linspace(0, 0.02, 400)
    for rate, r in runs.items():
        _, coll, reco = cumulative_fractions(r["ep"], t_c)
        scale = r["ep"].produced_cm2 / r["ep"].cumulative("source", 0.02)
        a_cum.plot(t_c * 1e3, reco * scale, color=r["color"], lw=1.4, label=f"recombined, {rate}")
    a_cum.set(xlabel="time (ms)", ylabel="recombined / produced so far", title="Cumulative loss")
    a_cum.legend(loc="lower right", fontsize=8)
    dot = a_cum.axvline(0, color=INK2, lw=1.2)
    lines = {}
    ymax = max(r["ep"].snapshots.n_pos_cm3.max() for r in runs.values()) / 1e8 * 1.1
    for a, rate in ((a_400, 400), (a_2400, 2400)):
        s_ = runs[rate]["ep"].snapshots
        (lp,) = a.plot(z, s_.n_pos_cm3[0] / 1e8, color=S1, lw=1.5, label="n⁺ → cathode")
        (ln,) = a.plot(z, s_.n_neg_cm3[0] / 1e8, color=S2, lw=1.5, label="n⁻ → anode")
        a.set(ylim=(0, ymax), xlabel="z (mm), anode at 0", ylabel="density (10⁸ cm⁻³)",
              title=f"{rate} MU/min")  # fmt: skip
        a.legend(loc="upper center", fontsize=8)
        lines[rate] = (lp, ln)
    e_lines = {}
    for rate, r in runs.items():
        (le,) = a_e.plot(zf, r["ep"].snapshots.field_V_cm[0] / ch.field_V_cm, color=r["color"], lw=1.5,
                         label=f"{rate} MU/min")  # fmt: skip
        e_lines[rate] = le
    all_e = (
        np.concatenate([r["ep"].snapshots.field_V_cm.ravel() for r in runs.values()])
        / ch.field_V_cm
    )
    a_e.set(ylim=(all_e.min() - 0.05, all_e.max() + 0.05), xlabel="z (mm), anode at 0",
            ylabel="E / (V/d)", title="Field (space charge)")  # fmt: skip
    a_e.axhline(1.0, **LIMIT)
    a_e.legend(loc="upper center", fontsize=8)
    title = fig.suptitle("", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    times = runs[400]["ep"].snapshots.times_s

    def update(k):
        t_ms = times[k] * 1e3
        cursor.set_xdata([t_ms, t_ms])
        dot.set_xdata([t_ms, t_ms])
        for rate in runs:
            s_ = runs[rate]["ep"].snapshots
            lines[rate][0].set_ydata(s_.n_pos_cm3[k] / 1e8)
            lines[rate][1].set_ydata(s_.n_neg_cm3[k] / 1e8)
            e_lines[rate].set_ydata(s_.field_V_cm[k] / ch.field_V_cm)
        title.set_text(f"10 MV FFF, 10 mm gap, 100 V · t = {t_ms:.2f} ms · same dose per pulse; "
                       "at 2400 MU/min the next pulse arrives before the ions of the last one are gone")  # fmt: skip
        return cursor, dot, title

    save_gif(FuncAnimation(fig, update, frames=len(times), blit=False), "diag_risoe2026_gap.gif")
    plt.close(fig)
    return {rate: r["ep"].f_recombination for rate, r in runs.items()}


def write_report(d, r, runtime):
    (OUT / "DIAGNOSTICS.md").write_text(
        f"""# Diagnostics: charge carriers in the chamber over time

Generated by `python validation/diagnostics.py` ({runtime:.0f} s). All runs use the default model
(free electrons and space charge). The benchmarks themselves are in
[BENCHMARK_DCPT2019.md](BENCHMARK_DCPT2019.md) and [BENCHMARK_RISOE2026.md](BENCHMARK_RISOE2026.md).

## DCPT 2019 · 244 MeV protons, DRF 1, Roos chamber (2 mm)

![DCPT diagnostics](figures/diag_dcpt2019.png)

(a) The local dose rate at the chamber centre over the whole 41 × 41 layer. Every spot within a
few σ adds a rectangular pulse, and the passing rows show up as groups. (b) The spots passing
closest to the centre, together with the charge collected at that point at 50 V. Collection
lags the ionisation by the ion transit time, and it falls short by the recombination loss.
(c) The chamber current over the layer at 50 and 200 V. (d) Cumulative produced, collected and
recombined charge for the whole chamber. Chamber k_s: {d["ks50"]:.4f} at 50 V, {d["ks200"]:.4f} at
200 V. (e) The variation of k_s − 1 over the Roos area at 50 V is below 10 ppm. With 2.5 mm spot
spacing and 3.8 mm σ, every point sees nearly the same sequence of spots, only shifted in time.
(f) Ion profiles at the centre: building up at the start of the closest spot, in steady state
during it, and draining after it.

**Ions in the gap while spots pass (50 V).** Each spot builds up the positive (drifting to the
cathode) and negative (drifting to the anode) ion profiles, and the gap drains in the dead time.

![DCPT gap animation](figures/diag_dcpt2019_gap.gif)

**One 244 MeV proton track (3D track model, 200 V).** The positive and negative columns slide
apart along the field while diffusion widens them. Recombination inside the track (initial
recombination) ends once they have separated. k_s − 1 of one track: {d["ks_track"] - 1:.1e}.

![DCPT track animation](figures/diag_dcpt2019_track.gif)

## Risø 2026 · 10 MV FFF, extrapolation chamber, 10 mm gap, 100 V (10 V/mm)

![Risø diagnostics](figures/diag_risoe2026.png)

(a) The pulse structure under H2 (the same dose per pulse at both dose rates; see the benchmark).
(b) The chamber current. At 400 MU/min the current falls to zero before the next pulse. At
2400 MU/min it does not, because the next pulse arrives while the slow positive ions (transit
time 7.4 ms) are still in the gap. (c) Cumulative charge. (d) The collection efficiency of each
pulse period, in windows centred between pulses. At 400 MU/min every pulse is independent. At
2400 MU/min it settles from the second pulse on at a lower value, because each pulse now meets
the ions of the previous ones. The first window is only half a period long, which is why its
value is low. Efficiency
over the first 100 ms: {r[400]:.3f} at 400 MU/min and {r[2400]:.3f} at 2400 MU/min.
(e) The space-charge field during the overlap, and (f) the ion profiles of successive pulses.

**Isolated against overlapping pulses.**

![Risø gap animation](figures/diag_risoe2026_gap.gif)
""",
        encoding="utf-8",
    )


def main():
    t0 = time.perf_counter()
    FIG.mkdir(parents=True, exist_ok=True)
    d = run_dcpt()
    r = run_risoe()
    write_report(d, r, time.perf_counter() - t0)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
