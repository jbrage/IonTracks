"""Validation of the track-resolved 3D model (Tile3D).

  fig9  (a) one track against Jaffe's theory over LET, and IonTracks v3 (legacy solver)
        (b) the voltage dependence of initial recombination (1/V)
        (c) convergence with the lateral and axial grid
        (d) from initial to general recombination: a continuous proton beam at rising fluence
            rate, against the coupled model (initial recombination feeding the 1D solver)
        (e, f) a carbon track sliding apart: x-z slices of the positive and negative ions

Writes validation/figures/fig9_tile3d.png and validation/VALIDATION_TILE3D.md.
Run from the repository root:  python validation/validate_tile3d.py   (~3 min)
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotstyle
from plotstyle import INK2, LIMIT, S1, S2, S3, SEQUENTIAL, THEORY

from iontracks import (
    Gas,
    GaussianTrack,
    Grid1D,
    Grid3D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Tile3D,
    Uniform1D,
    track_radius_cm,
)
from iontracks.analytic import jaffe_parallel_f, jaffe_parameters

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
FIG = OUT / "figures"
CLASSIC = {"free_electrons": False, "space_charge": False}
LEGACY = Gas.air_legacy()  # the constants of Jaffe's comparison in IonTracks v3 (one mu, one D)
GRID = Grid3D(n_xy=48, n_z=100)
CHECKS: list[dict] = []
D_ROWS: list[tuple] = []
plotstyle.apply()


def check(name, value, tol, fmt=".2e", *, below=True):
    ok = value <= tol if below else value >= tol
    rel = "<=" if below else ">="
    CHECKS.append({"name": name, "value": f"{value:{fmt}}", "tol": f"{rel} {tol:{fmt}}", "ok": ok})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:{fmt}} ({rel} {tol:{fmt}})")


def legacy_track(let_keV_um):
    return GaussianTrack(let_keV_um * 1e7 / LEGACY.W_eV, track_radius_cm(let_keV_um))


def single(track, chamber, grid=GRID, gas=LEGACY, physics=CLASSIC):
    return Tile3D(track, grid, **physics).single_track(chamber, gas).ks - 1


def main():
    t0 = time.perf_counter()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))

    # (a) one track against Jaffe over LET, 1 mm, 200 V and 300 V; IonTracks v3 results
    print("(a) single track vs Jaffe over LET")
    a = ax[0, 0]
    lets = np.logspace(-3.4, -0.5, 8)  # keV/um in air: protons to heavy ions
    line = np.logspace(-3.5, -0.4, 100)
    worst_small_g = 0.0
    rows = []
    for volts, color in ((200.0, S1), (300.0, S2)):
        ch = ParallelPlateChamber(0.1, volts)
        a.plot(line, [1 / jaffe_parallel_f(legacy_track(x), ch, LEGACY) - 1 for x in line],
               color=color, lw=1.2, label=f"Jaffe, {volts:g} V")  # fmt: skip
        sim = []
        for let in lets:
            trk = legacy_track(let)
            ks3 = single(trk, ch)
            ksj = 1 / jaffe_parallel_f(trk, ch, LEGACY) - 1
            g, _ = jaffe_parameters(trk, ch, LEGACY)
            if g <= 0.05:
                worst_small_g = max(worst_small_g, abs(ks3 / ksj - 1))
            rows.append((volts, let, g, ksj, ks3))
            sim.append(ks3)
        a.plot(lets, sim, "o", color=color, label=f"Tile3D, {volts:g} V")
    ref = list(csv.DictReader(open(OUT / "data" / "iontracks_v3" / "single_track_ks.csv")))
    for r in ref:
        color = S1 if float(r["voltage_V"]) == 200 else S2
        a.plot(float(r["LET_keV_um"]), float(r["ks"]) - 1, "x", color=color, ms=8, mew=1.5)
    a.plot([], [], "x", color=INK2, ms=8, mew=1.5, label="IonTracks v3 (legacy solver)")
    a.set(xscale="log", yscale="log", xlabel="LET in air (keV/µm)", ylabel="k_s − 1 (one track)",
          title="(a) Initial recombination, 1 mm gap")  # fmt: skip
    a.legend(loc="upper left")
    check("single track vs Jaffe, max rel. deviation of k_s-1 (g <= 0.05)", worst_small_g, 0.02)
    v3_dev = max(
        abs(single(legacy_track(float(r["LET_keV_um"])),
                   ParallelPlateChamber(0.1, float(r["voltage_V"]))) / (float(r["ks"]) - 1) - 1)
        for r in ref
    )  # fmt: skip
    check("single track vs IonTracks v3 (8 cases), max rel. deviation of k_s-1", v3_dev, 0.03)

    # (b) voltage dependence, 2 mm
    print("(b) voltage dependence")
    a = ax[0, 1]
    volts = np.array([50.0, 83.0, 125.0, 200.0, 300.0, 400.0])
    worst_v = 0.0
    for let, color, name in ((1.14e-3, S1, "60 MeV p"), (4.1e-2, S3, "60 MeV/u C")):
        trk = legacy_track(let)
        vl = np.linspace(40, 450, 80)
        jl = [1 / jaffe_parallel_f(trk, ParallelPlateChamber(0.2, v), LEGACY) - 1 for v in vl]
        ref_scale = 1 / jaffe_parallel_f(trk, ParallelPlateChamber(0.2, 200.0), LEGACY) - 1
        a.plot(1 / vl, np.array(jl) / ref_scale, color=color, lw=1.2, label=f"Jaffe, {name}")
        sim = []
        for v in volts:
            ch = ParallelPlateChamber(0.2, v)
            ks3 = single(trk, ch)
            worst_v = max(worst_v, abs(ks3 / (1 / jaffe_parallel_f(trk, ch, LEGACY) - 1) - 1))
            sim.append(ks3 / ref_scale)
        a.plot(1 / volts, sim, "o", color=color, label=f"Tile3D, {name}")
    a.set(xlabel="1 / V (1/V)", ylabel="(k_s − 1) / (k_s − 1 at 200 V, Jaffe)",
          title="(b) Voltage dependence, 2 mm gap")  # fmt: skip
    a.legend(loc="upper left")
    check("single track vs Jaffe over 50-400 V (2 mm), max rel. deviation of k_s-1", worst_v, 0.02)

    # (c) convergence
    print("(c) grid convergence")
    a = ax[0, 2]
    trk = legacy_track(4.1e-2)
    ch = ParallelPlateChamber(0.1, 200.0)
    ksj = 1 / jaffe_parallel_f(trk, ch, LEGACY) - 1
    nxy = np.array([16, 24, 32, 48, 64, 96])
    err_xy = [abs(single(trk, ch, Grid3D(n_xy=int(n), n_z=100)) / ksj - 1) for n in nxy]
    tile = Tile3D(trk, GRID, **CLASSIC).tile_size_cm(ch, LEGACY)
    a.plot(tile / nxy / trk.radius_cm, err_xy, "o-", color=S1, label="lateral cells (n_z = 100)")
    nz = np.array([25, 50, 100, 200])
    err_z = [abs(single(trk, ch, Grid3D(n_xy=48, n_z=int(n))) / ksj - 1) for n in nz]
    a.plot(ch.gap_cm / nz / trk.radius_cm, err_z, "s-", color=S2,
           label="axial cells (n_xy = 48)")  # fmt: skip
    a.set(xscale="log", yscale="log", xlabel="cell size / track radius b",
          ylabel="|k_s − 1 (Tile3D) / Jaffe − 1|", title="(c) Grid convergence (60 MeV/u C)")  # fmt: skip
    a.legend(loc="upper left")
    order = np.polyfit(np.log(tile / nxy[-4:]), np.log(err_xy[-4:]), 1)[0]
    check("lateral convergence order towards Jaffe (last 4 grids)", order, 0.8, ".2f", below=False)

    # (d) from initial to general recombination
    print("(d) initial -> general recombination (continuous protons)")
    a = ax[1, 0]
    gas = Gas.air(particle="proton")
    ch = ParallelPlateChamber(0.2, 200.0)
    trk = GaussianTrack.from_stopping_power(5.44, gas)  # ~150 MeV protons in air
    model = Tile3D(trk, Grid3D(n_xy=40, n_z=80), seed=7)
    f_init = 1 / (model.single_track(ch, gas).ks)
    duration = 4 * ch.collection_times_s(gas)[0]
    rates = np.logspace(-1, 2.5, 7)  # Gy/s in air
    per_gy = gas.ion_pairs_per_cm3_per_Gy()
    seeds = (1, 2, 3, 4)
    k3, se3, kc, kg = [], [], [], []
    for rate in rates:
        src = PiecewiseConstant.constant(rate * per_gy, 0, duration)
        runs = [
            Tile3D(trk, Grid3D(n_xy=40, n_z=80), seed=s).run_episode(src, ch, gas).ks - 1
            for s in seeds
        ]
        k3.append(np.mean(runs))
        se3.append(np.std(runs, ddof=1) / np.sqrt(len(seeds)))
        gen = Uniform1D(Grid1D(n_cells=80)).run_episode(src.scaled(f_init), ch, gas)
        kc.append(1 / (f_init * gen.f_recombination) - 1)
        kg.append(Uniform1D(Grid1D(n_cells=80)).run_episode(src, ch, gas).ks - 1)
    k3, se3, kc = np.array(k3), np.array(se3), np.array(kc)
    a.axhline(1 / f_init - 1, **LIMIT, label="initial only (one track)")
    a.plot(rates, kg, color=S3, lw=1.2, label="general only (uniform 1D)")
    a.plot(rates, kc, **THEORY, label="coupled: initial feeding general")
    a.errorbar(rates, k3, yerr=2 * se3, fmt="o", color=S1, capsize=2,
               label=f"Tile3D, mean of {len(seeds)} seeds (± 2 SE)")  # fmt: skip
    a.set(xscale="log", yscale="log", xlabel="dose rate in air (Gy/s)", ylabel="k_s − 1",
          title="(d) 150 MeV protons, 2 mm, 200 V")  # fmt: skip
    a.legend(loc="upper left")
    overlap = rates >= 20
    check("Tile3D vs coupled model where tracks overlap (>= 20 Gy/s), max rel. deviation",
          float(np.max(np.abs(k3[overlap] / kc[overlap] - 1))), 0.01)  # fmt: skip
    z = np.abs(k3 - kc) / np.maximum(se3, 1e-30)
    check("Tile3D vs coupled model at all rates, max |difference| / SE (4 seeds)",
          float(z.max()), 3.0, ".2f")  # fmt: skip
    D_ROWS.extend(zip(rates, k3, se3, kc, strict=True))

    # (e, f) a carbon track sliding apart
    print("(e, f) track snapshots")
    trk = legacy_track(4.1e-2)
    ch = ParallelPlateChamber(0.2, 200.0)
    tc = ch.gap_cm / (LEGACY.mu_pos_cm2_Vs * ch.field_V_cm)
    m = Tile3D(trk, Grid3D(n_xy=48, n_z=100), **CLASSIC)
    snap = m.single_track(ch, LEGACY, snapshot_times_s=np.array([0.02, 0.35, 0.7]) * tc).snapshots
    x = (snap.x_cm - snap.x_cm.mean()) * 1e4
    for a, dens, name in ((ax[1, 1], snap.n_pos_xz_cm3, "positive"), (ax[1, 2], snap.n_neg_xz_cm3, "negative")):  # fmt: skip
        stacked = np.concatenate([d.T for d in dens], axis=1)
        a.imshow(np.log10(np.maximum(stacked, 1.0)), origin="lower", aspect="auto", cmap=SEQUENTIAL,
                 extent=(0, 3 * (x[-1] - x[0]), 0, ch.gap_cm * 10), vmin=4)  # fmt: skip
        for k, t in enumerate(snap.times_s):
            a.text((k + 0.5) * (x[-1] - x[0]), ch.gap_cm * 10 * 1.01, f"t = {t / tc:.2f} t_c",
                   ha="center", va="bottom", fontsize=8)  # fmt: skip
            if k:
                a.axvline(k * (x[-1] - x[0]), color="white", lw=1)
        a.set(xticks=[], xlabel=f"x (three snapshots, each {x[-1] - x[0]:.0f} µm wide)",
              ylabel="z (mm), anode at 0")  # fmt: skip
        a.set_title(f"({'e' if name == 'positive' else 'f'}) {name} ions, one carbon track, 200 V",
                    pad=16)  # fmt: skip
        a.grid(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig9_tile3d.png")
    plt.close(fig)
    write_report(rows, time.perf_counter() - t0)
    print(
        f"{sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} checks pass; {time.perf_counter() - t0:.0f} s"
    )
    if not all(c["ok"] for c in CHECKS):
        raise SystemExit(1)


def write_report(rows, runtime):
    checks = "\n".join(
        f"| {'✅' if c['ok'] else '❌'} | {c['name']} | {c['value']} | {c['tol']} |" for c in CHECKS
    )
    d_table = "\n".join(
        f"| {r:.3g} | {k:.4e} | {se:.1e} | {c:.4e} | {k / c:.3f} |" for r, k, se, c in D_ROWS
    )
    table = "\n".join(
        f"| {v:g} | {let:.3g} | {g:.2e} | {kj:.4e} | {k3:.4e} | {k3 / kj - 1:+.2%} |"
        for v, let, g, kj, k3 in rows
    )
    (OUT / "VALIDATION_TILE3D.md").write_text(
        f"""# Track-resolved 3D model (Tile3D): validation

Generated by `python validation/validate_tile3d.py` ({runtime:.0f} s). **{sum(c["ok"] for c in CHECKS)} of
{len(CHECKS)} checks pass.**

`Tile3D` simulates one lateral sample point as a periodic tile of the gap. Ion tracks cross the gap
perpendicular to the electrodes, at Poisson-distributed times and positions. Each track is a
Gaussian column: N0 ion pairs per cm, radius b from the IonTracks v3 LET table (floor 20 µm). The
ions drift along z, diffuse in 3D and recombine. Free electrons and space charge (mean field along
z) are those of the 1D model.

| | Check | Value | Criterion |
|---|---|---|---|
{checks}

![Figure 9](figures/fig9_tile3d.png)

(a) Initial recombination of one track against Jaffé's theory for a column parallel to the field.
The comparison uses the IonTracks v3 gas constants (one mobility, one diffusion coefficient), as
Jaffé's theory assumes. Crosses are the eight single-track results of the legacy v3 solver
(`validation/data/iontracks_v3/`). Jaffé's prescribed-diffusion approximation (the column keeps its
Gaussian shape) holds for weak recombination, g = αN0/(8πD) ≪ 1. At high LET the
simulation departs from it, as found by Christensen et al. (2016).
(b) The 1/V dependence of initial recombination, normalised to Jaffé at 200 V.
(c) Convergence towards Jaffé as the cells shrink relative to the track radius.
(d) A continuous 150 MeV proton beam at rising dose rate (default physics: free electrons and
space charge). At low rates every track recombines alone and k_s − 1 is the single-track value.
At high rates the tracks overlap and general recombination takes over. The coupled model
(the surviving fraction of each track, f_init, feeding the uniform 1D solver:
k_s = 1 / (f_init f_general(q f_init))) follows the full 3D simulation, which validates the
coupling used for scanned-field benchmarks.
(e, f) The positive and negative columns of one carbon track slide apart along the field, which
ends their recombination. Diffusion widens them in the meantime.

## Single tracks against Jaffé (1 mm gap)

| V | LET in air (keV/µm) | g | k_s − 1, Jaffé | k_s − 1, Tile3D | difference |
|---|---|---|---|---|---|
{table}

## From initial to general recombination (panel d)

| dose rate in air (Gy/s) | k_s − 1, Tile3D (mean of 4 seeds) | standard error | k_s − 1, coupled model | ratio |
|---|---|---|---|---|
{d_table}

With few tracks per tile at low rates, the general part of the recombination scatters between
seeds, and the coupled model agrees within the statistics. Where tracks overlap strongly, the two
agree to better than 1 %.

## Limitations

* Tracks are perpendicular to the electrodes. Oblique tracks (Jaffé's angle dependence) are not
  implemented.
* The track radius b is an effective, empirical input: the v3 table after Rossomme et al., with a
  20 µm floor that applies to all protons. No independent data constrain it here.
* The periodic tile carries the images of every track. By default it spans three track widths
  after one slow-ion transit; the images then meet only at low density, and (c) shows that the
  lateral cell size, not the tile size, sets the accuracy.
* Space charge acts through the laterally averaged field. The lateral field of a single track
  (below 0.1 V/cm for protons, ~10 V/cm for heavy ions) is neglected.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
