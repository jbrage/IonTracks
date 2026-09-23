"""Reproduce the numerical findings quoted in report/code_review_and_refactoring_plan.md.

Run from the repository root (needs numpy + numba only, no Cython build). On the v4 branch the
legacy code lives under legacy/; on master (tag v3-legacy) it is at the repository root.

    python report/verification/verify_findings.py            # all checks (~1-3 min)
    python report/verification/verify_findings.py --quick    # skip the multi-seed scoring study

Checks
  A. Numba continuous-beam backend scores recombination from t=0, the Cython reference only after
     build-up -> different k_s for identical seeds.
  B. Scoring-window bias: production scored over 2 separation times, recombination over 3 (drain
     double-counted) vs. consistent same-window estimators with different build-up lengths.
  C. Positivity of the repo's Lax-Wendroff z-stencil (incl. diffusion) at a column end.
  D. Cost model: time steps per collection time, voxel counts, alpha*n*dt for uniform high DPP.
"""

import contextlib
import io
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "legacy" if (REPO / "legacy" / "hadrons").is_dir() else REPO
sys.path.insert(0, str(LEGACY))
NUMBA_SRC = LEGACY / "hadrons" / "numba_files" / "continuous_beam_numba.py"

D_ION, MU_ION = 3.7e-2, 1.65  # repo defaults, cm^2/s and cm^2/(V s)


def _patched_numba_solver():
    """Load the repo's Numba solver with a configurable build-up / scoring window.

    extra_params_dic keys added: n_total (insertion window in separation times), n_buildup,
    n_drain, score_end (last scored step). Defaults reproduce the Cython reference logic.
    """
    src = NUMBA_SRC.read_text()
    edits = [
        (
            "    n_separation_times = 3\n",
            "    n_separation_times = extra_params_dic.get('n_total', 3)\n"
            "    buildup_steps = int(extra_params_dic.get('n_buildup', 1) * separation_time_steps)\n",
        ),
        (
            "dont_initialise_tracks = np.zeros(separation_time_steps)",
            "dont_initialise_tracks = np.zeros(int(extra_params_dic.get('n_drain', 1) * separation_time_steps))",
        ),
        (
            "            inner_radius,\n        )\n        stop_calculating_new_densites",
            "            inner_radius,\n"
            "            buildup_steps < time_step <= extra_params_dic.get('score_end', 10**9),\n"
            "        )\n        stop_calculating_new_densites",
        ),
        (
            "    mid_xy_array,\n    inner_radius,\n):\n    no_recombined = 0.0",
            "    mid_xy_array,\n    inner_radius,\n    score,\n):\n    no_recombined = 0.0",
        ),
        (
            "                if no_z_electrode < k < (no_z + no_z_electrode):\n                    if (",
            "                if score and no_z_electrode < k < (no_z + no_z_electrode):\n                    if (",
        ),
        (
            "            time_step,\n            separation_time_steps,\n            positive_array,",
            "            time_step if time_step <= extra_params_dic.get('score_end', 10**9) else -1,\n"
            "            buildup_steps,\n            positive_array,",
        ),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(f"repo source changed; cannot apply patch:\n{old}")
        src = src.replace(old, new)
    mod = types.ModuleType("continuous_beam_numba_patched")
    exec(compile(src, "continuous_beam_numba_patched", "exec"), mod.__dict__)
    return mod.continuous_beam_PDEsolver


def _quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


PARAMS = dict(LET_keV_um=0.02, voltage_V=100.0, electrode_gap_cm=0.05)
EXTRA = dict(track_radius_cm=2e-3, SHOW_PLOT=False, PRINT_parameters=False, unit_length_cm=10e-4)


def check_a(patched):
    from hadrons.numba_files.continuous_beam_numba import continuous_beam_PDEsolver as repo_solver

    print("A. Numba backend (repo) vs. same code gated like the Cython reference")
    print(f"   {'fluence':>9} {'seed':>4} {'ks repo Numba':>14} {'ks gated':>10}")
    for flu in (1e9, 2e10):
        for seed in (1, 2):
            p = dict(PARAMS, fluencerate_cm2_s=flu)
            a = _quiet(repo_solver, p, dict(EXTRA, seed=seed))
            b = _quiet(patched, p, dict(EXTRA, seed=seed))
            print(
                f"   {flu:9.1e} {seed:4d} {a:14.6f} {b:10.6f}   (ks-1 ratio {((a - 1) / (b - 1)):.2f})"
            )


def _separation_steps(d_cm, V, dx):
    E, dt = V / d_cm, 1.0
    while True:  # identical to the repo loop
        dt /= 1.01
        s, cz = D_ION * dt / dx**2, MU_ION * E * dt / dx
        if 6 * s + cz**2 <= 1:
            return int(d_cm / (2 * MU_ION * E * dt))


def check_b(patched, n_seeds=6):
    sep = _separation_steps(
        PARAMS["electrode_gap_cm"], PARAMS["voltage_V"], EXTRA["unit_length_cm"]
    )
    schemes = {
        "repo: build-up d/2uE, P over 2 sep, R over 3 sep": dict(n_total=3, n_buildup=1, n_drain=1),
        "same window, build-up d/2uE (sep..3sep)": dict(n_total=3, n_buildup=1, score_end=3 * sep),
        "same window, build-up d/uE  (2sep..6sep)": dict(n_total=6, n_buildup=2, score_end=6 * sep),
        "same window, build-up 2d/uE (4sep..8sep)": dict(n_total=8, n_buildup=4, score_end=8 * sep),
    }
    p = dict(PARAMS, fluencerate_cm2_s=2e10)
    print(f"\nB. Scoring-window study ({n_seeds} seeds, fluence rate 2e10 /cm2/s)")
    for name, kw in schemes.items():
        ks = (
            np.array([_quiet(patched, p, dict(EXTRA, seed=s, **kw)) for s in range(1, n_seeds + 1)])
            - 1
        )
        print(
            f"   {name:50s} (ks-1)*1e3 = {ks.mean() * 1e3:6.3f} +- {ks.std(ddof=1) / np.sqrt(n_seeds) * 1e3:5.3f}"
        )


def check_c():
    print("\nC. Lax-Wendroff z-stencil with diffusion, repo dt choice, sharp column end")
    for d, V, dx in [(0.2, 200, 5e-4), (0.1, 300, 5e-4)]:
        E, dt = V / d, 1.0
        while True:
            dt /= 1.01
            s, cz = D_ION * dt / dx**2, MU_ION * E * dt / dx
            if 6 * s + cz**2 <= 1:
                break
        w_m, w_p, w_0 = s + cz * (cz + 1) / 2, s + cz * (cz - 1) / 2, 1 - cz**2 - 2 * s
        u = np.zeros(400)
        u[20:300] = 1.0
        lo = 0.0
        for _ in range(150):
            u[1:-1] = w_m * u[:-2] + w_p * u[2:] + w_0 * u[1:-1]
            lo = min(lo, u.min())
        print(
            f"   d={d} cm V={V}: cz={cz:.3f} sz={s:.3f} downstream weight={w_p:+.4f} "
            f"min density={lo:+.4f} (column=1), cell Peclet={MU_ION * E * dx / D_ION:.0f}"
        )


def check_d():
    print("\nD. Cost model (current explicit scheme, repo constants; t_c uses mu+ = 1.36)")
    print(
        f"   {'gap':>5} {'V':>4} {'dx um':>5} {'dt ns':>6} {'t_c us':>7} {'steps/t_c':>9} {'voxels':>9}"
    )
    for d, V, dx_um in [(0.2, 200, 5), (0.1, 300, 5), (0.2, 200, 2), (0.2, 200, 1)]:
        dx, E = dx_um * 1e-4, V / d
        a, b = (MU_ION * E / dx) ** 2, 6 * D_ION / dx**2
        dt = (-b + np.sqrt(b * b + 4 * a)) / (2 * a)  # closed form of the repo criterion
        tc = d / (1.36 * E)
        nxy, nz = int(2 * 0.012 / dx) + 20, int(d / dx) + 10
        print(
            f"   {d:5.2f} {V:4d} {dx_um:5d} {dt * 1e9:6.0f} {tc * 1e6:7.1f} {tc / dt:9.0f} {nxy * nxy * nz:9.2e}"
        )
    pairs_per_gy = 1.205e-6 / (33.97 * 1.602e-19)
    print(f"   ion pairs/cm3/Gy in air = {pairs_per_gy:.2e}; alpha*n*dt at dt=265 ns:")
    for dpp in (0.01, 0.1, 1.0, 10.0):
        print(f"     {dpp:5.2f} Gy/pulse -> {1.6e-6 * dpp * pairs_per_gy * 265e-9:.2e}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    solver = _patched_numba_solver()
    check_a(solver)
    if not quick:
        check_b(solver)
    check_c()
    check_d()
