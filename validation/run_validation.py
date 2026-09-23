"""Validation suite for the IonTracks v4 point model and field solver.

Every figure compares the implementation against an independent reference:

  fig1  analytic limits: Boag pulsed, grid convergence, Boag continuous, and conservation
        and Ramo residuals over random scenarios
  fig2  transport: density profiles and induced current against the exact solution of Boag's
        model (drift + recombination, no diffusion), and the scheme at a charge front
  fig3  time structure: pulse-width and pulse-period sweeps between their analytic limits,
        and build-up and drain of a continuous beam
  fig4  steady-state skipping against full stepping, its cost, and grid convergence
  fig5  synthetic 250 MeV cyclotron PBS field (2 mm, 200 V): k_s against quasi-static Boag,
        the transient effect, the k_s map, the current trace, and lateral convergence
  fig8  free electrons (exact fraction, Fenwick-Kumar exact solution) and space charge
        (onset scaling, field screening)

Writes validation/figures/*.png and validation/VALIDATION.md (with a table of quantitative
checks). Run from the repository root:  python validation/run_validation.py   (~1-2 min)
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotstyle  # noqa: E402
from plotstyle import (  # noqa: E402
    AXIS,
    INK2,
    LIMIT,
    MUTED,
    S1,
    S2,
    S3,
    SEQUENTIAL,
    SHADE,
    SURFACE,
    THEORY,
)

from iontracks import (  # noqa: E402
    Gas,
    Grid1D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Uniform1D,
    solve_field,
    solve_point,
)
from iontracks.analytic import (  # noqa: E402
    boag_continuous_xi2,
    boag_pulsed_f,
    boag_pulsed_free_electron_f,
    boag_pulsed_u,
    boag_quasi_static_loss,
    space_charge_parameter,
    space_charge_parameter_pulsed,
)
from iontracks.constants import ELEMENTARY_CHARGE_C  # noqa: E402
from iontracks.solvers.kernels import react as _react  # noqa: E402
from iontracks.solvers.kernels import transport as _transport  # noqa: E402
from iontracks.synthetic import (  # noqa: E402
    PROTON_250MEV_S_AIR_MEV_CM2_G,
    PROTON_250MEV_S_WATER_MEV_CM2_G,
    cyclotron_pbs_field,
)

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"

plotstyle.apply()

CHECKS: list[dict] = []


def check(name: str, value: float, tol: float, fmt: str = ".2e", *, below: bool = True) -> None:
    ok = value <= tol if below else value >= tol
    CHECKS.append(
        {
            "name": name,
            "value": f"{value:{fmt}}",
            "tol": f"{'<=' if below else '>='} {tol:{fmt}}",
            "ok": ok,
        }  # fmt: skip
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:{fmt}} ({'<=' if below else '>='} {tol:{fmt}})"
    )


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ shared set-up
CH = ParallelPlateChamber(gap_cm=0.2, voltage_V=200.0, sensitive_radius_cm=0.78)
GAS = Gas.air(particle="proton")
GAS0 = GAS.with_(D_pos_cm2_s=0.0, D_neg_cm2_s=0.0)  # Boag's theories neglect diffusion
V_POS, V_NEG = CH.drift_velocities_cm_s(GAS)
T_C = CH.gap_cm / V_POS  # slow (positive) ion transit time
INSTANT = 1e-12


def boag_model(**grid) -> Uniform1D:
    """Point model of Boag's classic theories: no free electrons, uniform field."""
    return Uniform1D(Grid1D(**grid), free_electrons=False, space_charge=False)


def boag_profiles(z, t, n0, alpha):
    """Exact n+(z, t), n-(z, t) after an instantaneous uniform pulse (drift + recombination)."""
    d, vs = CH.gap_cm, V_POS + V_NEG
    overlap = (z >= V_POS * t) & (z <= d - V_NEG * t)
    t_pos = np.clip((d - (z - V_POS * t)) / vs, 0, t)  # when a positive left the overlap
    t_neg = np.clip((z + V_NEG * t) / vs, 0, t)
    n_pos = np.where(z >= V_POS * t, n0 / (1 + alpha * n0 * np.where(overlap, t, t_pos)), 0.0)
    n_neg = np.where(z <= d - V_NEG * t, n0 / (1 + alpha * n0 * np.where(overlap, t, t_neg)), 0.0)
    return n_pos, n_neg


def step_profiles(model, n0, gas, times):
    """Run the kernels step by step from a uniform pulse and return profiles at ``times``."""
    st = model.steps(CH, gas)
    n = model.grid.n_cells
    n_pos, n_neg = np.full(n, n0), np.full(n, n0)
    n_neg_rev = n_neg[::-1]
    flux = np.zeros(n + 1)
    half = 0.5 * gas.alpha_cm3_s * st.dt_s
    out, k = {}, 0
    targets = {int(round(t / st.dt_s)): t for t in times}
    while k <= max(targets):
        if k in targets:
            out[targets[k]] = (n_pos.copy(), n_neg.copy())
        _react(n_pos, n_neg, half)
        _transport(n_pos, st.courant_pos, st.lam_pos, flux)
        _transport(n_neg_rev, st.courant_neg, st.lam_neg, flux)
        _react(n_pos, n_neg, half)
        k += 1
    z = (np.arange(n) + 0.5) * st.dz_cm
    return z, out, st


# ================================================================== figure 1
def figure1():
    print("fig1: analytic limits")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    u_unit = boag_pulsed_u(1.0, CH, GAS0)

    # (a) Boag pulsed
    model = boag_model(n_cells=400)
    u_sim = np.logspace(-2, 2, 9)
    f_sim, rel = [], []
    for u in u_sim:
        ep = model.run_episode(PiecewiseConstant.pulse(u / u_unit, 0, INSTANT), CH, GAS0)
        f_b = boag_pulsed_f(u / u_unit, CH, GAS0)
        f_sim.append(ep.f_recombination)
        rel.append(abs((1 - ep.f_recombination) / (1 - f_b) - 1))
    u_line = np.logspace(-2.2, 2.2, 300)
    a = ax[0, 0]
    a.plot(u_line, np.log1p(u_line) / u_line, **THEORY, label="Boag: ln(1+u)/u")
    a.plot(u_sim, f_sim, "o", color=S1, label="IonTracks (400 cells)")
    a.set(xscale="log", xlabel="u = αn₀d² / ((µ⁺+µ⁻)V)", ylabel="collection efficiency f",
          title="(a) Instantaneous pulse vs Boag")  # fmt: skip
    a.legend(loc="lower left")
    check("Boag pulsed, max rel. error in (1-f), u = 0.01...100", max(rel), 1e-3)

    # (b) convergence
    a = ax[0, 1]
    cells = np.array([25, 50, 100, 200, 400, 800, 1600])
    for u, color in zip((0.1, 1.0, 10.0), (S1, S2, S3), strict=True):
        f_b = boag_pulsed_f(u / u_unit, CH, GAS0)
        err = []
        for n in cells:
            ep = boag_model(n_cells=int(n)).run_episode(
                PiecewiseConstant.pulse(u / u_unit, 0, INSTANT), CH, GAS0
            )
            err.append(abs((1 - ep.f_recombination) / (1 - f_b) - 1))
        order = -np.polyfit(np.log(cells[-4:]), np.log(err[-4:]), 1)[0]
        a.plot(cells, err, "o-", color=color, label=f"u = {u:g} (order {order:.2f})")
        check(f"Boag pulsed convergence order, u = {u:g}", order, 1.0, ".2f", below=False)
    a.plot(cells, 0.02 / cells, **LIMIT, label="first order")
    a.set(xscale="log", yscale="log", xlabel="cells across the gap",
          ylabel="relative error in (1 − f)", title="(b) Grid convergence (pulsed)")  # fmt: skip
    a.legend(loc="lower left")

    # (c) Boag continuous, steady state
    a = ax[1, 0]
    xi2s = np.logspace(-4, -0.5, 8)
    ratios = []
    model = boag_model(n_cells=200)
    for xi2 in xi2s:
        q = xi2 / boag_continuous_xi2(1.0, CH, GAS0)
        ep = model.run_episode(PiecewiseConstant.constant(q, 0, 8 * T_C), CH, GAS0)
        t0, t1 = 3 * T_C, 8 * T_C - ep.dt_s
        loss = (
            np.diff(ep.cumulative("recombined", [t0, t1]))[0]
            / np.diff(ep.cumulative("source", [t0, t1]))[0]
        )
        ratios.append(loss / xi2)
    x = np.logspace(-4.2, -0.3, 200)
    a.axhline(1.0, **LIMIT, label="first order: loss = ξ²")
    a.plot(x, 1 / (1 + x), **THEORY, label="Boag: loss = ξ²/(1+ξ²)")
    a.plot(xi2s, ratios, "o", color=S1, label="IonTracks, steady state")
    a.set(xscale="log", xlabel="ξ² = αqd⁴ / (6µ⁺µ⁻V²)", ylabel="steady-state loss / ξ²",
          title="(c) Continuous beam vs Boag")  # fmt: skip
    a.legend(loc="lower left")
    check("Boag continuous, |loss/xi2 - 1| at xi2 = 1e-4", abs(ratios[0] - 1), 1e-3)

    # (d) conservation and Ramo residuals over random scenarios
    rng = np.random.default_rng(42)
    cons, ramo = [], []
    for _ in range(60):
        ch = ParallelPlateChamber(rng.uniform(0.05, 0.3), rng.uniform(50, 500))
        gas = Gas.air(particle="electron") if rng.random() < 0.7 else Gas.air_legacy()
        tc = ch.gap_cm / (gas.mu_pos_cm2_Vs * ch.field_V_cm)
        n_pairs = gas.ion_pairs_per_cm3_per_Gy() * 10 ** rng.uniform(-3, 0.5)
        kind = rng.integers(3)
        if kind == 0:
            src = PiecewiseConstant.pulse(n_pairs, 0, 10 ** rng.uniform(-9, -4))
        elif kind == 1:
            src = PiecewiseConstant.constant(n_pairs / tc, 0, tc * rng.uniform(0.5, 20))
        else:
            src = PiecewiseConstant.pulse_train(n_pairs, tc * 1e-3, tc * rng.uniform(0.05, 2), 10)
        physics = {
            "free_electrons": gas.has_free_electrons and rng.random() < 0.5,
            "space_charge": bool(rng.random() < 0.5),
        }
        ep = Uniform1D(Grid1D(n_cells=int(rng.integers(40, 300))), **physics).run_episode(
            src, ch, gas
        )
        cons.append(
            abs(ep.produced_cm2 - ep.recombined_cm2 - ep.collected_pos_cm2 - ep.remaining_cm2)
            / ep.produced_cm2
        )
        net = ep.collected_pos_cm2 - ep.backflow_pos_cm2 - ep.backflow_neg_cm2
        ramo.append(abs(ep.signal_cm2 - net) / ep.produced_cm2)
    cons = np.maximum(cons, 1e-17)
    ramo = np.maximum(ramo, 1e-17)
    a = ax[1, 1]
    idx = np.arange(60)
    a.plot(idx, cons, "o", color=S1, label="charge conservation: |P − R − C − remaining| / P")
    a.plot(idx, ramo, "s", color=S2, ms=5, label="Ramo: |∫I dt / e − C| / P")
    a.axhline(1e-10, **LIMIT, label="test tolerance")
    a.set(yscale="log", ylim=(1e-18, 1e-8), xlabel="random scenario (gap, V, gas, source, grid)",
          ylabel="relative residual", title="(d) Conservation and induced-current identity")  # fmt: skip
    a.legend(loc="upper left")
    check("max conservation residual (60 random scenarios)", float(cons.max()), 1e-10)
    check("max Ramo residual (60 random scenarios)", float(ramo.max()), 1e-10)

    fig.suptitle("Figure 1 · Analytic limits and bookkeeping", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_analytic_limits.png")


# ================================================================== figure 2
def figure2():
    print("fig2: transport")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    u = 2.0
    n0 = u / boag_pulsed_u(1.0, CH, GAS0)
    model = boag_model(n_cells=400)
    times = [0.1 * T_C, 0.25 * T_C, 0.5 * T_C]
    z, prof, st = step_profiles(model, n0, GAS0, times)
    zf = np.linspace(0, CH.gap_cm, 4001)
    colors = (S1, S2, S3)
    for species, a in ((0, ax[0, 0]), (1, ax[0, 1])):
        l1 = []
        for t, color in zip(times, colors, strict=True):
            exact = boag_profiles(zf, t, n0, GAS0.alpha_cm3_s)[species]
            sim = prof[t][species]
            a.plot(zf / CH.gap_cm, exact / n0, **THEORY)
            a.plot(z / CH.gap_cm, sim / n0, color=color, label=f"t = {t / T_C:.2f} $t_c$")
            ex_cells = boag_profiles(z, t, n0, GAS0.alpha_cm3_s)[species]
            l1.append(np.abs(sim - ex_cells).sum() / ex_cells.sum())
        a.plot([], [], **THEORY, label="exact (Boag model)")
        name = "positive" if species == 0 else "negative"
        a.set(xlabel="z / d  (positives drift to z = d)", ylabel=f"n{'⁺' if species == 0 else '⁻'} / n₀",
              ylim=(-0.03, 1.08), title=f"({'ab'[species]}) {name.capitalize()} ions after a pulse, u = 2")  # fmt: skip
        a.legend(loc="upper right" if species == 1 else "upper left")
        check(f"{name} ion profiles, max relative L1 error vs exact", max(l1), 0.02)

    # (c) scheme at a charge front: new limited upwind vs legacy Lax-Wendroff
    a = ax[1, 0]
    n_cells, k = 200, 100
    c = GAS.mu_pos_cm2_Vs / GAS.mu_neg_cm2_Vs  # positive-ion Courant number at the default dt
    lam = GAS.D_pos_cm2_s * (1e-3 / V_NEG) / 1e-3**2  # D dt / dz^2 for dz = 10 um
    new = np.ones(n_cells)
    flux = np.zeros(n_cells + 1)
    lw = np.ones(n_cells)
    wm, wp, w0 = lam + c * (c + 1) / 2, lam + c * (c - 1) / 2, 1 - c * c - 2 * lam
    for _ in range(k):
        _transport(new, c, lam, flux)
        padded = np.concatenate(([0.0], lw, [lw[-1]]))
        lw = wm * padded[:-2] + wp * padded[2:] + w0 * padded[1:-1]
    j = np.arange(n_cells) + 0.5
    front = c * k
    exact = np.array([0.5 * (1 + math.erf((x - front) / math.sqrt(4 * lam * k))) for x in j])
    sel = (j > front - 25) & (j < front + 25)
    a.plot(j[sel] - front, exact[sel], **THEORY, label="exact (drift + diffusion)")
    a.plot(j[sel] - front, new[sel], "o-", color=S1, ms=4, label="v4: flux-limited upwind")
    a.plot(j[sel] - front, lw[sel], "s-", color=S2, ms=4, label="v3: Lax–Wendroff")
    a.axhline(0, color=AXIS, lw=0.8)
    a.set(xlabel="cells from the exact front position", ylabel="n / n₀",
          title=f"(c) Drifting charge front after {k} steps (Courant {c:.2f})")  # fmt: skip
    a.legend(loc="upper left")
    check("v4 scheme: minimum density at a front (must be >= 0)", float(new.min()), 0.0, ".2e",
          below=False)  # fmt: skip
    CHECKS.append(
        {"name": "v3 Lax-Wendroff: minimum density at the same front (reference)",
         "value": f"{lw.min():.2e}", "tol": "(negative = unphysical)", "ok": True}
    )  # fmt: skip

    # (d) induced current vs exact
    a = ax[1, 1]
    for u_i, color in ((0.0, S1), (2.0, S2)):
        gas = GAS0.with_(alpha_cm3_s=0.0) if u_i == 0 else GAS0
        n0_i = (2.0 if u_i == 0 else u_i) / boag_pulsed_u(1.0, CH, GAS0)
        ep = model.run_episode(PiecewiseConstant.pulse(n0_i, 0, INSTANT), CH, gas)
        t_mid = 0.5 * (ep.record_edges_s[:-1] + ep.record_edges_s[1:])
        i0 = ELEMENTARY_CHARGE_C * n0_i * (V_POS + V_NEG)
        i_sim = ep.current_density_A_cm2 / i0
        i_exact = []
        for t in t_mid:
            n_p, n_n = boag_profiles(zf, t, n0_i, gas.alpha_cm3_s)
            i_exact.append(
                np.trapezoid(V_POS * n_p + V_NEG * n_n, zf) / CH.gap_cm * ELEMENTARY_CHARGE_C / i0
            )
        i_exact = np.array(i_exact)
        a.plot(
            t_mid / T_C,
            i_sim,
            color=color,
            lw=3.4,
            label=f"IonTracks, {'no recombination' if u_i == 0 else 'u = 2'}",
        )
        a.plot(t_mid / T_C, i_exact, **THEORY)  # thin exact curve on top of the wide simulation
        check(f"induced current vs exact, {'alpha = 0' if u_i == 0 else 'u = 2'}: max |dI| / I(0)",
              float(np.abs(i_sim - i_exact).max()), 0.01)  # fmt: skip
    a.plot([], [], **THEORY, label="exact (Boag model)")
    a.set(xlabel="t / $t_c$ (slow-ion transit time)", ylabel="I(t) / I(0)",
          title="(d) Induced (Ramo–Shockley) current after a pulse")  # fmt: skip
    a.legend(loc="upper right")

    fig.suptitle("Figure 2 · Transport against the exact solution of Boag's model", x=0.01,
                 ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "fig2_transport.png")


# ================================================================== figure 3
def figure3():
    print("fig3: time structure")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    model = boag_model(n_cells=200)
    u_unit = boag_pulsed_u(1.0, CH, GAS0)
    xi2_unit = boag_continuous_xi2(1.0, CH, GAS0)

    # (a) pulse width
    a = ax[0, 0]
    n0 = 1.0 / u_unit
    taus = np.logspace(-3, 2, 11) * T_C
    loss = np.array(
        [
            1 - solve_point(PiecewiseConstant.pulse(n0, 0, tau), CH, GAS0, model).f_recombination
            for tau in taus
        ]  # fmt: skip
    )
    tau_line = np.logspace(-3.2, 2.2, 200) * T_C
    xi2 = xi2_unit * n0 / tau_line
    a.axhline(1 - boag_pulsed_f(n0, CH, GAS0), **LIMIT, label="short-pulse limit: Boag pulsed")
    a.plot(tau_line / T_C, xi2 / (1 + xi2), **THEORY, label="long-pulse limit: Boag continuous")
    a.plot(taus / T_C, loss, "o", color=S1, label="IonTracks")
    a.set(xscale="log", yscale="log", ylim=(3e-3, 0.6), xlabel="pulse width τ / $t_c$",
          ylabel="recombination loss 1 − f", title="(a) One pulse, same charge, varying width")  # fmt: skip
    a.legend(loc="lower left")
    check("short pulse (1e-3 t_c): rel. deviation from Boag pulsed",
          abs(loss[0] / (1 - boag_pulsed_f(n0, CH, GAS0)) - 1), 0.01)  # fmt: skip
    xi2_last = xi2_unit * n0 / taus[-1]
    check("long pulse (100 t_c): rel. deviation from Boag continuous",
          abs(loss[-1] / (xi2_last / (1 + xi2_last)) - 1), 0.03)  # fmt: skip

    # (b) pulse period
    a = ax[0, 1]
    n0 = 0.5 / u_unit
    duration = 30 * T_C

    def width(period):
        return 1e-3 * min(T_C, period)

    periods = np.logspace(-2, 1, 10) * T_C
    loss = []
    for period in periods:
        n_pulses = max(10, math.ceil(duration / period))
        train = PiecewiseConstant.pulse_train(n0, width(period), period, n_pulses)
        loss.append(1 - solve_point(train, CH, GAS0, model).f_recombination)
    loss = np.array(loss)
    single = (
        1
        - solve_point(
            PiecewiseConstant.pulse(n0, 0, width(periods[-1])), CH, GAS0, model
        ).f_recombination
    )
    p_line = np.logspace(-2.1, 0.3, 14) * T_C
    cont = [
        1
        - solve_point(
            PiecewiseConstant.constant(n0 / p, 0, max(duration, 10 * p)), CH, GAS0, model
        ).f_recombination
        for p in p_line
    ]
    a.axhline(1 - boag_pulsed_f(n0, CH, GAS0), **LIMIT, label="independent pulses: Boag pulsed")
    a.plot(
        p_line / T_C, cont, color=S3, lw=1.2, label="continuous beam at the mean rate (IonTracks)"
    )
    a.plot(periods / T_C, loss, "o", color=S1, label="IonTracks pulse train (≥ 30 $t_c$)")
    mem = model.memory_time_s(CH, GAS0) / T_C
    a.axvline(mem, color=AXIS, lw=0.8)
    a.text(mem * 1.08, 0.3, "memory time:\nepisodes split", color=INK2, fontsize=7.5)
    a.set(xscale="log", yscale="log", xlabel="pulse period T / $t_c$",
          ylabel="recombination loss 1 − f", title="(b) Pulse train, same dose per pulse (u = 0.5)")  # fmt: skip
    a.legend(loc="lower left")
    check(
        "long period (10 t_c): rel. deviation from one simulated pulse",
        abs(loss[-1] / single - 1),
        1e-9,
    )
    cont_short = (
        1
        - solve_point(
            PiecewiseConstant.constant(n0 / periods[0], 0, duration), CH, GAS0, model
        ).f_recombination
    )
    check("short period (0.01 t_c): rel. deviation from continuous beam at the mean rate",
          abs(loss[0] / cont_short - 1), 0.03)  # fmt: skip

    # (c, d) build-up and drain of a continuous beam
    xi2_target = 0.05
    q = xi2_target / xi2_unit
    t_off = 4 * T_C
    src = PiecewiseConstant.constant(q, 0, t_off)
    full = boag_model(n_cells=400, steady_state_tolerance=0.0)
    ep = full.run_episode(src, CH, GAS0)
    ep0 = full.run_episode(src, CH, GAS0.with_(alpha_cm3_s=0.0))
    t = 0.5 * (ep.record_edges_s[:-1] + ep.record_edges_s[1:])
    i_sat = ELEMENTARY_CHARGE_C * q * CH.gap_cm

    def f_on(x_cm):
        x = np.minimum(x_cm, CH.gap_cm)
        return np.where(
            x > 0, np.where(x_cm < CH.gap_cm, x**2 / 2 + x * (CH.gap_cm - x), CH.gap_cm**2 / 2), 0.0
        )

    def i_exact(tt):  # no recombination, linear superposition of switch-on and switch-off
        on = (f_on(V_POS * tt) + f_on(V_NEG * tt)) / CH.gap_cm**2
        off = (f_on(V_POS * (tt - t_off)) + f_on(V_NEG * (tt - t_off))) / CH.gap_cm**2
        return on - np.where(tt > t_off, off, 0.0)

    t0 = 0.5 * (ep0.record_edges_s[:-1] + ep0.record_edges_s[1:])
    a = ax[1, 0]
    a.axvspan(0, t_off / T_C, color=SHADE, lw=0)
    a.plot(
        t0 / T_C,
        ep0.current_density_A_cm2 / i_sat,
        color=S1,
        lw=3.4,
        label="IonTracks, no recombination",
    )
    a.plot(
        t / T_C, ep.current_density_A_cm2 / i_sat, color=S2, label=f"IonTracks, ξ² = {xi2_target}"
    )
    a.plot(t0 / T_C, i_exact(t0), **THEORY, label="exact, no recombination")
    a.set(xlabel="t / $t_c$  (beam on in the shaded window)", ylabel="I(t) / I_sat",
          title="(c) Current build-up and drain, continuous beam")  # fmt: skip
    a.legend(loc="lower center")
    check("continuous switch-on/off current (alpha = 0) vs exact: max |dI| / I_sat",
          float(np.abs(ep0.current_density_A_cm2 / i_sat - i_exact(t0)).max()), 0.01)  # fmt: skip

    a = ax[1, 1]
    rec_rate = ep.recombined_per_step_cm2 / ep.dt_s / (q * CH.gap_cm)
    a.axvspan(0, t_off / T_C, color=SHADE, lw=0)
    a.axhline(xi2_target / (1 + xi2_target), **LIMIT, label="Boag steady state ξ²/(1+ξ²)")
    a.plot(t / T_C, rec_rate, color=S2, label="IonTracks")
    a.set(xlabel="t / $t_c$  (beam on in the shaded window)",
          ylabel="recombination rate / production rate (steady)",
          title="(d) Recombination rate: build-up over ~$t_c$, drain after beam-off")  # fmt: skip
    a.legend(loc="lower center")

    fig.suptitle("Figure 3 · Time structure between the analytic limits", x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "fig3_time_structure.png")


# ================================================================== figure 4
def figure4():
    print("fig4: steady-state skipping and convergence")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    xi2_unit = boag_continuous_xi2(1.0, CH, GAS)
    q = 0.02 / xi2_unit
    starts = np.array([0.0, 5.1e-3, 25.2e-3])
    durations = np.array([5e-3, 20e-3, 2e-3])
    src = PiecewiseConstant.from_intervals(
        starts, starts + durations, q * np.array([1.0, 3.0, 0.5])
    )
    fast = boag_model(n_cells=200).run_episode(src, CH, GAS)
    full = boag_model(n_cells=200, steady_state_tolerance=0.0).run_episode(src, CH, GAS)
    edges = np.linspace(0, full.n_steps * full.dt_s, 2001)
    to_current = ELEMENTARY_CHARGE_C / np.diff(edges) * 1e9  # nA/cm^2
    j_fast = np.diff(fast.cumulative("induced", edges)) * to_current
    j_full = np.diff(full.cumulative("induced", edges)) * to_current
    mid = 0.5 * (edges[1:] + edges[:-1]) * 1e3

    a = ax[0, 0]
    a.plot(mid, j_full, color=S1, lw=2.4, label=f"full stepping ({full.simulated_steps:,} steps)")
    a.plot(
        mid,
        j_fast,
        color=S2,
        lw=1.0,
        label=f"steady-state skipping ({fast.simulated_steps:,} steps)",
    )
    a.set(xlabel="time (ms)", ylabel="current density (nA/cm²)",
          title="(a) Three spots of different intensity: current")  # fmt: skip
    a.legend(loc="center")

    a = ax[0, 1]
    diff = np.abs(j_fast - j_full) / j_full.max()
    a.plot(mid, np.maximum(diff, 1e-17), color=S1)
    a.set(yscale="log", ylim=(1e-17, 1e-8), xlabel="time (ms)",
          ylabel="|I_skip − I_full| / I_max", title="(b) Difference between the two runs")  # fmt: skip
    check("steady-state skipping vs full stepping: max |dI| / I_max", float(diff.max()), 1e-8)
    check("steady-state skipping vs full stepping: rel. difference in k_s - 1",
          abs((fast.ks - 1) / (full.ks - 1) - 1), 1e-9)  # fmt: skip

    a = ax[1, 0]
    durations = np.logspace(0, 3, 7) * T_C
    sim_fast, sim_full, wall_fast, wall_full = [], [], [], []
    for dur in durations:
        s = PiecewiseConstant.constant(q, 0, dur)
        for tol, steps, wall in ((1e-12, sim_fast, wall_fast), (0.0, sim_full, wall_full)):
            m = boag_model(n_cells=200, steady_state_tolerance=tol)
            t0 = time.perf_counter()
            steps.append(m.run_episode(s, CH, GAS).simulated_steps)
            wall.append(time.perf_counter() - t0)
    a.plot(durations / T_C, sim_full, "o-", color=S1, label="full stepping")
    a.plot(durations / T_C, sim_fast, "o-", color=S2, label="steady-state skipping")
    a.annotate(f"{wall_full[-1]:.2f} s", (durations[-1] / T_C, sim_full[-1]), textcoords="offset points",
               xytext=(-10, 6), ha="right", color=INK2, fontsize=8)  # fmt: skip
    a.annotate(f"{wall_fast[-1] * 1e3:.1f} ms", (durations[-1] / T_C, sim_fast[-1]),
               textcoords="offset points", xytext=(-10, 6), ha="right", color=INK2, fontsize=8)  # fmt: skip
    a.set(xscale="log", yscale="log", xlabel="duration of a constant-rate segment / $t_c$",
          ylabel="time steps computed", title="(c) Cost of a long constant segment (200 cells)")  # fmt: skip
    a.legend(loc="upper left")

    a = ax[1, 1]
    pts = CH.sample_points(4, 8)
    cells = [50, 100, 200, 400, 800]
    for current, color in ((2.0, S1), (10.0, S2)):
        field = cyclotron_pbs_field(beam_current_nA=current)
        ks = [solve_field(field, CH, GAS, boag_model(n_cells=n), points=pts).ks for n in cells]
        dev = np.abs((np.array(ks[:-1]) - 1) / (ks[-1] - 1) - 1)
        a.plot(cells[:-1], dev, "o-", color=color, label=f"synthetic field, {current:g} nA")
        if current == 2.0:
            check("synthetic field (2 nA): |k_s-1| change 200 -> 800 cells", float(dev[2]), 1e-3)
    a.set(xscale="log", yscale="log", xlabel="cells across the gap",
          ylabel="relative deviation of k_s − 1 from 800 cells", title="(d) Grid convergence of a whole field")  # fmt: skip
    a.legend(loc="upper right")

    fig.suptitle("Figure 4 · Steady-state skipping and grid convergence", x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "fig4_skipping_convergence.png")


# ================================================================== figure 5
def figure5():
    print("fig5: synthetic 250 MeV cyclotron field, 2 mm / 200 V")
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    model = boag_model(n_cells=200)
    pts = CH.sample_points(4, 8)
    currents = np.array([1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 800.0])
    ks, ks_qs, eta, dwell = [], [], [], []
    for current in currents:
        field = cyclotron_pbs_field(beam_current_nA=current)
        res = solve_field(field, CH, GAS, model, points=pts)
        p = r = peak = 0.0
        for x, y, w in zip(pts.x_cm, pts.y_cm, pts.weight_cm2, strict=True):
            s = field.local_source(x, y, GAS)
            pp, rr = boag_quasi_static_loss(s, CH, GAS)
            p, r, peak = p + w * pp, r + w * rr, max(peak, s.max_rate)
        ks.append(res.ks)
        ks_qs.append(p / (p - r))
        eta.append(space_charge_parameter(peak, CH, GAS))
        dwell.append(field.t_off_s[0] - field.t_on_s[0])
        if current in (1.0, 2.0):
            check(f"synthetic field {current:g} nA: rel. deviation of k_s-1 from quasi-static Boag",
                  abs((ks[-1] - 1) / (ks_qs[-1] - 1) - 1), 0.01)  # fmt: skip
            check(f"synthetic field {current:g} nA: field-level conservation residual",
                  abs(res.produced - res.recombined - res.signal) / res.produced, 1e-10)  # fmt: skip
    ks, ks_qs, eta, dwell = map(np.array, (ks, ks_qs, eta, dwell))
    invalid_from = currents[np.argmax(eta > 1)]

    valid = eta <= 1
    hollow = {"mfc": SURFACE, "mec": S1, "mew": 1.5}

    a = ax[0, 0]
    a.axvspan(invalid_from / 1.4, currents[-1] * 1.4, color=SHADE, lw=0)
    a.text(
        invalid_from / 1.25,
        2e-3,
        "η > 1: space charge matters\n(classic model here; see Fig. 8)",
        color=INK2,
        fontsize=8,
    )
    a.plot(currents, ks_qs - 1, **THEORY, label="quasi-static Boag (per constant segment)")
    a.plot(currents[valid], ks[valid] - 1, "o", color=S1, label="IonTracks (time-resolved)")
    a.plot(
        currents[~valid],
        ks[~valid] - 1,
        "o",
        **hollow,
        label="IonTracks, η > 1 (no space charge here)",
    )
    a.set(xscale="log", yscale="log", xlim=(0.7, 1150), xlabel="nozzle current (nA)",
          ylabel="k_s − 1", title="(a) Recombination vs beam current")  # fmt: skip
    a.legend(loc="upper left")

    a = ax[0, 1]
    ratio = (ks - 1) / (ks_qs - 1)
    a.axhline(1.0, **LIMIT, label="quasi-static Boag")
    a.plot(dwell / T_C, ratio, color=S1, lw=1.0)
    a.plot(dwell[valid] / T_C, ratio[valid], "o", color=S1, label="IonTracks")
    a.plot(dwell[~valid] / T_C, ratio[~valid], "o", **hollow, label="η > 1 (no space charge here)")
    for d, c_, e in zip(dwell / T_C, currents, eta, strict=True):
        a.annotate(f"{c_:g} nA", (d, 0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                   textcoords="offset points", ha="center", fontsize=7, color=INK2 if e <= 1 else MUTED)  # fmt: skip
    a.set(xscale="log", xlim=(0.6, 1600), xlabel="spot dwell / $t_c$", ylabel="(k_s − 1) / (k_s,QS − 1)",
          title="(b) Transients when spots approach $t_c$")  # fmt: skip
    a.legend(loc="center right")

    # (c) k_s map at 10 nA
    field10 = cyclotron_pbs_field(beam_current_nA=10.0)
    fine = CH.sample_points(8, 16)
    res10 = solve_field(field10, CH, GAS, model, points=fine)
    x, y, k = res10.ks_map()
    a = ax[0, 2]
    cmap = plt.get_cmap(SEQUENTIAL)
    sc = a.scatter(
        x * 10, y * 10, c=(k - 1) * 100, cmap=cmap, s=38, edgecolors=SURFACE, linewidths=0.8
    )
    a.add_patch(plt.Circle((0, 0), CH.sensitive_radius_cm * 10, fill=False, color=AXIS, lw=0.8))
    a.set(
        aspect="equal",
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="(c) k_s − 1 over the sensitive area, 10 nA",
    )
    a.grid(False)
    cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("k_s − 1 (%)", color=INK2)
    cb.outline.set_edgecolor(AXIS)

    # (d, e) local dose rate at the centre and chamber current, 10 nA
    s_c = field10.local_source(0.0, 0.0, GAS)
    dose_rate = (
        s_c.rates
        / GAS.ion_pairs_per_cm3_per_Gy()
        * (PROTON_250MEV_S_WATER_MEV_CM2_G / PROTON_250MEV_S_AIR_MEV_CM2_G)
    )
    t_peak = s_c.edges[np.argmax(s_c.rates)]
    window = (t_peak - 0.06, t_peak + 0.06)
    a = ax[1, 0]
    a.stairs(dose_rate, s_c.edges, color=S1, lw=1.4)
    a.set(xlim=window, xlabel="time (s)", ylabel="dose rate to water (Gy/s)",
          title="(d) Local dose rate at the chamber centre, 10 nA")  # fmt: skip

    t_edges = np.linspace(*window, 1201)
    res10_0 = solve_field(field10, CH, GAS.with_(alpha_cm3_s=0.0), model, points=pts)
    res10_c = solve_field(field10, CH, GAS, model, points=pts)
    a = ax[1, 1]
    a.plot(0.5 * (t_edges[1:] + t_edges[:-1]), res10_0.current_A(t_edges) * 1e9, color=S2, lw=1.2,
           label="no recombination (saturation)")  # fmt: skip
    a.plot(0.5 * (t_edges[1:] + t_edges[:-1]), res10_c.current_A(t_edges) * 1e9, color=S1, lw=1.2,
           label="IonTracks")  # fmt: skip
    a.set(xlim=window, xlabel="time (s)", ylabel="chamber current (nA)",
          title="(e) Chamber current, 10 nA (same window)")  # fmt: skip
    a.set_ylim(top=a.get_ylim()[1] * 1.3)
    a.legend(loc="upper left")

    # (f) lateral quadrature convergence
    a = ax[1, 2]
    grids = [(1, 8), (2, 8), (4, 8), (6, 12), (8, 16), (10, 20)]
    n_pts = [r * az for r, az in grids]
    for current, color in ((2.0, S1), (10.0, S2)):
        field = cyclotron_pbs_field(beam_current_nA=current)
        vals = [
            solve_field(field, CH, GAS, model, points=CH.sample_points(r, az)).ks - 1
            for r, az in grids
        ]
        a.plot(n_pts, np.array(vals) / vals[-1], "o-", color=color, label=f"{current:g} nA")
        if current == 2.0:
            check("synthetic field (2 nA): |k_s-1| change 32 -> 200 sample points",
                  abs(vals[2] / vals[-1] - 1), 0.01)  # fmt: skip
    a.axhline(1.0, **LIMIT)
    a.set(xscale="log", xlabel="sample points over the sensitive area",
          ylabel="(k_s − 1) / (k_s − 1 at 200 points)", title="(f) Lateral quadrature convergence")  # fmt: skip
    a.legend(loc="lower right")

    fig.suptitle("Figure 5 · Synthetic 250 MeV cyclotron PBS field, 2 mm gap, 200 V (2 Gy to water)",
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    save(fig, "fig5_cyclotron_field.png")


# ================================================================== report
# ================================================================== figure 8
def figure8():
    print("fig8: free electrons and space charge")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    gas = Gas.air(particle="electron")
    fe_only = {"free_electrons": True, "space_charge": False}

    # (a) free-electron fraction: exact formula vs simulated pulses without recombination
    a = ax[0, 0]
    fields = np.logspace(2, 4, 40)
    worst = 0.0
    for gap, color in zip((0.05, 0.1, 0.2), (S1, S2, S3), strict=True):
        a.plot(fields, [gas.free_electron_fraction(gap, e) for e in fields], color=color,
               label=f"(λ/d)(1 − exp(−d/λ)), d = {gap * 10:g} mm")  # fmt: skip
        for e in (200.0, 1000.0, 5000.0):
            ch = ParallelPlateChamber(gap, e * gap)
            ep = Uniform1D(Grid1D(n_cells=100), **fe_only).run_episode(
                PiecewiseConstant.pulse(1e9, 0, INSTANT), ch, gas.with_(alpha_cm3_s=0.0)
            )
            exact = gas.free_electron_fraction(gap, e)
            worst = max(worst, abs(ep.free_electron_fraction / exact - 1))
            a.plot(e, ep.free_electron_fraction, "o", color=color)
    a.plot([], [], "o", color=INK2, label="IonTracks, no recombination")
    a.set(xscale="log", xlabel="field (V/cm)", ylabel="free-electron fraction p",
          title="(a) Electrons collected before attaching")  # fmt: skip
    a.legend(loc="upper left")
    check("free-electron fraction vs exact, max rel. deviation", worst, 1e-10)

    # (b) pulsed efficiency: Boag, Fenwick & Kumar (exact with free electrons), full model
    a = ax[0, 1]
    ch = ParallelPlateChamber(0.1, 300.0)
    g0 = gas.with_(D_pos_cm2_s=0.0, D_neg_cm2_s=0.0)
    per_gy = gas.ion_pairs_per_cm3_per_Gy()
    dpp = np.logspace(-3, 0, 10)  # Gy per pulse (in air)
    line = np.logspace(-3.1, 0.1, 200)
    a.plot(line, [boag_pulsed_f(n * per_gy, ch, g0) for n in line], **LIMIT,
           label="Boag (no free electrons)")  # fmt: skip
    a.plot(line, [boag_pulsed_free_electron_f(n * per_gy, ch, g0) for n in line], **THEORY,
           label="Fenwick & Kumar, exact with free electrons")  # fmt: skip
    fe, full, rel = [], [], []
    for dose in dpp:
        src = PiecewiseConstant.pulse(dose * per_gy, 0, INSTANT)
        e1 = Uniform1D(Grid1D(n_cells=400), **fe_only).run_episode(src, ch, g0)
        e2 = Uniform1D(Grid1D(n_cells=400)).run_episode(src, ch, gas)
        fe.append(e1.collection_efficiency)
        full.append(e2.collection_efficiency)
        exact = boag_pulsed_free_electron_f(dose * per_gy, ch, g0)
        rel.append(abs((1 - e1.collection_efficiency) / (1 - exact) - 1))
    a.plot(dpp, fe, "o", color=S1, label="IonTracks: free electrons only")
    a.plot(dpp, full, "s", color=S2, ms=5, label="IonTracks: + space charge + diffusion")
    a.set(xscale="log", xlabel="dose per pulse in air (Gy)", ylabel="collection efficiency",
          title="(b) Instantaneous pulse, 1 mm, 300 V")  # fmt: skip
    a.legend(loc="lower left")
    check("free electrons vs Fenwick-Kumar exact, max rel. error in (1-f), 1 mGy-1 Gy",
          max(rel), 1e-3)  # fmt: skip

    # (c) the space-charge effect against the pulsed space-charge parameter
    a = ax[1, 0]
    slopes = []
    for (gap, volts), color in zip(((0.1, 100.0), (0.2, 200.0), (0.2, 50.0)), (S1, S2, S3),
                                   strict=True):  # fmt: skip
        ch_c = ParallelPlateChamber(gap, volts)
        effect, param = [], []
        for n0 in np.logspace(7, 11, 9):
            src = PiecewiseConstant.pulse(n0, 0, INSTANT)
            f_sc = Uniform1D(Grid1D(n_cells=200)).run_episode(src, ch_c, gas).f_recombination
            f_no = (
                Uniform1D(Grid1D(n_cells=200), space_charge=False)
                .run_episode(src, ch_c, gas)
                .f_recombination
            )
            effect.append((1 - f_sc) / (1 - f_no) - 1)
            param.append(space_charge_parameter_pulsed(n0, ch_c))
        a.plot(param, np.abs(effect), "o-", color=color, label=f"d = {gap * 10:g} mm, {volts:g} V")
        slopes.append(np.polyfit(np.log(param[:3]), np.log(np.abs(effect[:3])), 1)[0])
    a.set(xscale="log", yscale="log", xlabel="e n₀ d² / (ε₀ V)",
          ylabel="relative increase of the loss 1 − f",
          title="(c) Space-charge onset (instantaneous pulse)")  # fmt: skip
    a.legend(loc="upper left")
    check("space-charge effect at low density: |log-log slope - 1|",
          float(np.max(np.abs(np.array(slopes) - 1))), 0.1, ".3f")  # fmt: skip

    # (d) field screening after a dense pulse
    a = ax[1, 1]
    ch_d = ParallelPlateChamber(0.2, 200.0)
    tc = ch_d.gap_cm / (gas.mu_pos_cm2_Vs * ch_d.field_V_cm)
    times = np.array([0.02, 0.1, 0.25, 0.5, 0.8]) * tc
    ep = Uniform1D(Grid1D(n_cells=200)).run_episode(
        PiecewiseConstant.pulse(0.05 * per_gy, 0, INSTANT), ch_d, gas, snapshot_times_s=times
    )
    zf = np.linspace(0, ch_d.gap_cm, 201)
    cmap = plt.get_cmap(SEQUENTIAL)
    for k, t in enumerate(ep.snapshots.times_s):
        a.plot(zf * 10, ep.snapshots.field_V_cm[k] / ch_d.field_V_cm, color=cmap(k / 4),
               label=f"t = {t / tc:.2f} t_c")  # fmt: skip
    a.axhline(1.0, **LIMIT)
    a.set(xlabel="z (mm), anode at 0", ylabel="E / (V/d)",
          title="(d) Field after a 50 mGy pulse (2 mm, 200 V)")  # fmt: skip
    a.legend(loc="upper center")
    fig.tight_layout()
    save(fig, "fig8_free_electrons_space_charge.png")


def write_report(runtime_s: float) -> None:
    n_ok = sum(c["ok"] for c in CHECKS)
    rows = "\n".join(
        f"| {'✅' if c['ok'] else '❌'} | {c['name'].replace('|', chr(92) + '|')} | {c['value']} | {c['tol']} |"
        for c in CHECKS
    )
    (OUT / "VALIDATION.md").write_text(
        f"""# IonTracks v4 validation

Generated by `python validation/run_validation.py` ({runtime_s:.0f} s). **{n_ok} of {len(CHECKS)} checks
pass.** Chamber for all runs unless stated: plane-parallel, 2 mm gap, 200 V, dry air with separate
ion mobilities (µ⁺ = 1.36, µ⁻ = 2.1 cm²/Vs). Comparisons against Boag's theories switch diffusion
off, because those theories neglect it.

| | Check | Value | Criterion |
|---|---|---|---|
{rows}

## Figure 1 · Analytic limits and bookkeeping

![Figure 1](figures/fig1_analytic_limits.png)

(a) Instantaneous uniform pulse against Boag's exact ln(1+u)/u over four decades of u.
(b) The error falls faster than first order as the grid is refined.
(c) Steady-state loss of a continuous beam divided by ξ². It tends to 1 (first-order Boag) for
small ξ². For large ξ² it falls below Boag's 1/(1+ξ²) formula, which is itself only first-order
accurate.
(d) Charge conservation and the Ramo identity (∫I dt equals the collected charge) hold to round-off
for 60 random chambers, gases, grids and sources.

## Figure 2 · Transport against the exact solution of Boag's model

![Figure 2](figures/fig2_transport.png)

(a, b) After an instantaneous pulse with drift and recombination (no diffusion), the model has a
closed-form solution: inside the overlap region n = n₀/(1+αn₀t), and outside it the density is
frozen at the moment the ion left the overlap. The simulated profiles follow it. The negative
ions (Courant 1) shift exactly; the positive ions show slight smoothing at the front.
(c) A drifting charge front: the v4 flux-limited scheme stays within [0, 1], while the v3
Lax–Wendroff scheme under- and overshoots.
(d) The induced current agrees with the current computed from the exact profiles, with and without
recombination.

## Figure 3 · Time structure between the analytic limits

![Figure 3](figures/fig3_time_structure.png)

(a) One pulse of fixed charge, stretched from 10⁻³ to 10² collection times. The solver moves
continuously from Boag's pulsed limit to Boag's continuous limit.
(b) A pulse train with fixed dose per pulse (u = 0.5). For long periods it gives exactly the
single-pulse value, because episodes are split and deduplicated at the memory time. For short
periods it approaches a continuous beam at the same mean rate. That reference is computed by the
solver, because at these mean rates ξ² ≫ 1 and Boag's first-order continuous formula no longer
applies. The overlap regime in between is where neither Boag formula applies (Brualla-González
et al. 2016).
(c) A continuous beam switched on and off. With recombination off, the current matches the exact
build-up and drain.
(d) The recombination rate builds up over about one collection time to Boag's steady state, and
recombination continues while the gap drains.

## Figure 4 · Steady-state skipping and convergence

![Figure 4](figures/fig4_skipping_convergence.png)

(a, b) Three spots of different intensity. Skipping steady-state stretches gives the same current
as stepping every step, to round-off.
(c) Cost: without skipping it grows linearly with segment length; with skipping it is constant.
(d) Grid convergence of k_s for the whole synthetic field.

## Figure 5 · Synthetic 250 MeV cyclotron PBS field

![Figure 5](figures/fig5_cyclotron_field.png)

(a) k_s − 1 against nozzle current. At clinical currents it matches Boag's continuous theory
applied to every constant segment. Above about 30 nA the peak local dose rate makes η > 1, so space
charge matters. Those points (hollow, shaded) use the classic model here; Figure 8 covers
space charge.
(b) The ratio to the quasi-static estimate as a function of spot dwell time: when spots last only a
few collection times, build-up and drain reduce recombination.
(c) k_s over the sensitive area. The pattern follows the 5 mm spot grid: points under a spot row
see higher instantaneous dose rates.
(d, e) The dose-rate history at the centre and the chamber current while spots pass. The gap
between the saturation current and the simulated current is the recombination loss.
(f) Convergence with the number of lateral sample points.

## Figure 8 · Free electrons and space charge

![Figure 8](figures/fig8_free_electrons_space_charge.png)

(a) Electrons that reach the anode before attaching (attachment data of Götz et al. 2013, as
tabulated by Fenwick & Kumar 2023). The attachment kernel reproduces (λ/d)(1 − exp(−d/λ)) to
round-off.
(b) Collection efficiency of an instantaneous pulse. With free electrons, a uniform field and no
diffusion, the model reproduces the exact solution of Fenwick & Kumar (2023). Adding space charge
and diffusion (the default model) lowers the efficiency at high dose per pulse.
(c) The relative space-charge increase of the loss is proportional to e n₀ d² / (ε₀ V), which is
`space_charge_parameter_pulsed`.
(d) The separating ion clouds screen the field in the middle of the gap. The field grows near the
electrodes, while the mean field stays V/d.

Figures 1–5 compare with Boag's classic theories and therefore use the model without free
electrons or space charge. The track model (initial recombination, overlapping tracks) is
validated in [VALIDATION_TILE3D.md](VALIDATION_TILE3D.md). For the blind comparisons with
measurements, see
[BENCHMARK_DCPT2019.md](BENCHMARK_DCPT2019.md) (scanned protons) and
[BENCHMARK_RISOE2026.md](BENCHMARK_RISOE2026.md) (pulsed electrons and photons).
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    t_start = time.perf_counter()
    for make in (figure1, figure2, figure3, figure4, figure5, figure8):
        make()
    runtime = time.perf_counter() - t_start
    write_report(runtime)
    print(f"\n{sum(c['ok'] for c in CHECKS)}/{len(CHECKS)} checks pass; {runtime:.0f} s")
    print(f"figures: {FIG}\nsummary: {OUT / 'VALIDATION.md'}")
    if not all(c["ok"] for c in CHECKS):
        raise SystemExit(1)
