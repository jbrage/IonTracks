"""Numerics of the 1D point model: kernels, conservation, and Boag's analytic limits."""

import numpy as np
import pytest

from iontracks import Grid1D, ParallelPlateChamber, PiecewiseConstant, Uniform1D
from iontracks.analytic import boag_continuous_xi2, boag_pulsed_f, boag_pulsed_u
from iontracks.solvers.uniform1d import _react, _transport, source_runs

INSTANT = 1e-12  # s; "instantaneous" pulse width, far below any time step


# ---------------------------------------------------------------- kernels
@pytest.mark.parametrize(
    "a0, c0, x",
    [(1.0, 0.7, 1e-8), (1.0, 1.0, 0.5), (1.0, 0.7, 0.3), (2.0, 1.0 + 1e-9, 1.0), (1.0, 3.0, 50.0)],
)
def test_reaction_matches_closed_form(a0, c0, x):
    a, c = np.array([a0]), np.array([c0])
    lost = _react(a, c, x)
    d = a0 - c0
    expected = a0 / (1 + a0 * x) if d == 0 else d / (1 - (c0 / a0) * np.exp(-d * x))
    assert a[0] == pytest.approx(expected, rel=1e-9, abs=1e-300)
    assert c[0] == pytest.approx(a[0] - d, rel=1e-9, abs=1e-12)
    assert lost == pytest.approx(a0 - a[0], rel=1e-12)


def test_reaction_stays_positive_for_huge_steps():
    a, c = np.array([1e15, 3e14, 0.0]), np.array([2e15, 3e14, 5.0])
    _react(a, c, 1e3)
    assert np.all(a >= 0) and np.all(c >= 0)
    assert a[0] == 0.0 and c[0] == pytest.approx(1e15)


def test_drift_at_courant_one_is_an_exact_shift():
    n = np.zeros(10)
    n[2:5] = [1.0, 2.0, 3.0]
    flux = np.zeros(11)
    outflow, _ = _transport(n, 1.0, 0.0, flux)
    np.testing.assert_array_equal(n[3:6], [1.0, 2.0, 3.0])
    assert n[2] == 0.0 and outflow == 0.0


@pytest.mark.parametrize("courant, lam", [(0.65, 0.0), (0.65, 0.3), (1.0, 0.3), (0.2, 0.1)])
def test_transport_is_positivity_preserving(courant, lam):
    rng = np.random.default_rng(3)
    n = rng.uniform(0, 1, 200) * (rng.uniform(0, 1, 200) > 0.5)  # rough, with sharp steps
    total = n.sum()
    flux = np.zeros(201)
    out = 0.0
    for _ in range(150):
        o, _ = _transport(n, courant, lam, flux)
        out += o
        assert n.min() >= -1e-15
    assert n.sum() + out == pytest.approx(total, rel=1e-12)


# ---------------------------------------------------------------- conservation and signal
@pytest.mark.parametrize(
    "source",
    [
        PiecewiseConstant.pulse(5e10, 0.0, INSTANT),
        PiecewiseConstant.constant(2e14, 0.0, 3e-4),
        PiecewiseConstant.pulse_train(3e9, 5e-6, 2e-5, 20),
    ],
    ids=["pulse", "continuous", "overlapping-train"],
)
def test_charge_conservation_and_ramo_identity(chamber, gas, source):
    ep = Uniform1D(Grid1D(n_cells=150)).run_episode(source, chamber, gas)
    assert ep.emptied
    assert ep.produced_cm2 == pytest.approx(source.total * chamber.gap_cm, rel=1e-12)
    # every ion pair is either recombined or collected (both species)
    assert ep.recombined_cm2 + ep.collected_pos_cm2 == pytest.approx(ep.produced_cm2, rel=1e-11)
    assert ep.collected_neg_cm2 == pytest.approx(ep.collected_pos_cm2, rel=1e-11)
    # time integral of the induced current = collected charge
    assert ep.signal_cm2 == pytest.approx(ep.collected_pos_cm2, rel=1e-10)
    per_record = ep.record_steps * ep.recombined_per_step_cm2
    assert per_record.sum() == pytest.approx(ep.recombined_cm2, rel=1e-12)
    assert ep.cumulative("source", 1e9) == pytest.approx(ep.produced_cm2, rel=1e-12)


def test_voltage_polarity_only_relabels_the_electrodes(chamber, gas):
    src = PiecewiseConstant.pulse_train(3e9, 5e-6, 2e-5, 20)
    reversed_chamber = ParallelPlateChamber(chamber.gap_cm, -chamber.voltage_V)
    model = Uniform1D(Grid1D(n_cells=100))
    a = model.run_episode(src, chamber, gas)
    b = model.run_episode(src, reversed_chamber, gas)
    assert a.recombined_cm2 == b.recombined_cm2 and a.signal_cm2 == b.signal_cm2


def test_no_recombination_without_alpha(chamber, gas):
    ep = Uniform1D().run_episode(
        PiecewiseConstant.pulse(1e12, 0.0, INSTANT), chamber, gas.with_(alpha_cm3_s=0.0)
    )
    assert ep.recombined_cm2 == 0.0
    assert ep.ks == 1.0
    assert ep.collection_efficiency == pytest.approx(1.0, rel=1e-11)


# ---------------------------------------------------------------- analytic limits
@pytest.mark.parametrize("u", [0.1, 1.0, 5.0])
def test_boag_pulsed(chamber, gas_no_diffusion, u):
    n0 = u / boag_pulsed_u(1.0, chamber, gas_no_diffusion)
    ep = Uniform1D(Grid1D(n_cells=400)).run_episode(
        PiecewiseConstant.pulse(n0, 0.0, INSTANT), chamber, gas_no_diffusion
    )
    f_boag = boag_pulsed_f(n0, chamber, gas_no_diffusion)
    assert (1 - ep.f_recombination) == pytest.approx(1 - f_boag, rel=1e-4)


def test_boag_pulsed_converges_with_grid(chamber, gas_no_diffusion):
    n0 = 1.0 / boag_pulsed_u(1.0, chamber, gas_no_diffusion)
    f_boag = boag_pulsed_f(n0, chamber, gas_no_diffusion)
    errors = []
    for n_cells in (50, 100, 200, 400):
        ep = Uniform1D(Grid1D(n_cells=n_cells)).run_episode(
            PiecewiseConstant.pulse(n0, 0.0, INSTANT), chamber, gas_no_diffusion
        )
        errors.append(abs(ep.f_recombination - f_boag))
    assert all(e2 < 0.6 * e1 for e1, e2 in zip(errors, errors[1:], strict=False))


def test_pulsed_result_depends_only_on_sum_of_mobilities(chamber, gas_no_diffusion):
    """Boag pulsed depends on mu+ + mu-: swapping the species' mobilities changes nothing."""
    swapped = gas_no_diffusion.with_(
        mu_pos_cm2_Vs=gas_no_diffusion.mu_neg_cm2_Vs, mu_neg_cm2_Vs=gas_no_diffusion.mu_pos_cm2_Vs
    )
    src = PiecewiseConstant.pulse(2e10, 0.0, INSTANT)
    model = Uniform1D(Grid1D(n_cells=300))
    f1 = model.run_episode(src, chamber, gas_no_diffusion).f_recombination
    f2 = model.run_episode(src, chamber, swapped).f_recombination
    assert f1 == pytest.approx(f2, rel=1e-5)


@pytest.mark.parametrize("xi2", [1e-3, 1e-2])
def test_boag_continuous_steady_state(chamber, gas_no_diffusion, xi2):
    """Steady-state loss equals xi^2 up to the second-order correction of the exact theory."""
    q = xi2 / boag_continuous_xi2(1.0, chamber, gas_no_diffusion)
    t_c = max(chamber.collection_times_s(gas_no_diffusion))
    duration = 8 * t_c
    ep = Uniform1D(Grid1D(n_cells=200)).run_episode(
        PiecewiseConstant.constant(q, 0.0, duration), chamber, gas_no_diffusion
    )
    t0, t1 = 3 * t_c, duration - ep.dt_s
    loss = (
        np.diff(ep.cumulative("recombined", [t0, t1]))[0]
        / np.diff(ep.cumulative("source", [t0, t1]))[0]
    )
    assert abs(loss / xi2 - 1) < 2.5 * xi2


def test_short_pulse_approaches_instantaneous_limit(chamber, gas_no_diffusion):
    t_c = max(chamber.collection_times_s(gas_no_diffusion))
    n0 = 1.0 / boag_pulsed_u(1.0, chamber, gas_no_diffusion)
    model = Uniform1D(Grid1D(n_cells=400))
    f_short = model.run_episode(
        PiecewiseConstant.pulse(n0, 0.0, 1e-3 * t_c), chamber, gas_no_diffusion
    )
    f_long = model.run_episode(
        PiecewiseConstant.pulse(n0, 0.0, 0.5 * t_c), chamber, gas_no_diffusion
    )
    f_boag = boag_pulsed_f(n0, chamber, gas_no_diffusion)
    assert f_short.f_recombination == pytest.approx(f_boag, rel=2e-3)
    assert f_long.f_recombination > f_short.f_recombination + 0.01  # spreading the dose helps


def test_diffusion_is_a_small_correction(chamber, gas, gas_no_diffusion):
    src = PiecewiseConstant.pulse(2e10, 0.0, INSTANT)
    model = Uniform1D(Grid1D(n_cells=300))
    with_d = model.run_episode(src, chamber, gas).f_recombination
    without = model.run_episode(src, chamber, gas_no_diffusion).f_recombination
    assert with_d == pytest.approx(without, rel=1e-3)


# ---------------------------------------------------------------- steady-state skipping
def test_source_runs_are_exact_and_compact():
    src = PiecewiseConstant.from_intervals([0.3e-6, 5e-3], [4.1e-3, 9.7e-3], [2e12, 5e11])
    lengths, amount = source_runs(src, 2.4e-7)
    assert np.dot(lengths, amount) == pytest.approx(src.total, rel=1e-12)
    assert lengths.size <= 3 * src.edges.size  # memory ~ breakpoints, not ~ time steps
    assert lengths.sum() >= 9.7e-3 / 2.4e-7


@pytest.mark.parametrize(
    "source",
    [
        PiecewiseConstant.constant(3e13, 0.0, 20e-3),  # long constant beam, ~100 memory times
        PiecewiseConstant.from_intervals([0.0, 5e-3], [8e-3, 9e-3], [1e13, 4e13]),  # rate change
    ],
    ids=["constant", "step-change"],
)
def test_steady_state_skipping_matches_full_stepping(chamber, gas, source):
    fast = Uniform1D(Grid1D(n_cells=120)).run_episode(source, chamber, gas)
    full = Uniform1D(Grid1D(n_cells=120, steady_state_tolerance=0.0)).run_episode(
        source, chamber, gas
    )
    assert full.simulated_steps == full.n_steps
    assert fast.simulated_steps < 0.3 * full.simulated_steps
    assert fast.n_steps == full.n_steps
    assert (fast.ks - 1) == pytest.approx(full.ks - 1, rel=1e-9)
    assert fast.signal_cm2 == pytest.approx(full.signal_cm2, rel=1e-10)
    edges = np.linspace(0, full.n_steps * full.dt_s, 57)
    np.testing.assert_allclose(
        np.diff(fast.cumulative("induced", edges)),
        np.diff(full.cumulative("induced", edges)),
        rtol=1e-8,
    )
