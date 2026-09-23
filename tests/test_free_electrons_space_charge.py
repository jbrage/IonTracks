"""Free electrons (exact attachment kernel, Fenwick-Kumar oracle) and space charge (Poisson)."""

import numpy as np
import pytest

from iontracks import Grid1D, ParallelPlateChamber, PiecewiseConstant, Uniform1D
from iontracks.analytic import boag_pulsed_f, boag_pulsed_free_electron_f, boag_pulsed_u
from iontracks.solvers.kernels import attach, solve_field
from iontracks.solvers.uniform1d import FIELD_COEF_V_CM

INSTANT = 1e-12
FE_ONLY = {"free_electrons": True, "space_charge": False}


def _constant_attachment(gas, lam_cm, field_V_cm):
    """Gas whose attachment length is lam_cm at the given field (field-independent rate)."""
    return gas.with_(attachment_rate_s=(gas.electron_mobility_cm2_Vs * field_V_cm / lam_cm, 0, 0))


# ---------------------------------------------------------------- free electrons
def test_attachment_kernel_is_exact_for_uniform_production():
    n = np.zeros(50)
    prod = np.ones(50)
    ell = np.full(50, 7.0)  # cells
    collected, _ = attach(n, prod, ell)
    L = 50 / 7.0
    assert collected / 50 == pytest.approx((1 - np.exp(-L)) / L, rel=1e-13)
    assert n.sum() + collected == pytest.approx(50.0, rel=1e-14)


def test_free_electron_fraction_of_a_pulse(chamber, gas):
    """Without recombination the directly collected electrons are exactly (lam/d)(1 - e^-d/lam)."""
    ep = Uniform1D(Grid1D(n_cells=100), **FE_ONLY).run_episode(
        PiecewiseConstant.pulse(1e9, 0.0, INSTANT), chamber, gas.with_(alpha_cm3_s=0.0)
    )
    expected = gas.free_electron_fraction(chamber.gap_cm, chamber.field_V_cm)
    assert ep.free_electron_fraction == pytest.approx(expected, rel=1e-10)
    assert ep.collection_efficiency == pytest.approx(1.0, rel=1e-11)


@pytest.mark.parametrize("u, lam_over_d", [(0.5, 0.05), (2.0, 0.1), (5.0, 0.3), (1.0, 1.0)])
def test_fenwick_kumar_exact_solution(chamber, gas_no_diffusion, u, lam_over_d):
    gas = _constant_attachment(gas_no_diffusion, lam_over_d * chamber.gap_cm, chamber.field_V_cm)
    n0 = u / boag_pulsed_u(1.0, chamber, gas)
    ep = Uniform1D(Grid1D(n_cells=400), **FE_ONLY).run_episode(
        PiecewiseConstant.pulse(n0, 0.0, INSTANT), chamber, gas
    )
    exact = boag_pulsed_free_electron_f(n0, chamber, gas)
    assert (1 - ep.collection_efficiency) == pytest.approx(1 - exact, rel=3e-4)
    assert exact > boag_pulsed_f(n0, chamber, gas)  # free electrons escape recombination


def test_fenwick_kumar_limits(chamber, gas_no_diffusion):
    n0 = 1.0 / boag_pulsed_u(1.0, chamber, gas_no_diffusion)
    short = _constant_attachment(gas_no_diffusion, 1e-7, chamber.field_V_cm)
    long = _constant_attachment(gas_no_diffusion, 100 * chamber.gap_cm, chamber.field_V_cm)
    assert boag_pulsed_free_electron_f(n0, chamber, short) == pytest.approx(
        boag_pulsed_f(n0, chamber, gas_no_diffusion), rel=1e-6
    )
    assert boag_pulsed_free_electron_f(n0, chamber, long) == pytest.approx(1.0, abs=5e-3)


# ---------------------------------------------------------------- space charge
def test_field_satisfies_gauss_and_the_applied_voltage():
    rng = np.random.default_rng(1)
    n_pos, n_neg = rng.uniform(0, 1e9, 64), rng.uniform(0, 1e9, 64)
    dz, e0 = 0.2 / 64, 1000.0
    ef = np.zeros(65)
    solve_field(n_pos, n_neg, e0, FIELD_COEF_V_CM, dz, ef)
    assert np.sum(0.5 * (ef[:-1] + ef[1:])) * dz == pytest.approx(e0 * 0.2, rel=1e-12)
    np.testing.assert_allclose(np.diff(ef), FIELD_COEF_V_CM * (n_pos - n_neg) * dz, rtol=1e-9)


@pytest.mark.parametrize("n0", [2e9, 1e10, 1e12])
def test_space_charge_conserves_charge_and_the_ramo_identity(chamber, gas, n0):
    ep = Uniform1D(Grid1D(n_cells=150)).run_episode(
        PiecewiseConstant.pulse(n0, 0.0, INSTANT), chamber, gas
    )
    assert ep.emptied
    assert ep.recombined_cm2 + ep.collected_pos_cm2 == pytest.approx(ep.produced_cm2, rel=1e-11)
    assert ep.collected_neg_cm2 == pytest.approx(ep.collected_pos_cm2, rel=1e-10)
    # charge collected on the wrong electrode induces the opposite signal
    net = ep.collected_pos_cm2 - ep.backflow_pos_cm2 - ep.backflow_neg_cm2
    assert ep.signal_cm2 == pytest.approx(net, rel=1e-10)


def test_space_charge_effect_vanishes_linearly_with_density(chamber, gas):
    """The field distortion is proportional to the charge density, and so is its effect."""
    effect = []
    for amount in (2e7, 2e6):
        src = PiecewiseConstant.pulse_train(amount, 4e-6, 3e-5, 10)
        a = Uniform1D(Grid1D(n_cells=150)).run_episode(src, chamber, gas)
        b = Uniform1D(Grid1D(n_cells=150), space_charge=False).run_episode(src, chamber, gas)
        effect.append((1 - a.f_recombination) / (1 - b.f_recombination) - 1)
    assert abs(effect[1]) < 5e-4
    assert effect[0] / effect[1] == pytest.approx(10.0, rel=0.2)


def test_space_charge_lowers_the_efficiency_of_dense_pulses(chamber, gas):
    """The separating charge screens the field in the overlap region, which then lasts longer."""
    src = PiecewiseConstant.pulse(1e10, 0.0, INSTANT)
    with_sc = Uniform1D(Grid1D(n_cells=150)).run_episode(src, chamber, gas)
    without = Uniform1D(Grid1D(n_cells=150), space_charge=False).run_episode(src, chamber, gas)
    assert with_sc.f_recombination < without.f_recombination - 0.01


def test_dense_plasma_stays_positive_and_does_not_reverse_spuriously(chamber, gas):
    """At 1e12 cm^-3 the dielectric relaxation time (~0.2 us) is shorter than a step: the drift
    sub-cycles instead of overshooting into a reversed field."""
    ep = Uniform1D(Grid1D(n_cells=150)).run_episode(
        PiecewiseConstant.pulse(1e12, 0.0, INSTANT), chamber, gas
    )
    assert ep.max_substeps > 5
    assert (ep.backflow_pos_cm2 + ep.backflow_neg_cm2) / ep.produced_cm2 < 5e-3


def test_snapshots_record_the_gap_profiles(chamber, gas):
    t_c = max(chamber.collection_times_s(gas))
    ep = Uniform1D(Grid1D(n_cells=80)).run_episode(
        PiecewiseConstant.pulse(1e9, 0.0, INSTANT), chamber, gas, snapshot_times_s=[0.0, 0.5 * t_c]
    )
    snap = ep.snapshots
    assert snap.n_pos_cm3.shape == (2, 80) and snap.field_V_cm.shape == (2, 81)
    assert snap.times_s[1] == pytest.approx(0.5 * t_c, abs=ep.dt_s)
    # halfway through, the positive ions have left the anode half of the gap
    assert snap.n_pos_cm3[1, :30].sum() < 1e-4 * snap.n_pos_cm3[0].sum()
    assert np.all(snap.n_neg_cm3 >= 0)
    chamber_ok = ParallelPlateChamber(chamber.gap_cm, chamber.voltage_V)
    assert snap.z_cm[-1] < chamber_ok.gap_cm
