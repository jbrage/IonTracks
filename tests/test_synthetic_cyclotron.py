"""Synthetic 250 MeV proton PBS field from an isochronous cyclotron, 2 mm gap at 200 V.

The field is a 10 x 10 cm^2 single-energy raster (5 mm spacing, sigma 3.5 mm) giving 2 Gy to
water, scanned row by row with 1 ms spot and 2 ms line switching. The checks cover:

* the synthetic delivery itself (dose, spot count, timing, jitter);
* conservation for the whole field;
* agreement with Boag's continuous theory applied quasi-statically to every constant segment
  when spots are long compared with the ion collection time (clinical currents);
* departure from the quasi-static picture when spots become as short as the collection time;
* irrelevance of the RF micro-structure (72.85 MHz bunches).
"""

import numpy as np
import pytest

from iontracks import (
    Gas,
    Grid1D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Uniform1D,
    solve_field,
    solve_point,
)
from iontracks.analytic import boag_quasi_static_loss
from iontracks.constants import ELEMENTARY_CHARGE_C, MEV_TO_J
from iontracks.synthetic import (
    PROTON_250MEV_S_WATER_MEV_CM2_G,
    W_AIR_PROTONS_EV,
    cyclotron_pbs_field,
    protons_per_spot,
    rf_bunched,
)

CHAMBER = ParallelPlateChamber(gap_cm=0.2, voltage_V=200.0, sensitive_radius_cm=0.78)
GAS = Gas.air(W_eV=W_AIR_PROTONS_EV)
MODEL = Uniform1D(Grid1D(n_cells=120))
POINTS = CHAMBER.sample_points(n_rings=2, n_azimuth=6)


def _quasi_static_ks(field) -> float:
    produced = recombined = 0.0
    for x, y, w in zip(POINTS.x_cm, POINTS.y_cm, POINTS.weight_cm2, strict=True):
        p, r = boag_quasi_static_loss(field.local_source(x, y, GAS), CHAMBER, GAS)
        produced += w * p
        recombined += w * r
    return produced / (produced - recombined)


@pytest.fixture(scope="module")
def results():
    """Field results for a clinical, an elevated and a FLASH-like nozzle current."""
    out = {}
    for current_nA in (2.0, 10.0, 100.0):
        field = cyclotron_pbs_field(beam_current_nA=current_nA)
        out[current_nA] = (field, solve_field(field, CHAMBER, GAS, MODEL, points=POINTS))
    return out


def test_synthetic_delivery(results):
    field, _ = results[2.0]
    assert len(field) == 21 * 21
    dose_w = (
        np.sum(field.fluence_per_spot_cm2(0.0, 0.0))
        * PROTON_250MEV_S_WATER_MEV_CM2_G
        * MEV_TO_J
        * 1e3
    )
    assert dose_w == pytest.approx(2.0, rel=5e-3)  # plateau dose of the uniform raster
    n = protons_per_spot(2.0, 0.5, PROTON_250MEV_S_WATER_MEV_CM2_G)
    dwell = n * ELEMENTARY_CHARGE_C / 2e-9
    np.testing.assert_allclose(field.t_off_s - field.t_on_s, dwell, rtol=1e-12)
    total = 441 * dwell + 420 * 1e-3 + 20 * 2e-3  # 420 spot switches, 20 line changes
    assert field.t_end_s == pytest.approx(total, rel=1e-12)


def test_current_jitter_changes_timing_not_dose():
    ref = cyclotron_pbs_field(beam_current_nA=2.0)
    jit = cyclotron_pbs_field(beam_current_nA=2.0, current_jitter=0.05, seed=3)
    np.testing.assert_array_equal(jit.n_particles, ref.n_particles)
    assert not np.allclose(jit.t_off_s - jit.t_on_s, ref.t_off_s - ref.t_on_s)
    assert jit.dose_to_gas_Gy(0.1, 0.2) == pytest.approx(ref.dose_to_gas_Gy(0.1, 0.2), rel=1e-12)


def test_field_conservation(results):
    for _, res in results.values():
        assert not res.warnings
        assert res.recombined + res.signal == pytest.approx(res.produced, rel=1e-10)


def test_clinical_current_matches_quasi_static_boag(results):
    """64 ms spots are ~300 memory times long, so Boag steady state holds within each spot."""
    field, res = results[2.0]
    ks_qs = _quasi_static_ks(field)
    assert 1.001 < res.ks < 1.01
    assert (res.ks - 1) == pytest.approx(ks_qs - 1, rel=0.01)


def test_recombination_grows_with_beam_current(results):
    ks = [results[i][1].ks for i in (2.0, 10.0, 100.0)]
    assert ks[0] < ks[1] < ks[2]


def test_short_spots_depart_from_quasi_static(results):
    """At 100 nA the 1.3 ms spots last only ~6 memory times: charge build-up and drain after
    every spot matter, and the quasi-static Boag estimate overestimates the loss."""
    field, res = results[100.0]
    ks_qs = _quasi_static_ks(field)
    assert (res.ks - 1) < 0.95 * (ks_qs - 1)


def test_rf_microstructure_is_irrelevant():
    """72.85 MHz bunches (10 % duty) versus the same charge delivered continuously."""
    rate = 1e14  # ion pairs / cm^3 / s, ~0.5 kGy/s in air: strong recombination
    t_on, t_off = 0.0, 0.4e-3
    continuous = solve_point(PiecewiseConstant.constant(rate, t_on, t_off), CHAMBER, GAS, MODEL)
    bunched = solve_point(rf_bunched(rate, t_on, t_off), CHAMBER, GAS, MODEL)
    assert bunched.produced_cm2 == pytest.approx(continuous.produced_cm2, rel=1e-9)
    assert (bunched.ks - 1) == pytest.approx(continuous.ks - 1, rel=1e-4)
