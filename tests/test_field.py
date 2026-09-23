"""Whole fields evaluated at lateral sample points of the chamber."""

import numpy as np
import pytest

from iontracks import Field, Grid1D, PiecewiseConstant, Uniform1D, solve_field, solve_point
from iontracks.constants import ELEMENTARY_CHARGE_C, MEV_TO_J

MODEL = Uniform1D(Grid1D(n_cells=80))
S_PROTON_AIR = 5.0  # MeV cm^2/g, roughly a 150 MeV proton


def _raster(t_dwell=2e-3, t_move=1e-4, n_side=5, spacing=0.4, n_particles=2e8, sigma=0.4):
    """Square raster of spots delivered one after another."""
    xs, ys = np.meshgrid(
        (np.arange(n_side) - n_side // 2) * spacing, (np.arange(n_side) - n_side // 2) * spacing
    )
    t_on = np.arange(n_side * n_side) * (t_dwell + t_move)
    return Field.from_spots(
        t_on_s=t_on,
        t_off_s=t_on + t_dwell,
        x_cm=xs.ravel(),
        y_cm=ys.ravel(),
        n_particles=n_particles,
        sigma_cm=sigma,
        stopping_power_MeV_cm2_g=S_PROTON_AIR,
    )


def test_sample_points_are_an_equal_area_quadrature(chamber):
    pts = chamber.sample_points(n_rings=5, n_azimuth=7)
    assert len(pts) == 35
    assert pts.weight_cm2.sum() == pytest.approx(np.pi * chamber.sensitive_radius_cm**2)
    r2 = pts.x_cm**2 + pts.y_cm**2
    assert np.average(r2, weights=pts.weight_cm2) == pytest.approx(
        chamber.sensitive_radius_cm**2 / 2
    )


def test_local_source_and_dose(gas):
    field = _raster()
    x, y = 0.1, -0.2
    fluence = field.fluence_per_spot_cm2(x, y)
    src = field.local_source(x, y, gas, cutoff_sigma=50.0)
    assert src.total == pytest.approx(
        np.sum(fluence) * gas.ion_pairs_per_cm(S_PROTON_AIR), rel=1e-12
    )
    dose = field.dose_to_gas_Gy(x, y)
    assert dose == pytest.approx(np.sum(fluence) * S_PROTON_AIR * MEV_TO_J * 1e3, rel=1e-12)
    # the ion-pair density of the whole field corresponds to this dose
    assert src.total == pytest.approx(dose * gas.ion_pairs_per_cm3_per_Gy(), rel=1e-12)


def test_far_point_gets_no_source(gas):
    src = _raster().local_source(100.0, 0.0, gas)
    assert src.total == 0.0 and src.support() == []


def test_broad_spot_reproduces_uniform_point(chamber, gas):
    field = Field.from_spots(0.0, 1e-3, 0.0, 0.0, n_particles=1e12, sigma_cm=100.0,
                             stopping_power_MeV_cm2_g=S_PROTON_AIR)  # fmt: skip
    res = solve_field(field, chamber, gas, MODEL, points=chamber.sample_points(2, 4), n_workers=1)
    uniform = solve_point(field.local_source(0.0, 0.0, gas), chamber, gas, MODEL)
    assert res.ks - 1 == pytest.approx(uniform.ks - 1, rel=1e-4)


def test_field_totals_are_charge_weighted_and_conserved(chamber, gas):
    res = solve_field(_raster(), chamber, gas, MODEL, points=chamber.sample_points(3, 6))
    assert not res.warnings
    w = np.array([p.weight_cm2 for p in res.points])
    produced = np.array([p.produced_cm2 for p in res.points])
    recombined = np.array([p.recombined_cm2 for p in res.points])
    assert res.ks == pytest.approx(
        np.sum(w * produced) / np.sum(w * (produced - recombined)), rel=1e-12
    )
    assert res.recombined + res.signal == pytest.approx(res.produced, rel=1e-10)
    assert res.ks > 1.0
    # a raster spreads the dose: every point sees a different history -> k_s varies over the area
    _, _, ks = res.ks_map()
    assert np.ptp(ks) > 0


def test_field_current_integrates_to_collected_charge(chamber, gas):
    field = _raster(n_side=3)
    res = solve_field(field, chamber, gas, MODEL, points=chamber.sample_points(2, 5))
    edges = np.linspace(field.t_start_s - 1e-3, field.t_end_s + 5e-3, 3001)
    current = res.current_A(edges)
    assert np.sum(current * np.diff(edges)) == pytest.approx(res.charge_C, rel=1e-9)
    assert res.charge_C == pytest.approx(ELEMENTARY_CHARGE_C * res.signal)


def test_parallel_and_serial_agree(chamber, gas):
    field = _raster(n_side=3)
    pts = chamber.sample_points(2, 4)
    serial = solve_field(field, chamber, gas, MODEL, points=pts, n_workers=1)
    parallel = solve_field(field, chamber, gas, MODEL, points=pts, n_workers=4)
    assert parallel.ks == serial.ks


def test_higher_intensity_increases_ks(chamber, gas):
    low = solve_field(
        _raster(n_particles=2e7), chamber, gas, MODEL, points=chamber.sample_points(2, 4)
    )
    high = solve_field(
        _raster(n_particles=2e9), chamber, gas, MODEL, points=chamber.sample_points(2, 4)
    )
    assert high.ks > low.ks > 1.0


def test_same_dose_faster_delivery_increases_ks(chamber, gas):
    """Same spots and particles, shorter dwell (higher instantaneous dose rate)."""
    pts = chamber.sample_points(2, 4)
    slow = solve_field(_raster(t_dwell=5e-3), chamber, gas, MODEL, points=pts)
    fast = solve_field(_raster(t_dwell=2e-4), chamber, gas, MODEL, points=pts)
    assert fast.produced == pytest.approx(slow.produced, rel=1e-12)
    assert fast.ks > slow.ks


def test_invalid_spots():
    with pytest.raises(ValueError):
        Field.from_spots(1.0, 1.0, 0.0, 0.0, 1e6, 0.3, 5.0)  # zero duration
    with pytest.raises(ValueError):
        Field.from_spots(0.0, 1.0, 0.0, 0.0, 1e6, 0.0, 5.0)  # zero sigma


def test_point_model_uniform_limit_via_time_structure(chamber, gas):
    """A continuous uniform beam given directly as a time structure works without a field."""
    rate = 1.0 * gas.ion_pairs_per_cm3_per_Gy()  # 1 Gy/s in air
    res = solve_point(PiecewiseConstant.constant(rate, 0.0, 0.01), chamber, gas, MODEL)
    assert 1.0 < res.ks < 1.01
