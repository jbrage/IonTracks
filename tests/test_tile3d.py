"""Track-resolved 3D model: bookkeeping, Jaffe's single-track theory, and the uniform limit."""

import numpy as np
import pytest

from iontracks import (
    Gas,
    GaussianTrack,
    Grid1D,
    Grid3D,
    ParallelPlateChamber,
    PiecewiseConstant,
    Tile3D,
    Uniform1D,
    solve_point,
    track_radius_cm,
)
from iontracks.analytic import jaffe_parallel_f
from iontracks.solvers.tile3d import _gauss_weights

CLASSIC = {"free_electrons": False, "space_charge": False}
CH = ParallelPlateChamber(0.1, 200.0)


def _legacy_track(let_keV_um):
    gas = Gas.air_legacy()
    return GaussianTrack(let_keV_um * 1e7 / gas.W_eV, track_radius_cm(let_keV_um))


def test_gaussian_weights_are_normalised_and_periodic():
    w = np.zeros(20)
    _gauss_weights(0.1, 0.3, 1.0, 20, w)  # wide Gaussian near the tile edge wraps around
    assert w.sum() == pytest.approx(1.0, rel=1e-14)
    assert w[-1] > w[10]  # the mass beyond x = 0 reappears at the far edge
    _gauss_weights(0.525, 0.002, 1.0, 20, w)  # narrow: all in the cell holding the centre
    assert w[10] == pytest.approx(1.0, abs=1e-12)


def test_track_radius_follows_the_v3_table():
    assert track_radius_cm(1e-3) == 2e-3  # protons sit on the 20 um floor
    assert 3e-3 < track_radius_cm(0.04) < 4e-3
    track = GaussianTrack.from_stopping_power(5.0, Gas.air(particle="proton"))
    assert track.let_air_keV_um(Gas.air(particle="proton")) == pytest.approx(5.0 * 1.20479e-4)


@pytest.mark.parametrize("let", [1.14e-3, 4.1e-2])  # 60 MeV protons and 60 MeV/u carbon
def test_single_track_matches_jaffe(let):
    gas = Gas.air_legacy()
    track = _legacy_track(let)
    ep = Tile3D(track, Grid3D(n_xy=48, n_z=100), **CLASSIC).single_track(CH, gas)
    assert ep.emptied
    assert (ep.ks - 1) == pytest.approx(1 / jaffe_parallel_f(track, CH, gas) - 1, rel=0.01)


def test_single_track_converges_towards_jaffe():
    gas = Gas.air_legacy()
    track = _legacy_track(4.1e-2)
    jaffe = 1 / jaffe_parallel_f(track, CH, gas) - 1
    err = [
        abs(Tile3D(track, Grid3D(n_xy=n, n_z=100), **CLASSIC).single_track(CH, gas).ks - 1 - jaffe)
        for n in (24, 48)
    ]
    assert err[1] < 0.6 * err[0]


@pytest.mark.parametrize("physics", [CLASSIC, {}], ids=["classic", "full"])
def test_conservation_and_ramo(physics):
    gas = Gas.air(particle="carbon")
    track = GaussianTrack(3e4, 3e-3)
    model = Tile3D(track, Grid3D(n_xy=24, n_z=60), seed=3, **physics)
    src = PiecewiseConstant.constant(3e13, 0.0, 2e-4)
    ep = model.run_episode(src, CH, gas)
    assert ep.emptied
    assert ep.recombined_cm2 + ep.collected_pos_cm2 == pytest.approx(ep.produced_cm2, rel=1e-11)
    assert ep.collected_neg_cm2 == pytest.approx(ep.collected_pos_cm2, rel=1e-10)
    net = ep.collected_pos_cm2 - ep.backflow_pos_cm2 - ep.backflow_neg_cm2
    assert ep.signal_cm2 == pytest.approx(net, rel=1e-9)


def test_track_count_follows_the_source():
    gas = Gas.air(particle="proton")
    track = GaussianTrack(200.0, 2e-3)
    model = Tile3D(track, Grid3D(n_xy=16, n_z=20), seed=1, **CLASSIC)
    src = PiecewiseConstant.constant(5e12, 0.0, 1e-4)
    ep = model.run_episode(src, CH, gas)
    expected = src.total * CH.gap_cm
    tile = model.tile_size_cm(CH, gas)
    sd = np.sqrt(expected * tile**2 / (track.pairs_per_cm * CH.gap_cm)) * (
        track.pairs_per_cm * CH.gap_cm / tile**2
    )
    assert abs(ep.produced_cm2 - expected) < 4 * sd


def test_wide_overlapping_tracks_reproduce_the_uniform_model():
    """Tracks much wider than the tile deposit uniformly: the 3D model must equal the 1D one."""
    gas = Gas.air(particle="proton")
    ch = ParallelPlateChamber(0.2, 200.0)
    model = Tile3D(
        GaussianTrack(2e3, 1.0), Grid3D(n_xy=8, n_z=100, tile_cm=0.01), seed=2, **CLASSIC
    )
    ep3 = model.run_episode(PiecewiseConstant.pulse(2e9, 0.0, 1e-9), ch, gas)
    n0 = ep3.produced_cm2 / ch.gap_cm  # the Poisson draw sets the actual density
    ep1 = Uniform1D(Grid1D(n_cells=100), **CLASSIC).run_episode(
        PiecewiseConstant.pulse(n0, 0.0, 1e-9), ch, gas
    )
    assert (1 - ep3.f_recombination) == pytest.approx(1 - ep1.f_recombination, rel=1e-3)


def test_planner_accepts_the_track_model():
    gas = Gas.air(particle="proton")
    model = Tile3D(GaussianTrack(200.0, 2e-3), Grid3D(n_xy=12, n_z=20), **CLASSIC)
    mem = model.memory_time_s(CH, gas)
    train = PiecewiseConstant.pulse_train(1e10, 4e-6, 3 * mem, 4)
    res = solve_point(train, CH, gas, model)
    assert res.plan.n_episodes == 4 and len(res.plan.groups) == 1
    assert 1.05 < res.ks < 1.2  # u ~ 0.2 per pulse


def test_track_snapshots():
    gas = Gas.air_legacy()
    model = Tile3D(_legacy_track(4.1e-2), Grid3D(n_xy=24, n_z=40), **CLASSIC)
    ep = model.single_track(CH, gas, snapshot_times_s=[0.0, 2e-5])
    snap = ep.snapshots
    assert snap.n_pos_xz_cm3.shape == (2, 24, 40) and snap.n_pos_mean_cm3.shape == (2, 40)
    assert snap.n_pos_xz_cm3[0].max() > snap.n_pos_xz_cm3[1].max()  # the column spreads out
