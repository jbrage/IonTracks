"""Episodes, deduplication, time-structure limits and the preflight budget."""

import numpy as np
import pytest

from iontracks import (
    BudgetExceededError,
    Grid1D,
    PiecewiseConstant,
    Uniform1D,
    plan_point,
    solve_point,
)
from iontracks.planner import split_episodes

MODEL = Uniform1D(Grid1D(n_cells=100))


def test_memory_time_exceeds_slow_ion_transit(chamber, gas):
    t_pos, t_neg = chamber.collection_times_s(gas)
    assert t_pos > t_neg  # positive ions are slower in air
    assert MODEL.memory_time_s(chamber, gas) > t_pos


def test_episode_splitting(chamber, gas):
    mem = MODEL.memory_time_s(chamber, gas)
    far = PiecewiseConstant.from_intervals([0.0, 2 * mem], [1e-6, 2 * mem + 1e-6], 1.0)
    near = PiecewiseConstant.from_intervals([0.0, 0.5 * mem], [1e-6, 0.5 * mem + 1e-6], 1.0)
    assert len(split_episodes(far, mem)) == 2
    assert len(split_episodes(near, mem)) == 1


def test_pulse_train_is_deduplicated(chamber, gas):
    mem = MODEL.memory_time_s(chamber, gas)
    train = PiecewiseConstant.pulse_train(1e9, 4e-6, 3 * mem, 50, t_start_s=0.01)
    plan = plan_point(train, chamber, gas, MODEL)
    assert len(plan.groups) == 1
    assert plan.groups[0].multiplicity == plan.n_episodes == 50


def test_dedupe_does_not_change_results(chamber, gas):
    mem = MODEL.memory_time_s(chamber, gas)
    train = PiecewiseConstant.pulse_train(1e10, 4e-6, 2 * mem, 5)
    fast = solve_point(train, chamber, gas, MODEL, dedupe=True)
    full = solve_point(train, chamber, gas, MODEL, dedupe=False)
    assert len(full.plan.groups) == 5 and len(fast.plan.groups) == 1
    assert fast.produced_cm2 == pytest.approx(full.produced_cm2, rel=1e-12)
    assert fast.recombined_cm2 == pytest.approx(full.recombined_cm2, rel=1e-12)


def test_independent_pulses_equal_single_pulse(chamber, gas):
    """Period longer than the chamber memory: k_s is that of one pulse (Petersson 2017)."""
    mem = MODEL.memory_time_s(chamber, gas)
    single = solve_point(PiecewiseConstant.pulse(2e10, 0.0, 4e-6), chamber, gas, MODEL)
    train = solve_point(
        PiecewiseConstant.pulse_train(2e10, 4e-6, 1.5 * mem, 30), chamber, gas, MODEL
    )
    assert train.ks == pytest.approx(single.ks, rel=1e-12)
    assert train.produced_cm2 == pytest.approx(30 * single.produced_cm2, rel=1e-12)


def test_overlapping_pulses_recombine_more(chamber, gas):
    """Same dose per pulse, period shorter than t_c: charge of successive pulses overlaps
    and recombination increases (Brualla-Gonzalez et al. 2016)."""
    t_c = max(chamber.collection_times_s(gas))
    separate = solve_point(
        PiecewiseConstant.pulse_train(2e10, 4e-6, 3 * t_c, 10), chamber, gas, MODEL
    )
    overlapping = solve_point(
        PiecewiseConstant.pulse_train(2e10, 4e-6, 0.2 * t_c, 10), chamber, gas, MODEL
    )
    assert len(overlapping.plan.groups) == 1 and overlapping.plan.n_episodes == 1
    assert overlapping.ks > separate.ks


def test_point_current_integrates_to_signal(chamber, gas):
    mem = MODEL.memory_time_s(chamber, gas)
    train = PiecewiseConstant.pulse_train(5e9, 4e-6, 2 * mem, 3)
    res = solve_point(train, chamber, gas, MODEL)
    edges = np.linspace(-mem, 8 * mem, 4001)
    j = res.current_density_A_cm2(edges)
    assert np.sum(j * np.diff(edges)) == pytest.approx(1.602176634e-19 * res.signal_cm2, rel=1e-10)
    assert res.collection_efficiency == pytest.approx(res.f_recombination, rel=1e-10)


def test_budget_is_enforced(chamber, gas):
    with pytest.raises(BudgetExceededError):
        solve_point(PiecewiseConstant.constant(1e12, 0.0, 1.0), chamber, gas, MODEL, budget_s=1e-6)


def test_zero_source_gives_nan_ks(chamber, gas):
    res = solve_point(PiecewiseConstant.zero(), chamber, gas, MODEL)
    assert res.produced_cm2 == 0.0
    assert res.plan.n_episodes == 0
    assert res.ks != res.ks  # nan
