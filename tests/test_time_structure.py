import numpy as np
import pytest

from iontracks import PiecewiseConstant


def _brute_integral(pc: PiecewiseConstant, t0: float, t1: float) -> float:
    lo = np.clip(pc.edges[:-1], t0, t1)
    hi = np.clip(pc.edges[1:], t0, t1)
    return float(np.sum(pc.rates * (hi - lo)))


def test_integral_is_exact_on_random_windows():
    rng = np.random.default_rng(1)
    edges = np.cumsum(rng.uniform(0.1, 1.0, 51))
    pc = PiecewiseConstant(edges, rng.uniform(0, 5, 50))
    for t0, t1 in rng.uniform(edges[0] - 2, edges[-1] + 2, (200, 2)):
        t0, t1 = min(t0, t1), max(t0, t1)
        assert pc.integral(t0, t1) == pytest.approx(
            _brute_integral(pc, t0, t1), rel=1e-12, abs=1e-12
        )


def test_vectorised_integral_matches_scalar():
    pc = PiecewiseConstant.pulse_train(2.0, 1e-6, 1e-3, 10)
    edges = np.linspace(-1e-3, 1.2e-2, 1001)
    per_bin = np.diff(pc.cumulative(edges))
    assert per_bin.sum() == pytest.approx(pc.total, rel=1e-12)
    assert pc.total == pytest.approx(20.0, rel=1e-12)


def test_superposition_and_exact_zeros():
    pc = PiecewiseConstant.from_intervals([0.0, 0.5, 2.0], [1.0, 1.5, 3.0], [1.0, 2.0, 0.1 + 0.2])
    assert pc.rate(0.25) == 1.0
    assert pc.rate(0.75) == 3.0
    assert pc.rate(1.75) == 0.0  # exactly zero, not round-off
    assert pc.support() == [(0.0, 1.5), (2.0, 3.0)]


def test_pulse_train_support_has_one_interval_per_pulse():
    pc = PiecewiseConstant.pulse_train(1.0, width_s=2e-6, period_s=1e-3, n_pulses=5, t_start_s=0.1)
    support = pc.support()
    assert len(support) == 5
    assert support[3][0] == pytest.approx(0.1 + 3e-3)
    gaps = [0.5 * (a[1] + b[0]) for a, b in zip(support, support[1:], strict=False)]
    assert all(pc.rate(t) == 0.0 for t in gaps)


def test_addition_is_linear():
    a = PiecewiseConstant.constant(1.0, 0.0, 2.0)
    b = PiecewiseConstant.pulse(3.0, 1.0, 0.5)
    s = a + b
    for t0, t1 in [(-1, 5), (0.2, 1.2), (1.1, 1.3)]:
        assert s.integral(t0, t1) == pytest.approx(a.integral(t0, t1) + b.integral(t0, t1))


def test_window_and_shift():
    pc = PiecewiseConstant.pulse_train(1.0, 1e-6, 1e-4, 4)
    w = pc.window(1e-4, 2.5e-4)
    assert w.total == pytest.approx(2.0)
    assert w.shifted(-1e-4).integral(0, 1e-6) == pytest.approx(1.0)


def test_shape_key_is_shift_invariant():
    p1 = PiecewiseConstant.pulse(5.0, 0.0, 3e-6)
    p2 = PiecewiseConstant.pulse(5.0, 0.37, 3e-6).shifted(-0.37)
    p3 = PiecewiseConstant.pulse(5.0 * (1 + 1e-6), 0.0, 3e-6)
    late = PiecewiseConstant.pulse(5.0, 12.345678, 3e-6).shifted(-12.345678)  # late in a delivery
    assert p1.shape_key() == p2.shape_key()
    assert p1.shape_key() == late.shape_key()
    assert p1.shape_key() != p3.shape_key()


@pytest.mark.parametrize(
    "edges, rates",
    [([0.0, 0.0], [1.0]), ([1.0, 0.0], [1.0]), ([0.0, 1.0], [-1.0]), ([0.0, 1.0, 2.0], [1.0])],
)
def test_invalid_input(edges, rates):
    with pytest.raises(ValueError):
        PiecewiseConstant(edges, rates)


def test_pulse_needs_positive_width():
    with pytest.raises(ValueError):
        PiecewiseConstant.pulse(1.0, 0.0, 0.0)
