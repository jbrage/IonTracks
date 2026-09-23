"""Beam time structures as exact piecewise-constant rates.

Every time structure (continuous beam, pulses, pulse trains, spot sequences, machine logs) is
represented as a non-negative rate that is constant between breakpoints. Integrals over any
interval are exact, so sources deposited per time step conserve charge for any step size and
sub-step structure (e.g. RF bunches) never needs to be resolved.
"""

import numpy as np


class PiecewiseConstant:
    """Rate r(t) = rates[i] on [edges[i], edges[i+1]), zero outside [edges[0], edges[-1])."""

    __slots__ = ("edges", "rates", "_cum")

    def __init__(self, edges, rates):
        edges = np.array(edges, dtype=float)
        rates = np.array(rates, dtype=float)
        if edges.ndim != 1 or rates.ndim != 1 or edges.size != rates.size + 1 or rates.size < 1:
            raise ValueError("need 1D edges with len(edges) == len(rates) + 1 >= 2")
        if not (np.all(np.isfinite(edges)) and np.all(np.isfinite(rates))):
            raise ValueError("edges and rates must be finite")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be strictly increasing")
        if np.any(rates < 0):
            raise ValueError("rates must be non-negative")
        self.edges = edges
        self.rates = rates
        self._cum = np.concatenate(([0.0], np.cumsum(rates * np.diff(edges))))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_intervals(cls, starts, ends, rates) -> "PiecewiseConstant":
        """Superpose constant rates on [starts[i], ends[i]). Overlaps add up."""
        starts, ends, rates = np.broadcast_arrays(
            np.asarray(starts, float), np.asarray(ends, float), np.asarray(rates, float)
        )
        starts, ends, rates = starts.ravel(), ends.ravel(), rates.ravel()
        if np.any(ends <= starts):
            raise ValueError("every interval needs end > start")
        if np.any(rates < 0):
            raise ValueError("rates must be non-negative")
        keep = rates > 0
        starts, ends, rates = starts[keep], ends[keep], rates[keep]
        if starts.size == 0:
            return cls([0.0, 1.0], [0.0])
        edges = np.unique(np.concatenate([starts, ends]))
        i_start = np.searchsorted(edges, starts)
        i_end = np.searchsorted(edges, ends)
        level = np.zeros(edges.size)
        active = np.zeros(edges.size, dtype=np.int64)
        np.add.at(level, i_start, rates)
        np.add.at(level, i_end, -rates)
        np.add.at(active, i_start, 1)
        np.add.at(active, i_end, -1)
        level = np.cumsum(level)[:-1]
        active = np.cumsum(active)[:-1]
        # exact zeros where no interval is active (avoids round-off "rates" of 1e-20)
        level = np.where(active > 0, np.maximum(level, 0.0), 0.0)
        return cls(edges, level)

    @classmethod
    def constant(cls, rate: float, t_start_s: float, t_end_s: float) -> "PiecewiseConstant":
        return cls.from_intervals([t_start_s], [t_end_s], [rate])

    @classmethod
    def pulse(cls, amount: float, t_start_s: float, width_s: float) -> "PiecewiseConstant":
        """Rectangular pulse delivering ``amount`` (rate x time) over ``width_s``."""
        if not width_s > 0:
            raise ValueError(
                "width_s must be positive; use a short width for 'instantaneous' pulses"
            )
        return cls.from_intervals([t_start_s], [t_start_s + width_s], [amount / width_s])

    @classmethod
    def pulse_train(
        cls,
        amount_per_pulse: float,
        width_s: float,
        period_s: float,
        n_pulses: int,
        t_start_s: float = 0.0,
    ) -> "PiecewiseConstant":
        """``n_pulses`` identical rectangular pulses, one every ``period_s``."""
        if not width_s > 0 or not period_s > 0 or n_pulses < 1:
            raise ValueError("need width_s > 0, period_s > 0 and n_pulses >= 1")
        starts = t_start_s + period_s * np.arange(n_pulses)
        return cls.from_intervals(starts, starts + width_s, amount_per_pulse / width_s)

    @classmethod
    def zero(cls) -> "PiecewiseConstant":
        return cls([0.0, 1.0], [0.0])

    # ------------------------------------------------------------------ evaluation
    @property
    def t_start_s(self) -> float:
        return float(self.edges[0])

    @property
    def t_end_s(self) -> float:
        return float(self.edges[-1])

    @property
    def total(self) -> float:
        """Integral over all time."""
        return float(self._cum[-1])

    @property
    def max_rate(self) -> float:
        return float(self.rates.max())

    def cumulative(self, t):
        """Integral of the rate from -inf to t (vectorised)."""
        t = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(self.edges, t, side="right") - 1, 0, self.rates.size - 1)
        value = self._cum[idx] + self.rates[idx] * (t - self.edges[idx])
        value = np.where(t <= self.edges[0], 0.0, value)
        return np.where(t >= self.edges[-1], self._cum[-1], value)

    def integral(self, t0, t1):
        """Integral of the rate over [t0, t1] (vectorised)."""
        return self.cumulative(t1) - self.cumulative(t0)

    def rate(self, t):
        """Rate at time t (vectorised; right-continuous)."""
        t = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(self.edges, t, side="right") - 1, 0, self.rates.size - 1)
        inside = (t >= self.edges[0]) & (t < self.edges[-1])
        return np.where(inside, self.rates[idx], 0.0)

    def support(self) -> list[tuple[float, float]]:
        """Maximal intervals where the rate is positive."""
        intervals: list[tuple[float, float]] = []
        for i in np.flatnonzero(self.rates > 0):
            lo, hi = float(self.edges[i]), float(self.edges[i + 1])
            if intervals and intervals[-1][1] == lo:
                intervals[-1] = (intervals[-1][0], hi)
            else:
                intervals.append((lo, hi))
        return intervals

    # ------------------------------------------------------------------ transformations
    def window(self, t0: float, t1: float) -> "PiecewiseConstant":
        """Restriction to [t0, t1] (zero outside)."""
        if not t1 > t0:
            raise ValueError("need t1 > t0")
        inner = self.edges[(self.edges > t0) & (self.edges < t1)]
        edges = np.concatenate(([t0], inner, [t1]))
        mids = 0.5 * (edges[:-1] + edges[1:])
        return PiecewiseConstant(edges, self.rate(mids))

    def shifted(self, dt_s: float) -> "PiecewiseConstant":
        return PiecewiseConstant(self.edges + dt_s, self.rates)

    def scaled(self, factor: float) -> "PiecewiseConstant":
        if factor < 0:
            raise ValueError("factor must be non-negative")
        return PiecewiseConstant(self.edges, self.rates * factor)

    def __add__(self, other: "PiecewiseConstant") -> "PiecewiseConstant":
        if not isinstance(other, PiecewiseConstant):
            return NotImplemented
        edges = np.union1d(self.edges, other.edges)
        mids = 0.5 * (edges[:-1] + edges[1:])
        return PiecewiseConstant(edges, self.rate(mids) + other.rate(mids))

    def shape_key(self) -> tuple:
        """Hashable description of the shape relative to its own start time.

        Two structures with equal keys differ only by a time shift (to ~1e-9 relative), so a
        simulation of one applies to the other. The tolerance absorbs the round-off of absolute
        times: a microsecond pulse at t ~ 1 s is only known to ~1e-10 relative. Far beyond ~100 s
        of absolute time, identical microsecond pulses may get different keys; that only costs
        runtime (no deduplication), never accuracy.
        """
        duration = self.t_end_s - self.t_start_s
        rel_edges = np.round((self.edges - self.edges[0]) / duration, 9)
        return (
            _significant(duration),
            tuple(rel_edges.tolist()),
            tuple(_significant(r) for r in self.rates),
        )

    def __repr__(self) -> str:
        return (
            f"PiecewiseConstant({self.rates.size} intervals on [{self.t_start_s:.6g}, "
            f"{self.t_end_s:.6g}] s, total={self.total:.6g})"
        )


def _significant(x: float, digits: int = 9) -> float:
    return float(f"{float(x):.{digits}g}")
