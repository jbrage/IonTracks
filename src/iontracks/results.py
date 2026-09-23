"""Result objects for a single lateral point and for a whole chamber."""

import math
from dataclasses import dataclass

import numpy as np

from .constants import ELEMENTARY_CHARGE_C
from .planner import PointPlan
from .solvers.uniform1d import EpisodeResult


def _ratio_or_nan(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


@dataclass(frozen=True)
class PointResult:
    """All episodes at one lateral point; tallies are per cm^2 of electrode area."""

    plan: PointPlan
    episodes: tuple[EpisodeResult, ...]  # aligned with plan.groups
    x_cm: float = math.nan
    y_cm: float = math.nan
    weight_cm2: float = 1.0
    warnings: tuple[str, ...] = ()

    def _sum(self, attr: str) -> float:
        return float(
            sum(
                g.multiplicity * getattr(e, attr)
                for g, e in zip(self.plan.groups, self.episodes, strict=True)
            )
        )

    @property
    def produced_cm2(self) -> float:
        return self._sum("produced_cm2")

    @property
    def recombined_cm2(self) -> float:
        return self._sum("recombined_cm2")

    @property
    def signal_cm2(self) -> float:
        return self._sum("signal_cm2")

    @property
    def remaining_cm2(self) -> float:
        return self._sum("remaining_cm2")

    @property
    def f_recombination(self) -> float:
        return 1.0 - _ratio_or_nan(self.recombined_cm2, self.produced_cm2)

    @property
    def ks(self) -> float:
        """Recombination correction factor P / (P - R)."""
        return _ratio_or_nan(self.produced_cm2, self.produced_cm2 - self.recombined_cm2)

    @property
    def collection_efficiency(self) -> float:
        """Signal / produced (equals f_recombination once every episode has emptied)."""
        return _ratio_or_nan(self.signal_cm2, self.produced_cm2)

    def current_density_A_cm2(self, t_edges_s) -> np.ndarray:
        """Induced current density averaged over the bins [t_edges[i], t_edges[i+1])."""
        t_edges_s = np.asarray(t_edges_s, dtype=float)
        charge = np.zeros(t_edges_s.size - 1)
        for group, ep in zip(self.plan.groups, self.episodes, strict=True):
            for start in group.starts_s:
                charge += np.diff(ep.cumulative("induced", t_edges_s - start))
        return ELEMENTARY_CHARGE_C * charge / np.diff(t_edges_s)


@dataclass(frozen=True)
class FieldResult:
    """Chamber totals are area-weighted sums over the sample points (charge-weighted k_s)."""

    points: tuple[PointResult, ...]

    def _weighted(self, attr: str) -> float:
        return float(sum(p.weight_cm2 * getattr(p, attr) for p in self.points))

    @property
    def produced(self) -> float:
        """Ion pairs created in the sampled sensitive volume."""
        return self._weighted("produced_cm2")

    @property
    def recombined(self) -> float:
        return self._weighted("recombined_cm2")

    @property
    def signal(self) -> float:
        """Collected (induced) charge / e."""
        return self._weighted("signal_cm2")

    @property
    def charge_C(self) -> float:
        return ELEMENTARY_CHARGE_C * self.signal

    @property
    def f_recombination(self) -> float:
        return 1.0 - _ratio_or_nan(self.recombined, self.produced)

    @property
    def ks(self) -> float:
        return _ratio_or_nan(self.produced, self.produced - self.recombined)

    @property
    def collection_efficiency(self) -> float:
        return _ratio_or_nan(self.signal, self.produced)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(w for p in self.points for w in p.warnings)

    def current_A(self, t_edges_s) -> np.ndarray:
        """Chamber current averaged over the bins of ``t_edges_s``."""
        t_edges_s = np.asarray(t_edges_s, dtype=float)
        total = np.zeros(t_edges_s.size - 1)
        for p in self.points:
            if p.weight_cm2 > 0:
                total += p.weight_cm2 * p.current_density_A_cm2(t_edges_s)
        return total

    def ks_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x_cm, y_cm, ks) per sample point (nan where nothing was produced)."""
        return (
            np.array([p.x_cm for p in self.points]),
            np.array([p.y_cm for p in self.points]),
            np.array([p.ks for p in self.points]),
        )
