"""High-level entry points: one lateral point, or a whole field over the chamber area."""

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from .chamber import ParallelPlateChamber, SamplePoints
from .field import Field
from .gas import Gas
from .planner import BudgetExceededError, PointPlan, plan_point
from .results import FieldResult, PointResult
from .solvers.uniform1d import EpisodeResult, Uniform1D
from .time_structure import PiecewiseConstant


class _EpisodeCache:
    """Thread-safe cache of episode results keyed by episode shape (model/chamber/gas fixed)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[tuple, EpisodeResult] = {}

    def get_or_run(self, source: PiecewiseConstant, run) -> EpisodeResult:
        key = source.shape_key()
        with self._lock:
            hit = self._results.get(key)
        if hit is not None:
            return hit
        result = run(source)
        with self._lock:
            self._results.setdefault(key, result)
        return result


def _check_budget(estimated_s: float, budget_s: float | None, detail: str) -> None:
    if budget_s is not None and estimated_s > budget_s:
        raise BudgetExceededError(
            f"estimated {estimated_s:.3g} s exceeds budget {budget_s:.3g} s ({detail}). "
            "Coarsen the grid, reduce sample points, or raise budget_s."
        )


def _run_plan(plan, chamber, gas, model, cache, x_cm, y_cm, weight_cm2) -> PointResult:
    def run(src):
        return model.run_episode(src, chamber, gas)

    episodes, warnings = [], []
    for group in plan.groups:
        ep = cache.get_or_run(group.source, run) if cache is not None else run(group.source)
        episodes.append(ep)
        if not ep.emptied:
            warnings.append(
                f"point ({x_cm:.4g}, {y_cm:.4g}) cm: episode not emptied after {ep.n_steps} steps "
                f"({ep.remaining_cm2 / ep.produced_cm2:.2e} of the charge left)"
            )
        drain_s = ep.n_steps * ep.dt_s - group.source.t_end_s
        if drain_s > min(group.gaps_after_s):
            warnings.append(
                f"point ({x_cm:.4g}, {y_cm:.4g}) cm: gap emptied after {drain_s:.3g} s, longer "
                f"than the {min(group.gaps_after_s):.3g} s beam-off gap; episodes may interact"
            )
    return PointResult(plan, tuple(episodes), x_cm, y_cm, weight_cm2, tuple(warnings))


def solve_point(
    source: PiecewiseConstant,
    chamber: ParallelPlateChamber,
    gas: Gas | None = None,
    model=None,
    *,
    dedupe: bool = True,
    budget_s: float | None = None,
) -> PointResult:
    """Recombination for a laterally uniform ionisation history.

    ``source`` is the ion-pair production rate density in cm^-3 s^-1 (for photon/electron beams
    use ``dose_rate * gas.ion_pairs_per_cm3_per_Gy()``).
    """
    gas = gas or Gas.air()
    model = model or Uniform1D()
    plan = plan_point(source, chamber, gas, model, dedupe=dedupe)
    _check_budget(plan.estimated_seconds, budget_s, plan.summary())
    cache = _EpisodeCache() if dedupe else None
    return _run_plan(plan, chamber, gas, model, cache, math.nan, math.nan, 1.0)


def plan_field(
    field: Field,
    chamber: ParallelPlateChamber,
    gas: Gas | None = None,
    model=None,
    *,
    points: SamplePoints | None = None,
    cutoff_sigma: float = 6.0,
    dedupe: bool = True,
) -> tuple[SamplePoints, list[PointPlan]]:
    """Plan every sample point without running anything (for inspection and budgeting)."""
    gas = gas or Gas.air()
    model = model or Uniform1D()
    points = points if points is not None else chamber.sample_points()
    plans = []
    for x, y in zip(points.x_cm, points.y_cm, strict=True):
        source = field.local_source(x, y, gas, cutoff_sigma=cutoff_sigma)
        plans.append(plan_point(source, chamber, gas, model, dedupe=dedupe))
    return points, plans


def solve_field(
    field: Field,
    chamber: ParallelPlateChamber,
    gas: Gas | None = None,
    model=None,
    *,
    points: SamplePoints | None = None,
    cutoff_sigma: float = 6.0,
    dedupe: bool = True,
    n_workers: int | None = None,
    budget_s: float | None = None,
) -> FieldResult:
    """Recombination for a whole field, evaluated at lateral sample points of the chamber.

    Points are independent (lateral ion transport during one collection time is ~0.1 mm) and
    run in parallel threads (the Numba kernels release the GIL). Episodes with identical shape
    are simulated once across all points.
    """
    gas = gas or Gas.air()
    model = model or Uniform1D()
    points, plans = plan_field(
        field, chamber, gas, model, points=points, cutoff_sigma=cutoff_sigma, dedupe=dedupe
    )
    workers = n_workers or os.cpu_count() or 1
    total_s = sum(p.estimated_seconds for p in plans)
    _check_budget(
        total_s / workers,
        budget_s,
        f"{len(points)} points, {total_s:.3g} CPU-s on {workers} threads",
    )
    cache = _EpisodeCache() if dedupe else None
    args = list(zip(plans, points.x_cm, points.y_cm, points.weight_cm2, strict=True))

    def run(item):
        plan, x, y, w = item
        return _run_plan(plan, chamber, gas, model, cache, float(x), float(y), float(w))

    if workers == 1:
        results = [run(a) for a in args]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, args))
    return FieldResult(tuple(results))
