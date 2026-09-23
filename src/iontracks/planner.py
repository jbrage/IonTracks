"""Split a time structure into independent episodes and estimate the cost before running.

The chamber has a finite memory: once the beam has been off for longer than the model's
``memory_time_s`` (about one slow-ion transit time), the gap is empty. Beam-on intervals
separated by longer gaps are therefore independent episodes. Their tallies add, and episodes
that differ only by a time shift (pulse trains, repeated patterns) are simulated once.
"""

from dataclasses import dataclass

from .chamber import ParallelPlateChamber
from .gas import Gas
from .time_structure import PiecewiseConstant


class BudgetExceededError(RuntimeError):
    """Raised when the estimated runtime exceeds the requested budget."""


@dataclass(frozen=True)
class EpisodeGroup:
    """Episodes with identical shape. ``source`` starts at t = 0."""

    source: PiecewiseConstant
    starts_s: tuple[float, ...]
    gaps_after_s: tuple[float, ...]  # beam-off time after each occurrence (inf for the last)

    @property
    def multiplicity(self) -> int:
        return len(self.starts_s)


@dataclass(frozen=True)
class PointPlan:
    groups: tuple[EpisodeGroup, ...]
    memory_time_s: float
    cell_steps: float
    estimated_seconds: float

    @property
    def n_episodes(self) -> int:
        return sum(g.multiplicity for g in self.groups)

    def summary(self) -> str:
        return (
            f"{self.n_episodes} episode(s), {len(self.groups)} distinct; memory time "
            f"{self.memory_time_s * 1e6:.1f} us; {self.cell_steps:.3g} cell-steps; "
            f"~{self.estimated_seconds:.3g} s"
        )


def split_episodes(source: PiecewiseConstant, memory_time_s: float) -> list[tuple[float, float]]:
    """Beam-on windows, merged whenever the beam-off gap between them is shorter than the memory."""
    windows: list[tuple[float, float]] = []
    for lo, hi in source.support():
        if windows and lo - windows[-1][1] < memory_time_s:
            windows[-1] = (windows[-1][0], hi)
        else:
            windows.append((lo, hi))
    return windows


def plan_point(
    source: PiecewiseConstant,
    chamber: ParallelPlateChamber,
    gas: Gas,
    model,
    *,
    dedupe: bool = True,
) -> PointPlan:
    memory = model.memory_time_s(chamber, gas)
    windows = split_episodes(source, memory)
    groups: dict[object, list] = {}
    for i, (lo, hi) in enumerate(windows):
        episode = source.window(lo, hi).shifted(-lo)
        gap_after = windows[i + 1][0] - hi if i + 1 < len(windows) else float("inf")
        key = episode.shape_key() if dedupe else i
        entry = groups.setdefault(key, [episode, [], []])
        entry[1].append(lo)
        entry[2].append(gap_after)
    plan_groups = tuple(
        EpisodeGroup(src, tuple(starts), tuple(gaps)) for src, starts, gaps in groups.values()
    )
    cell_steps = sum(model.estimate_cell_steps(g.source, chamber, gas) for g in plan_groups)
    return PointPlan(
        groups=plan_groups,
        memory_time_s=memory,
        cell_steps=float(cell_steps),
        estimated_seconds=float(cell_steps) * model.seconds_per_cell_step,
    )
