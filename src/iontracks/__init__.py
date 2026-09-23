"""IonTracks: ion recombination in parallel-plate ionization chambers.

Beams with arbitrary (user-defined) time structure, evaluated at lateral points of the chamber.
"""

from .api import plan_field, solve_field, solve_point
from .chamber import ParallelPlateChamber, SamplePoints
from .field import Field
from .gas import Gas
from .planner import BudgetExceededError, PointPlan, plan_point
from .results import FieldResult, PointResult
from .solvers import EpisodeResult, Grid1D, Grid3D, Tile3D, Uniform1D
from .time_structure import PiecewiseConstant
from .tracks import GaussianTrack, track_radius_cm

__version__ = "4.0.0.dev0"

__all__ = [
    "BudgetExceededError",
    "EpisodeResult",
    "Field",
    "FieldResult",
    "GaussianTrack",
    "Gas",
    "Grid1D",
    "Grid3D",
    "ParallelPlateChamber",
    "PiecewiseConstant",
    "PointPlan",
    "PointResult",
    "SamplePoints",
    "Tile3D",
    "Uniform1D",
    "plan_field",
    "plan_point",
    "solve_field",
    "solve_point",
    "track_radius_cm",
]
