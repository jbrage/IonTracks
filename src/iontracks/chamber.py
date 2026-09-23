"""Parallel-plate ionization chamber geometry and lateral sample points."""

from dataclasses import dataclass

import numpy as np

from .gas import Gas


@dataclass(frozen=True)
class SamplePoints:
    """Lateral positions in the sensitive volume with their area weights."""

    x_cm: np.ndarray
    y_cm: np.ndarray
    weight_cm2: np.ndarray

    def __post_init__(self):
        x, y, w = (
            np.atleast_1d(np.asarray(a, float)) for a in (self.x_cm, self.y_cm, self.weight_cm2)
        )
        if not x.shape == y.shape == w.shape or x.ndim != 1:
            raise ValueError("x_cm, y_cm and weight_cm2 must be 1D arrays of equal length")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        object.__setattr__(self, "x_cm", x)
        object.__setattr__(self, "y_cm", y)
        object.__setattr__(self, "weight_cm2", w)

    def __len__(self) -> int:
        return len(self.x_cm)


@dataclass(frozen=True)
class ParallelPlateChamber:
    """Plane-parallel chamber with electrodes at z = 0 and z = gap_cm.

    The field is uniform (no space charge). Positive ions drift towards z = gap_cm and negative
    ions towards z = 0; the polarity of ``voltage_V`` only relabels the electrodes.
    """

    gap_cm: float
    voltage_V: float
    sensitive_radius_cm: float | None = None
    center_x_cm: float = 0.0
    center_y_cm: float = 0.0

    def __post_init__(self):
        if not self.gap_cm > 0:
            raise ValueError("gap_cm must be positive")
        if self.voltage_V == 0:
            raise ValueError("voltage_V must be non-zero")
        if self.sensitive_radius_cm is not None and not self.sensitive_radius_cm > 0:
            raise ValueError("sensitive_radius_cm must be positive")

    @property
    def field_V_cm(self) -> float:
        return abs(self.voltage_V) / self.gap_cm

    def drift_velocities_cm_s(self, gas: Gas) -> tuple[float, float]:
        """(positive, negative) ion drift speeds."""
        return gas.mu_pos_cm2_Vs * self.field_V_cm, gas.mu_neg_cm2_Vs * self.field_V_cm

    def collection_times_s(self, gas: Gas) -> tuple[float, float]:
        """(positive, negative) ion transit times across the full gap."""
        v_pos, v_neg = self.drift_velocities_cm_s(gas)
        return self.gap_cm / v_pos, self.gap_cm / v_neg

    def sample_points(self, n_rings: int = 4, n_azimuth: int = 8) -> SamplePoints:
        """Equal-area quadrature over the circular sensitive area.

        Ring i sits at the radius that halves its annulus' area; azimuths are staggered between
        neighbouring rings. All points carry the same weight (total = sensitive area).
        """
        if self.sensitive_radius_cm is None:
            raise ValueError("sensitive_radius_cm is required to generate sample points")
        if n_rings < 1 or n_azimuth < 1:
            raise ValueError("n_rings and n_azimuth must be >= 1")
        radius = self.sensitive_radius_cm
        ring = np.arange(n_rings)
        r = radius * np.sqrt((ring + 0.5) / n_rings)
        phi = 2 * np.pi * (np.arange(n_azimuth)[None, :] + 0.5 * (ring[:, None] % 2)) / n_azimuth
        x = self.center_x_cm + (r[:, None] * np.cos(phi)).ravel()
        y = self.center_y_cm + (r[:, None] * np.sin(phi)).ravel()
        weight = np.full(x.size, np.pi * radius**2 / x.size)
        return SamplePoints(x, y, weight)
