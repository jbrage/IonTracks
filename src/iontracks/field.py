"""Scanned (or scattered) ion fields: spots with a time window, position, size and intensity.

Particles are assumed to traverse the chamber perpendicular to the electrodes. The ionisation
history at a lateral point (x, y) is the superposition of all spots' Gaussian fluence
contributions over their on-times. It is piecewise constant in time and exact.
"""

from dataclasses import dataclass

import numpy as np

from .constants import MEV_TO_J
from .gas import Gas
from .time_structure import PiecewiseConstant


@dataclass(frozen=True)
class Field:
    """Spot list. All arrays have one entry per spot.

    ``n_particles`` is the number of primaries in the spot. ``stopping_power_MeV_cm2_g`` is the
    (unrestricted) mass stopping power in the chamber gas at the chamber position. Each spot's
    particles are delivered uniformly over [t_on_s, t_off_s).
    """

    t_on_s: np.ndarray
    t_off_s: np.ndarray
    x_cm: np.ndarray
    y_cm: np.ndarray
    sigma_x_cm: np.ndarray
    sigma_y_cm: np.ndarray
    n_particles: np.ndarray
    stopping_power_MeV_cm2_g: np.ndarray

    def __post_init__(self):
        names = (
            "t_on_s", "t_off_s", "x_cm", "y_cm", "sigma_x_cm", "sigma_y_cm", "n_particles",
            "stopping_power_MeV_cm2_g",
        )  # fmt: skip
        arrays = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(getattr(self, n), float)) for n in names)
        )
        if arrays[0].ndim != 1:
            raise ValueError("spot arrays must be 1D")
        for name, arr in zip(names, arrays, strict=True):
            object.__setattr__(self, name, np.array(arr))
        if np.any(self.t_off_s <= self.t_on_s):
            raise ValueError("every spot needs t_off_s > t_on_s")
        if np.any(self.sigma_x_cm <= 0) or np.any(self.sigma_y_cm <= 0):
            raise ValueError("spot sigmas must be positive")
        if np.any(self.n_particles < 0) or np.any(self.stopping_power_MeV_cm2_g <= 0):
            raise ValueError("n_particles must be >= 0 and stopping powers > 0")

    @classmethod
    def from_spots(
        cls,
        t_on_s,
        t_off_s,
        x_cm,
        y_cm,
        n_particles,
        sigma_cm,
        stopping_power_MeV_cm2_g,
        sigma_y_cm=None,
    ) -> "Field":
        """Build a field; scalars are broadcast. ``sigma_cm`` is used for x (and y if not given)."""
        return cls(
            t_on_s=t_on_s,
            t_off_s=t_off_s,
            x_cm=x_cm,
            y_cm=y_cm,
            sigma_x_cm=sigma_cm,
            sigma_y_cm=sigma_cm if sigma_y_cm is None else sigma_y_cm,
            n_particles=n_particles,
            stopping_power_MeV_cm2_g=stopping_power_MeV_cm2_g,
        )

    def __len__(self) -> int:
        return self.t_on_s.size

    @property
    def t_start_s(self) -> float:
        return float(self.t_on_s.min())

    @property
    def t_end_s(self) -> float:
        return float(self.t_off_s.max())

    def fluence_per_spot_cm2(self, x_cm: float, y_cm: float) -> np.ndarray:
        """Fluence (particles/cm^2) that each spot delivers at (x, y)."""
        u = (x_cm - self.x_cm) / self.sigma_x_cm
        v = (y_cm - self.y_cm) / self.sigma_y_cm
        return (
            self.n_particles
            * np.exp(-0.5 * (u * u + v * v))
            / (2 * np.pi * self.sigma_x_cm * self.sigma_y_cm)
        )

    def dose_to_gas_Gy(self, x_cm: float, y_cm: float) -> float:
        """Absorbed dose to the chamber gas at (x, y)."""
        fluence = self.fluence_per_spot_cm2(x_cm, y_cm)
        return float(np.sum(fluence * self.stopping_power_MeV_cm2_g) * MEV_TO_J * 1e3)

    def local_source(
        self, x_cm: float, y_cm: float, gas: Gas, cutoff_sigma: float = 6.0
    ) -> PiecewiseConstant:
        """Ion-pair production rate density (cm^-3 s^-1) at (x, y) as an exact time structure.

        Spots farther than ``cutoff_sigma`` (Mahalanobis distance) contribute less than
        exp(-cutoff_sigma^2 / 2) of their peak and are skipped.
        """
        u = (x_cm - self.x_cm) / self.sigma_x_cm
        v = (y_cm - self.y_cm) / self.sigma_y_cm
        near = (u * u + v * v) <= cutoff_sigma**2
        if not np.any(near):
            return PiecewiseConstant.zero()
        fluence = self.fluence_per_spot_cm2(x_cm, y_cm)[near]
        pairs_per_cm = gas.ion_pairs_per_cm(self.stopping_power_MeV_cm2_g[near])
        duration = self.t_off_s[near] - self.t_on_s[near]
        return PiecewiseConstant.from_intervals(
            self.t_on_s[near], self.t_off_s[near], fluence * pairs_per_cm / duration
        )
