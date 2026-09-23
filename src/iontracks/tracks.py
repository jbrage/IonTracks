"""Ion-track structure for the track-resolved point model.

A track crosses the gap perpendicular to the electrodes as a column of ion pairs with a Gaussian
radial profile n(r) = N0 / (pi b^2) exp(-r^2 / b^2), where N0 is the line density (ion pairs per
cm) and b the track radius. b is an effective radius that includes the spread of the secondary
electrons and the early thermalisation of the ions. It is a model input, not a microscopic
track-structure result.
"""

import math
from dataclasses import dataclass

import numpy as np

from .gas import Gas

# Gaussian track radius against LET in air, from IonTracks v3 (hadrons/data_LET/LET_b.dat, after
# Rossomme et al.). Columns: LET in air (keV/um at air density) x 1e3, radius in units of 10 um.
_LET_RADIUS_TABLE = np.array(
    [
        [6.02089449334, 1.38028169014],
        [8.68926517146, 2.25352112676],
        [10.3171399579, 2.10563380282],
        [32.7532377201, 3.45070422535],
        [53.5566691771, 3.69718309859],
        [56.1245074143, 3.49295774648],
        [58.3581740474, 3.4014084507],
        [136.64483493, 4.14084507042],
        [395.011891135, 4.90845070423],
        [1283.73261818, 5.21126760563],
    ]
)
_RADIUS_FIT = np.polyfit(np.log10(_LET_RADIUS_TABLE[:, 0] * 1e-3), _LET_RADIUS_TABLE[:, 1], 2)
MIN_TRACK_RADIUS_CM = 2e-3  # the v3 floor (20 um), which applies to protons


def track_radius_cm(let_air_keV_um: float) -> float:
    """Gaussian track radius b (cm) for the LET in air, as in IonTracks v3 (20 um floor)."""
    if not let_air_keV_um > 0:
        raise ValueError("LET must be positive")
    b_cm = float(np.polyval(_RADIUS_FIT, math.log10(let_air_keV_um))) * 1e-3
    return max(b_cm, MIN_TRACK_RADIUS_CM)


@dataclass(frozen=True)
class GaussianTrack:
    """A track perpendicular to the electrodes: N0 ion pairs per cm, Gaussian radius b."""

    pairs_per_cm: float
    radius_cm: float

    def __post_init__(self):
        if not self.pairs_per_cm > 0 or not self.radius_cm > 0:
            raise ValueError("pairs_per_cm and radius_cm must be positive")

    @classmethod
    def from_stopping_power(
        cls, stopping_power_MeV_cm2_g: float, gas: Gas, radius_cm: float | None = None
    ) -> "GaussianTrack":
        """Track of a particle with the given mass stopping power in the gas.

        Without ``radius_cm`` the radius follows ``track_radius_cm`` for the LET in the gas.
        """
        let_keV_um = stopping_power_MeV_cm2_g * gas.density_g_cm3 * 0.1
        return cls(
            pairs_per_cm=float(gas.ion_pairs_per_cm(stopping_power_MeV_cm2_g)),
            radius_cm=radius_cm if radius_cm is not None else track_radius_cm(let_keV_um),
        )

    def let_air_keV_um(self, gas: Gas) -> float:
        return self.pairs_per_cm * gas.W_eV * 1e-7
