"""Jaffe's theory of initial (columnar) recombination for a track parallel to the field.

Jaffe G (1913) Ann. Phys. 347 303; the form for a column parallel to the field of a plane-parallel
chamber (track perpendicular to the electrodes) is the one used by Kanai et al. (1998) and
IonTracks v3:

    f = exp(-1/g) / (g X) * [Ei(1/g + ln(1 + X)) - Ei(1/g)],
    g = alpha N0 / (8 pi D),    X = 2 d D / (mu b^2 E),

for a Gaussian column of N0 ion pairs per cm and radius b. The theory assumes one mobility and one
diffusion coefficient for both species ("prescribed diffusion": the column keeps its Gaussian
shape), no free electrons and no space charge. For a gas with separate values the means are used.
"""

import math

from ..chamber import ParallelPlateChamber
from ..gas import Gas
from ..tracks import GaussianTrack

_EULER_GAMMA = 0.5772156649015329


def _exp_ei(x: float) -> float:
    """exp(-x) Ei(x) for x > 0 (series below 40, asymptotic series above)."""
    if not x > 0:
        raise ValueError("need x > 0")
    if x > 40.0:
        total, term, k = 0.0, 1.0 / x, 0
        while abs(term) > 1e-17 * abs(total) and k < 60:
            total += term
            k += 1
            term *= k / x
        return total
    total, term, k = 0.0, 1.0, 0
    while True:
        k += 1
        term *= x / k
        add = term / k
        total += add
        if add < 1e-17 * total:
            break
    return math.exp(-x) * (_EULER_GAMMA + math.log(x) + total)


def jaffe_parameters(
    track: GaussianTrack, chamber: ParallelPlateChamber, gas: Gas
) -> tuple[float, float]:
    """(g, X) of Jaffe's theory for this track, chamber and gas."""
    mu = 0.5 * (gas.mu_pos_cm2_Vs + gas.mu_neg_cm2_Vs)
    diff = 0.5 * (gas.D_pos_cm2_s + gas.D_neg_cm2_s)
    g = gas.alpha_cm3_s * track.pairs_per_cm / (8.0 * math.pi * diff)
    x = 2.0 * chamber.gap_cm * diff / (mu * track.radius_cm**2 * chamber.field_V_cm)
    return g, x


def jaffe_parallel_f(track: GaussianTrack, chamber: ParallelPlateChamber, gas: Gas) -> float:
    """Collection efficiency of one track perpendicular to the electrodes (Jaffe)."""
    g, x = jaffe_parameters(track, chamber, gas)
    if g == 0:
        return 1.0
    a = 1.0 / g
    return ((1.0 + x) * _exp_ei(a + math.log1p(x)) - _exp_ei(a)) / (g * x)
