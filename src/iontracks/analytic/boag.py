"""Boag's recombination theories for uniform ionisation (no diffusion, no space charge).

References: Boag J W (1950) Br. J. Radiol. 23, 601 (pulsed); Boag J W, "Ionization chambers",
in The Dosimetry of Ionizing Radiation vol. II (1987) (continuous); Boutillon M (1998)
Phys. Med. Biol. 43, 2061.
"""

import math

import numpy as np

from ..chamber import ParallelPlateChamber
from ..constants import ELEMENTARY_CHARGE_C, VACUUM_PERMITTIVITY_F_CM
from ..gas import Gas
from ..time_structure import PiecewiseConstant


def boag_pulsed_u(n0_cm3: float, chamber: ParallelPlateChamber, gas: Gas) -> float:
    """u = alpha n0 d^2 / ((mu+ + mu-) V) for an instantaneous uniform pulse (n0 pairs/cm^3)."""
    mu_sum = gas.mu_pos_cm2_Vs + gas.mu_neg_cm2_Vs
    return gas.alpha_cm3_s * n0_cm3 * chamber.gap_cm**2 / (mu_sum * abs(chamber.voltage_V))


def boag_pulsed_f(n0_cm3: float, chamber: ParallelPlateChamber, gas: Gas) -> float:
    """Collection efficiency ln(1 + u) / u. Exact for the model without diffusion."""
    u = boag_pulsed_u(n0_cm3, chamber, gas)
    return 1.0 if u == 0 else math.log1p(u) / u


def boag_continuous_xi2(q_cm3_s: float, chamber: ParallelPlateChamber, gas: Gas) -> float:
    """xi^2 = alpha q d^4 / (6 mu+ mu- V^2) for a constant uniform rate q (ion pairs/cm^3/s)."""
    mu_prod = gas.mu_pos_cm2_Vs * gas.mu_neg_cm2_Vs
    return gas.alpha_cm3_s * q_cm3_s * chamber.gap_cm**4 / (6.0 * mu_prod * chamber.voltage_V**2)


def boag_continuous_f(q_cm3_s: float, chamber: ParallelPlateChamber, gas: Gas) -> float:
    """Steady-state efficiency 1 / (1 + xi^2); exact to first order in xi^2."""
    return 1.0 / (1.0 + boag_continuous_xi2(q_cm3_s, chamber, gas))


def space_charge_parameter(q_cm3_s: float, chamber: ParallelPlateChamber, gas: Gas) -> float:
    """eta = e q d^4 / (eps0 mu+ V^2): steady-state space charge relative to the applied field.

    The field distortion is of order eta / 10. The present solvers assume a uniform field, so
    results are reliable only for eta << 1. Note eta / xi^2 = 6 e mu- / (eps0 alpha), about 14 in
    air: space charge becomes important at roughly the recombination level where Boag's
    continuous formula itself loses accuracy. For short pulses use
    ``space_charge_parameter_pulsed``.
    """
    return (
        ELEMENTARY_CHARGE_C
        * q_cm3_s
        * chamber.gap_cm**4
        / (VACUUM_PERMITTIVITY_F_CM * gas.mu_pos_cm2_Vs * chamber.voltage_V**2)
    )


def space_charge_parameter_pulsed(n0_cm3: float, chamber: ParallelPlateChamber) -> float:
    """e n0 d^2 / (eps0 V): field of one pulse's fully separated charge over the applied field.

    Use this for pulses much shorter than the ion transit time. There the continuous ``eta``
    evaluated at the peak rate overstates space charge, because the charge never builds up to
    its steady state. In air it is about 3.9 times Boag's u, since e (mu+ + mu-) / (eps0 alpha)
    = 3.9, so it exceeds 1 once u exceeds about 0.25.
    """
    return (
        ELEMENTARY_CHARGE_C
        * n0_cm3
        * chamber.gap_cm**2
        / (VACUUM_PERMITTIVITY_F_CM * abs(chamber.voltage_V))
    )


def boag_quasi_static_loss(
    source: PiecewiseConstant, chamber: ParallelPlateChamber, gas: Gas
) -> tuple[float, float]:
    """(produced, recombined) per cm^3 if every constant segment were in its Boag steady state.

    Valid when each segment lasts much longer than the ion collection time. It ignores the
    build-up and drain transients, which is exactly what the time-resolved solver adds.
    """
    widths = np.diff(source.edges)
    produced = source.rates * widths
    xi2 = boag_continuous_xi2(1.0, chamber, gas) * source.rates
    return float(produced.sum()), float(np.sum(produced * xi2 / (1.0 + xi2)))
