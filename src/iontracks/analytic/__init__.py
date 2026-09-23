"""Analytic recombination theories used as references and validation oracles."""

from .boag import (
    boag_continuous_f,
    boag_continuous_xi2,
    boag_pulsed_f,
    boag_pulsed_free_electron_f,
    boag_pulsed_u,
    boag_quasi_static_loss,
    space_charge_parameter,
    space_charge_parameter_pulsed,
)
from .jaffe import jaffe_parallel_f, jaffe_parameters

__all__ = [
    "boag_continuous_f",
    "boag_continuous_xi2",
    "boag_pulsed_f",
    "boag_pulsed_free_electron_f",
    "boag_pulsed_u",
    "boag_quasi_static_loss",
    "jaffe_parallel_f",
    "jaffe_parameters",
    "space_charge_parameter",
    "space_charge_parameter_pulsed",
]
