"""Relations between the analytic parameters (Boag u and xi^2, the space-charge parameters)."""

import pytest

from iontracks import Gas, ParallelPlateChamber
from iontracks.analytic import (
    boag_continuous_xi2,
    boag_pulsed_u,
    space_charge_parameter,
    space_charge_parameter_pulsed,
)
from iontracks.constants import ELEMENTARY_CHARGE_C, VACUUM_PERMITTIVITY_F_CM


@pytest.mark.parametrize("gap_cm, voltage_V", [(0.2, 200.0), (0.05, 20.0), (1.0, -100.0)])
def test_pulsed_space_charge_is_a_fixed_multiple_of_u(gap_cm, voltage_V):
    gas = Gas.air()
    chamber = ParallelPlateChamber(gap_cm, voltage_V)
    ratio = space_charge_parameter_pulsed(1e9, chamber) / boag_pulsed_u(1e9, chamber, gas)
    expected = (
        ELEMENTARY_CHARGE_C
        * (gas.mu_pos_cm2_Vs + gas.mu_neg_cm2_Vs)
        / (VACUUM_PERMITTIVITY_F_CM * gas.alpha_cm3_s)
    )
    assert ratio == pytest.approx(expected, rel=1e-12)
    assert ratio == pytest.approx(3.9, abs=0.05)  # air, Boag's constants


def test_continuous_space_charge_is_a_fixed_multiple_of_xi2():
    gas = Gas.air()
    chamber = ParallelPlateChamber(0.2, 200.0)
    ratio = space_charge_parameter(1e12, chamber, gas) / boag_continuous_xi2(1e12, chamber, gas)
    assert ratio == pytest.approx(
        6 * ELEMENTARY_CHARGE_C * gas.mu_neg_cm2_Vs / (VACUUM_PERMITTIVITY_F_CM * gas.alpha_cm3_s),
        rel=1e-12,
    )
    assert ratio == pytest.approx(14.2, abs=0.1)
