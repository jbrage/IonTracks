import pytest

from iontracks import Gas, ParallelPlateChamber


@pytest.fixture
def chamber():
    return ParallelPlateChamber(gap_cm=0.2, voltage_V=200.0, sensitive_radius_cm=0.5)


@pytest.fixture
def gas():
    return Gas.air()


@pytest.fixture
def gas_no_diffusion():
    """Boag's theories neglect diffusion; use this gas to compare against them."""
    return Gas.air().with_(D_pos_cm2_s=0.0, D_neg_cm2_s=0.0)
