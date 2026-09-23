import pytest

from iontracks import Gas, ParallelPlateChamber

# Boag's classic theories: every electron attaches where it is produced, and the field is uniform
BOAG = {"free_electrons": False, "space_charge": False}


@pytest.fixture
def chamber():
    return ParallelPlateChamber(gap_cm=0.2, voltage_V=200.0, sensitive_radius_cm=0.5)


@pytest.fixture
def gas():
    return Gas.air(particle="electron")


@pytest.fixture
def gas_no_diffusion():
    """Boag's theories neglect diffusion: compare against them with this gas and BOAG models."""
    return Gas.air(particle="electron").with_(D_pos_cm2_s=0.0, D_neg_cm2_s=0.0)
