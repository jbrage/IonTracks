"""
conftest.py
"""

import pytest
import numpy as np
from pathlib import Path
from tests.ks_initial.testing_parameters import TEST_DATA_DICT
from hadrons.functions import E_MeV_u_to_LET_keV_um, calc_b_cm


@pytest.fixture
def expected_result():
    return np.load(Path(Path(__file__).parent.absolute(), 'ks_initial', 'expected.npy'), allow_pickle=True)
