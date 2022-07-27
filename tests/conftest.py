"""
conftest.py
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def expected_result():
    return np.load(Path(Path(__file__).parent.absolute(), 'ks_initial', 'expected.npy'), allow_pickle=True)
