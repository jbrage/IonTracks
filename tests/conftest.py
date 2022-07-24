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
    return np.load(f'{Path(__file__).parent.absolute()}/ks_initial/expected.npy', allow_pickle=True)

@pytest.mark.parametrize("E_MeV_u", TEST_DATA_DICT["E_MeV_u"])
@pytest.mark.parametrize("voltage_V", TEST_DATA_DICT["voltage_V"])
@pytest.mark.parametrize("electrode_gap_cm", TEST_DATA_DICT["electrode_gap_cm"])
@pytest.mark.parametrize("particle", TEST_DATA_DICT["particle"])
@pytest.mark.parametrize("grid_size_um", TEST_DATA_DICT["grid_size_um"])
@pytest.fixture
def single_track_PDEsolver_input(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um):
    
    LET_keV_um = E_MeV_u_to_LET_keV_um(E_MeV_u, particle)
    track_radius_cm = calc_b_cm(LET_keV_um)
    
    return ({
        "E_MeV_u": E_MeV_u,
        "voltage_V": voltage_V,
        "electrode_gap_cm": electrode_gap_cm,
        "LET_keV_um": float(LET_keV_um),
        "a0_nm": 8.0,
        "particle": particle,
        "RDD_model": "Gauss",
        "IC_angle_rad": 0,
    },{
        "unit_length_cm": grid_size_um*1e-4,
        "track_radius_cm": track_radius_cm,
        "SHOW_PLOT": False,
        "PRINT_parameters": False,
    })
