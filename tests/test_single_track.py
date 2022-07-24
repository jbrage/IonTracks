import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tests.ks_initial.testing_parameters import MATRIX_DF, TEST_DATA_DICT
from hadrons.functions import E_MeV_u_to_LET_keV_um, calc_b_cm, ks_initial_IonTracks
from initial_recombination import single_track_PDEsolver

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


def test_single_track_PDEsolver(single_track_PDEsolver_input, expected_result):
    calculated_result = single_track_PDEsolver(*single_track_PDEsolver_input)
    
    assert np.allclose(expected_result['ks_Jaffe'], calculated_result)


def test_ks_initial_IonTracks(expected_result):
    # DISABLED FOR NOW, fihish testing lower level functions first

    IonTracks_df = pd.DataFrame()

    for _, data in MATRIX_DF.iterrows():
        temp_df = ks_initial_IonTracks(**data,
                                       RDD_model="Gauss")

        IonTracks_df = pd.concat([IonTracks_df, temp_df], ignore_index=True)

    assert np.allclose(IonTracks_df.values, expected_result)
