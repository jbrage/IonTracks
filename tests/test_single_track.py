"""
test_single_track.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tests.ks_initial.testing_parameters import MATRIX_DF, TEST_DATA_DICT
from hadrons.functions import ks_initial_IonTracks
from initial_recombination import single_track_PDEsolver
from tests.conftest import expected_result
from hadrons.functions import E_MeV_u_to_LET_keV_um, calc_b_cm


def get_PDEsolver_input(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um):
    
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

# @pytest.mark.parametrize("E_MeV_u", TEST_DATA_DICT["E_MeV_u"])
# @pytest.mark.parametrize("voltage_V", TEST_DATA_DICT["voltage_V"])
# @pytest.mark.parametrize("electrode_gap_cm", TEST_DATA_DICT["electrode_gap_cm"])
# @pytest.mark.parametrize("particle", TEST_DATA_DICT["particle"])
# @pytest.mark.parametrize("grid_size_um", TEST_DATA_DICT["grid_size_um"])
@pytest.mark.parametrize("E_MeV_u", [250])
@pytest.mark.parametrize("voltage_V", [50])
@pytest.mark.parametrize("electrode_gap_cm", [0.2])
@pytest.mark.parametrize("particle", ["proton"])
@pytest.mark.parametrize("grid_size_um", [5])
def test_single_track_PDEsolver(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um, expected_result):
    single_track_PDEsolver_input = get_PDEsolver_input(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um)
    calculated_result = single_track_PDEsolver(*single_track_PDEsolver_input)

    def row_filter(row):    
        return (
            row.particle == single_track_PDEsolver_input[0]['particle'] and
            np.allclose(row.LET_keV_um, single_track_PDEsolver_input[0]['LET_keV_um']) and
            row.voltage_V == single_track_PDEsolver_input[0]['voltage_V'] and
            row.electrode_gap_cm == single_track_PDEsolver_input[0]['electrode_gap_cm'] and
            row.IC_angle_rad == single_track_PDEsolver_input[0]['IC_angle_rad'] and
            row.E_MeV_u == single_track_PDEsolver_input[0]['E_MeV_u']
        )

    expected = expected_result[[idx for idx, row in enumerate(expected_result) if row_filter(row)]]

    print(calculated_result)

    assert len(expected) > 0
    
    assert np.allclose(expected[0]['ks_Jaffe'], calculated_result)


# def test_ks_initial_IonTracks(expected_result):
#     # DISABLED FOR NOW, fihish testing lower level functions first

#     IonTracks_df = pd.DataFrame()

#     for _, data in MATRIX_DF.iterrows():
#         temp_df = ks_initial_IonTracks(**data,
#                                        RDD_model="Gauss")

#         IonTracks_df = pd.concat([IonTracks_df, temp_df], ignore_index=True)

#     assert np.allclose(IonTracks_df.values, expected_result)
