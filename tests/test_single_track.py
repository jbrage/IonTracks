"""
test_single_track.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tests.ks_initial.testing_parameters import MATRIX_DF
from hadrons.functions import ks_initial_IonTracks
from initial_recombination import single_track_PDEsolver
from tests.conftest import single_track_PDEsolver_input, expected_result

def test_single_track_PDEsolver(single_track_PDEsolver_input, expected_result):
    calculated_result = single_track_PDEsolver(*single_track_PDEsolver_input)

    def row_filter(row):    
        return (
            row.particle == single_track_PDEsolver_input.particle and
            np.allclose(row.LET_keV_um, single_track_PDEsolver_input.LET_keV_um) and
            row.voltage_V == single_track_PDEsolver_input.voltage_V and
            row.electrode_gap_cm == single_track_PDEsolver_input.electrode_gap_cm and
            row.IC_angle_rad == single_track_PDEsolver_input.IC_angle_rad and
            row.E_MeV_u == single_track_PDEsolver_input.E_MeV_u
        )

    expected = expected_result[[idx for idx, row in enumerate(expected_result) if row_filter(row)]][0]
    
    assert np.allclose(expected['ks_Jaffe'], calculated_result)


def test_ks_initial_IonTracks(expected_result):
    # DISABLED FOR NOW, fihish testing lower level functions first

    IonTracks_df = pd.DataFrame()

    for _, data in MATRIX_DF.iterrows():
        temp_df = ks_initial_IonTracks(**data,
                                       RDD_model="Gauss")

        IonTracks_df = pd.concat([IonTracks_df, temp_df], ignore_index=True)

    assert np.allclose(IonTracks_df.values, expected_result)
