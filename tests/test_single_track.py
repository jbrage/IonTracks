"""
test_single_track.py
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tests.ks_initial.testing_parameters import TEST_DATA_DICT
from hadrons.cython_files.initial_recombination import single_track_PDEsolver as cython_single_track_PDEsolver
from hadrons.initial_recombination import single_track_PDEsolver as python_single_track_PDEsolver
from hadrons.functions import E_MeV_u_to_LET_keV_um, calc_b_cm
from hadrons.solver import SolverType, solvePDE


def get_PDEsolver_input(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um):
    
    LET_keV_um = E_MeV_u_to_LET_keV_um(E_MeV_u, particle)
    track_radius_cm = calc_b_cm(LET_keV_um)
    
    return (dict(
        LET_keV_um = float(LET_keV_um),
        voltage_V = voltage_V,
        IC_angle_rad = 0,
        electrode_gap_cm = electrode_gap_cm,
        E_MeV_u = E_MeV_u,
        a0_nm = 8.0,
        RDD_model = "Gauss",
        unit_length_cm = grid_size_um*1e-4,
        track_radius_cm = track_radius_cm,
    ))

@pytest.mark.parametrize("E_MeV_u", TEST_DATA_DICT["E_MeV_u"])
@pytest.mark.parametrize("voltage_V", TEST_DATA_DICT["voltage_V"])
@pytest.mark.parametrize("electrode_gap_cm", TEST_DATA_DICT["electrode_gap_cm"])
@pytest.mark.parametrize("particle", TEST_DATA_DICT["particle"])
@pytest.mark.parametrize("grid_size_um", TEST_DATA_DICT["grid_size_um"])
def test_single_track_PDEsolver_cython(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um, expected_result):
    single_track_PDEsolver_input = get_PDEsolver_input(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um)
    calculated_result = solvePDE(single_track_PDEsolver_input, SolverType.CYTHON)

    assert calculated_result is not None
    assert isinstance(calculated_result, float)
    # TODO: Add resonable range sanity test here
    # assert calculated_result >= reasonable_minimum and calculated_result <= reasonable_maximum

    def row_filter(row):    
        return (
            row.particle == particle and
            np.allclose(row.LET_keV_um, single_track_PDEsolver_input['LET_keV_um']) and
            row.voltage_V == single_track_PDEsolver_input['voltage_V'] and
            row.electrode_gap_cm == single_track_PDEsolver_input['electrode_gap_cm'] and
            row.IC_angle_rad == single_track_PDEsolver_input['IC_angle_rad'] and
            row.E_MeV_u == single_track_PDEsolver_input['E_MeV_u']
        )

    expected = expected_result[[idx for idx, row in enumerate(expected_result) if row_filter(row)]]

    print(calculated_result)

    assert len(expected) > 0
    
    assert np.allclose(expected[0]['ks_Jaffe'], calculated_result, rtol=1e-3)

@pytest.mark.parametrize("E_MeV_u", TEST_DATA_DICT["E_MeV_u"])
@pytest.mark.parametrize("voltage_V", TEST_DATA_DICT["voltage_V"])
@pytest.mark.parametrize("electrode_gap_cm", TEST_DATA_DICT["electrode_gap_cm"])
@pytest.mark.parametrize("particle", TEST_DATA_DICT["particle"])
@pytest.mark.parametrize("grid_size_um", TEST_DATA_DICT["grid_size_um"])
def test_single_track_PDEsolver_python(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um, expected_result):
    LET_keV_um = E_MeV_u_to_LET_keV_um(E_MeV_u, particle)
    track_radius_cm = calc_b_cm(LET_keV_um)

    calculated_result = solvePDE(dict(
            E_MeV_u = E_MeV_u,
            voltage_V = voltage_V,
            electrode_gap_cm = electrode_gap_cm,
            LET_keV_um = float(LET_keV_um),
            a0_nm = 8.0,
            RDD_model = "Gauss",
            IC_angle_rad = 0,
            unit_length_cm = grid_size_um*1e-4,
            track_radius_cm = track_radius_cm,
        ), SolverType.PYTHON)

    assert calculated_result is not None
    assert isinstance(calculated_result, float)
    # TODO: Add resonable range sanity test here
    # assert calculated_result >= reasonable_minimum and calculated_result <= reasonable_maximum

    def row_filter(row):    
        return (
            row.particle == particle and
            np.allclose(row.LET_keV_um, LET_keV_um) and
            row.voltage_V == voltage_V and
            row.electrode_gap_cm == electrode_gap_cm and
            row.IC_angle_rad == 0 and
            row.E_MeV_u == E_MeV_u
        )

    expected = expected_result[[idx for idx, row in enumerate(expected_result) if row_filter(row)]]

    print(calculated_result)

    assert len(expected) > 0
    
    assert np.allclose(expected[0]['ks_Jaffe'], calculated_result, rtol=1e-3)