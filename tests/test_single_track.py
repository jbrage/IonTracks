import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from ks_initial.testing_parameters import MATRIX_DF
from hadrons.functions import ks_initial_IonTracks


@pytest.fixture
def expected_result():
    return np.load(f'{Path(__file__).absolute()}/expected.npy', allow_pickle=True)


def test_single_track_PDEsolver():
    pass


def test_ks_initial_IonTracks(expected_result):
    # DISABLED FOR NOW, fihish testing lower level functions first

    # IonTracks_df = pd.DataFrame()

    # for _, data in MATRIX_DF.iterrows():
    #     temp_df = ks_initial_IonTracks(**data,
    #                                    RDD_model="Gauss")

    #     IonTracks_df = pd.concat([IonTracks_df, temp_df], ignore_index=True)

    # assert np.allclose(IonTracks_df.values, expected_result)
    pass
