from pytest import fixture
import numpy as np
from pathlib import Path


@fixture
def expected_result():
    return np.load(f'{Path(__file__).absolute()}/expected.npy', allow_pickle=True)


def test_ks_initial_IonTracks():
    pass
