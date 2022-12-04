import numpy as np
import pandas as pd
import itertools

# set parameters
TEST_DATA_DICT = dict(
    E_MeV_u=[10, 250],
    voltage_V=[50, 300],
    electrode_gap_cm=[0.01],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton", "carbon"],
    grid_size_um=[5, 10],
)

SLOW_TEST_DATA_DICT = dict(
    E_MeV_u=[10, 250],
    voltage_V=[50, 300],
    electrode_gap_cm=[0.5],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton", "carbon"],
    grid_size_um=[5, 10],
)


def create_matrix(data_dict):
    return pd.DataFrame.from_records(data=itertools.product(*data_dict.values()),
                                     columns=data_dict.keys())


# create a data frame with all the variables (combinations matrix)
MATRIX_DF = create_matrix(TEST_DATA_DICT)


SLOW_MATRIX_DF = create_matrix(SLOW_TEST_DATA_DICT)
