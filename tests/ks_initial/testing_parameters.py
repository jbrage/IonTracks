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

TEST_GAP_DATA_DICT = dict(
    E_MeV_u=[250],
    voltage_V=[300],
    electrode_gap_cm=[0.1, 0.5, 1, 1.5, 2],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton"],
    grid_size_um=[10],
)

TEST_GRID_SIZE_DATA_DICT = dict(
    E_MeV_u=[250],
    voltage_V=[300],
    electrode_gap_cm=[0.5],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton"],
    grid_size_um=[2, 2.5, 3, 3.5, 4],
)

TEST_E_DATA_DICT = dict(
    E_MeV_u=[10, 50, 100, 150, 200, 250],
    voltage_V=[300],
    electrode_gap_cm=[0.5],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton"],
    grid_size_um=[10],
)

TEST_V_DATA_DICT = dict(
    E_MeV_u=[250],
    voltage_V=[50, 100, 150, 200, 250, 300],
    electrode_gap_cm=[0.5],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton"],
    grid_size_um=[10],
)

def create_matrix(data_dict):
    return pd.DataFrame.from_records(data=itertools.product(*data_dict.values()),
                                      columns=data_dict.keys())


# create a data frame with all the variables (combinations matrix)
MATRIX_DF = create_matrix(TEST_DATA_DICT)


SLOW_MATRIX_DF = create_matrix(SLOW_TEST_DATA_DICT)

TEST_DATA_FRAMES = [
    create_matrix(TEST_GAP_DATA_DICT),
    create_matrix(TEST_GRID_SIZE_DATA_DICT),
    create_matrix(TEST_E_DATA_DICT),
    create_matrix(TEST_V_DATA_DICT),
]