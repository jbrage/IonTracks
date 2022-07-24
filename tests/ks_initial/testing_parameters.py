import numpy as np
import pandas as pd
import itertools

# set parameters
TEST_DATA_DICT = dict(
    E_MeV_u=[10, 20, 30, 60, 150, 250],
    voltage_V=[50, 100, 200, 300],
    electrode_gap_cm=[0.2],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton", "carbon"],
    grid_size_um=[1, 5, 10],
)

# create a data frame with all the variables (combinations matrix)
MATRIX_DF = pd.DataFrame.from_records(data=itertools.product(*TEST_DATA_DICT.values()),
                                      columns=TEST_DATA_DICT.keys())
