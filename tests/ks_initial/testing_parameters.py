import numpy as np
import pandas as pd
import itertools

# set parameters
TEST_DATA_DICT = dict(
    E_MeV_u=10*np.arange(1, 2, 2),
    voltage_V=50*np.arange(1, 2),
    electrode_gap_cm=[0.2],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton"],
    grid_size_um=np.arange(1, 2),
)

# create a data frame with all the variables (combinations matrix)
MATRIX_DF = pd.DataFrame.from_records(data=itertools.product(*TEST_DATA_DICT.values()),
                                      columns=TEST_DATA_DICT.keys())
