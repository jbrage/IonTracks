import numpy as np
import pandas as pd
import itertools

# set parameters
data_dict = dict(
    E_MeV_u=10*np.arange(1, 26, 2),
    voltage_V=50*np.arange(1, 7),
    electrode_gap_cm=[0.2],
    # currently jaffe theory crashes for oxygen and helium
    particle=["proton", "carbon"],
    grid_size_um=np.arange(1, 11),
)

# create a data frame with all the variables (combinations matrix)
MATRIX_DF = pd.DataFrame.from_records(data=itertools.product(*data_dict.values()),
                                      columns=data_dict.keys())
