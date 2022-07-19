import os
from pathlib import Path
import runpy
import sys
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

def test_example_results():

    # get reference dataframe
    ref_df = pd.read_csv('refIonTracks.csv')

    # remember current working directory 
    # as we will be later changing to to somewhere else 
    # and will need a way to come back here
    current_working_dir = os.getcwd()

    # do some hacking to fix sys path needed for proper imports
    example_script_dir = Path('..', 'hadrons')
    sys.path.append(str(example_script_dir))
    sys.path.append(str(Path(example_script_dir, 'cython_files')))

    # change the directory as we load some data files from a relative paths
    os.chdir(example_script_dir)

    # remove previously generated output
    generated_df_path = Path('IonTracks.csv')
    generated_df_path.unlink() 

    # generate new dataframe using iontracks
    runpy.run_path(Path(example_script_dir, 'example_single_track.py'))
    df = pd.read_csv(generated_df_path)

    # reset current working directory to the original value
    os.chdir(current_working_dir)

    # check if reference dataframe is the same as the generated one
    assert_frame_equal(df, ref_df)