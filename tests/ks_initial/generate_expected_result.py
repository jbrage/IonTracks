import numpy as np
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from hadrons.functions import Jaffe_theory
from testing_parameters import MATRIX_DF, TEST_DATA_DICT

# absolute path of the file as string
ABS_PATH = str(Path(__file__).parent.absolute())


def calculate_expected_df():
    result_df = pd.DataFrame()
    for _, data in MATRIX_DF.iterrows():
        Jaffe_df = Jaffe_theory(x=data.E_MeV_u, **data, input_is_LET=False)
        result_df = pd.concat([result_df, Jaffe_df], ignore_index=True)

    return result_df


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--override', type=bool, default=False, help='Override previous generation')
    return parser


def results_file_exist():
    try:
        np.load(Path(ABS_PATH,'expected.npy'), allow_pickle=True)
        return True
    except FileNotFoundError:
        return False


def main():
    parser = configure_parser()
    args = parser.parse_args()

    if not args.override and results_file_exist():
        return

    np.save(f'{ABS_PATH}/expected.npy', calculate_expected_df().to_records())


if __name__ == '__main__':
    main()
    sys.exit(0)
