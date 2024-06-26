import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from hadrons.functions import Jaffe_theory
from testing_parameters import MATRIX_DF

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
    parser.add_argument(
        "--override",
        type=bool,
        default=False,
        help="Override previously generated reference data",
    )
    return parser


def results_file_exist():
    try:
        np.load(Path(ABS_PATH, "expected.npy"), allow_pickle=True)
        return True
    except FileNotFoundError:
        return False


def main():
    parser = configure_parser()
    args = parser.parse_args()

    if not args.override and results_file_exist():
        return

    Path(ABS_PATH, "expected.npy").unlink(missing_ok=True)

    np.save(Path(ABS_PATH, "expected.npy"), calculate_expected_df().to_records())


if __name__ == "__main__":
    main()
    sys.exit(0)
