from math import exp, log, pi, sin, sqrt

import mpmath
import pandas as pd
from scipy.special import hankel1

from hadrons.utils.common import calculate_track_radius, get_LET_per_um
from hadrons.utils.consts import W, alpha, ion_diff, ion_mobility


def Jaffe_theory(
    x,
    voltage_V,
    electrode_gap_cm,
    input_is_LET=True,
    particle="proton",
    IC_angle_rad=0.0,
    **kwargs,
):
    """
    The Jaffe theory for initial recombination. Returns the inverse
    collection efficiency, i.e. the recombination correction factor
    """

    # provide either x as LET (keV/um) or enery (MeV/u), default is input_is_LET=True
    if not input_is_LET:
        LET_keV_um = get_LET_per_um(x, particle=particle)

    LET_eV_cm = LET_keV_um * 1e7
    electric_field = voltage_V / electrode_gap_cm

    # estimate the Gaussian track radius for the given LET
    b_cm = calculate_track_radius(LET_keV_um)

    N0 = LET_eV_cm / W
    g = alpha * N0 / (8.0 * pi * ion_diff)

    # ion track inclined with respect to the electric field?
    if abs(IC_angle_rad) > 0:
        x = (
            b_cm * ion_mobility * electric_field * sin(IC_angle_rad) / (2 * ion_diff)
        ) ** 2

        def nasty_function(y):
            order = 0.0
            if y < 1e3:
                # exp() overflows for larger y
                value = exp(y) * (1j * pi / 2) * hankel1(order, 1j * y)
            else:
                # approximation from Zankowski and Podgorsak (1998)
                value = sqrt(2.0 / (pi * y))
            return value

        f = 1.0 / (1 + g * nasty_function(x)).real

    else:
        """
        Pretty ugly function splitted up in three parts using mpmath package for precision
        """
        factor = (
            mpmath.exp(-1.0 / g)
            * ion_mobility
            * b_cm**2
            * electric_field
            / (2.0 * g * electrode_gap_cm * ion_diff)
        )
        first_term = mpmath.ei(
            1.0 / g
            + log(
                1.0
                + (
                    2.0
                    * electrode_gap_cm
                    * ion_diff
                    / (ion_mobility * b_cm**2 * electric_field)
                )
            )
        )
        second_term = mpmath.ei(1.0 / g)
        f = factor * (first_term - second_term)

    result_dic = {
        "particle": particle,
        "LET_keV_um": LET_keV_um,
        "voltage_V": voltage_V,
        "electrode_gap_cm": electrode_gap_cm,
        "IC_angle_rad": IC_angle_rad,
        "ks_Jaffe": float(1 / f),
    }

    if not input_is_LET:
        result_dic["E_MeV_u"] = x

    return pd.DataFrame([result_dic])
