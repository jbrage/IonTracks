"""Synthetic beam deliveries for tests, examples and method comparisons.

``cyclotron_pbs_field`` builds a spot-scanned proton field as delivered by an isochronous
cyclotron:

* quasi-continuous current during each spot (the RF micro-structure is far below any time step;
  ``rf_bunched`` builds it explicitly to demonstrate that);
* spot dwell set by the dose-per-spot and the nozzle current;
* dead times between spots and rows;
* optional spot-to-spot current jitter. The dose per spot is fixed by the monitor, so jitter
  changes the dwell times, not the dose.
"""

import numpy as np

from .constants import ELEMENTARY_CHARGE_C, MEV_TO_J
from .field import Field
from .time_structure import PiecewiseConstant

# 250 MeV protons: unrestricted mass stopping power from the libamtrack table shipped with
# IonTracks v3 (legacy/hadrons/data_LET), air converted with rho = 1.20479e-3 g/cm^3.
PROTON_250MEV_S_AIR_MEV_CM2_G = 3.446
PROTON_250MEV_S_WATER_MEV_CM2_G = 3.893
# ICRU Report 90: W_air for protons (high-energy asymptote).
W_AIR_PROTONS_EV = 34.44


def protons_per_spot(
    dose_to_water_Gy: float, spot_spacing_cm: float, stopping_power_water_MeV_cm2_g: float
) -> float:
    """Protons per spot giving ``dose_to_water_Gy`` in the plateau of a large uniform raster."""
    fluence_cm2 = dose_to_water_Gy / (stopping_power_water_MeV_cm2_g * MEV_TO_J * 1e3)
    return fluence_cm2 * spot_spacing_cm**2


def cyclotron_pbs_field(
    *,
    field_size_cm: float = 10.0,
    spot_spacing_cm: float = 0.5,
    sigma_cm: float = 0.35,
    dose_to_water_Gy: float = 2.0,
    beam_current_nA: float = 2.0,
    spot_switch_s: float = 1.0e-3,
    line_switch_s: float = 2.0e-3,
    current_jitter: float = 0.0,
    stopping_power_air_MeV_cm2_g: float = PROTON_250MEV_S_AIR_MEV_CM2_G,
    stopping_power_water_MeV_cm2_g: float = PROTON_250MEV_S_WATER_MEV_CM2_G,
    seed: int = 0,
) -> Field:
    """Single-energy square raster, scanned row by row in a serpentine pattern.

    The field is centred on (0, 0). Spot positions run from -field_size/2 to +field_size/2.
    """
    n_side = int(round(field_size_cm / spot_spacing_cm)) + 1
    coords = (np.arange(n_side) - (n_side - 1) / 2) * spot_spacing_cm
    xs, ys = [], []
    for row, y in enumerate(coords):
        for x in coords if row % 2 == 0 else coords[::-1]:
            xs.append(x)
            ys.append(y)
    n_spots = len(xs)

    n_per_spot = protons_per_spot(dose_to_water_Gy, spot_spacing_cm, stopping_power_water_MeV_cm2_g)
    protons_per_s = beam_current_nA * 1e-9 / ELEMENTARY_CHARGE_C
    rng = np.random.default_rng(seed)
    factor = 1.0 + current_jitter * rng.standard_normal(n_spots)
    dwell = n_per_spot / (protons_per_s * np.clip(factor, 0.5, None))

    new_row = np.zeros(n_spots, dtype=bool)
    new_row[n_side::n_side] = True
    pause = np.where(new_row, line_switch_s, spot_switch_s)
    pause[0] = 0.0
    t_on = np.cumsum(pause + np.concatenate(([0.0], dwell[:-1])))
    return Field.from_spots(
        t_on_s=t_on,
        t_off_s=t_on + dwell,
        x_cm=np.array(xs),
        y_cm=np.array(ys),
        n_particles=n_per_spot,
        sigma_cm=sigma_cm,
        stopping_power_MeV_cm2_g=stopping_power_air_MeV_cm2_g,
    )


def rf_bunched(
    rate: float,
    t_start_s: float,
    t_end_s: float,
    rf_frequency_Hz: float = 72.85e6,
    duty_cycle: float = 0.1,
) -> PiecewiseConstant:
    """Replace a constant ``rate`` on [t_start, t_end) by RF bunches carrying the same charge.

    Each RF period holds one rectangular bunch of length duty_cycle / f_RF.
    """
    period = 1.0 / rf_frequency_Hz
    starts = np.arange(t_start_s, t_end_s, period)
    width = duty_cycle * period
    ends = np.minimum(starts + width, t_end_s)
    keep = ends > starts
    total = rate * (t_end_s - t_start_s)
    widths = ends[keep] - starts[keep]
    return PiecewiseConstant.from_intervals(
        starts[keep], ends[keep], total / widths.sum() * np.ones(widths.size)
    )
