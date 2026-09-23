"""Ion recombination for a synthetic 250 MeV proton PBS field from an isochronous cyclotron.

Chamber: plane-parallel, 2 mm electrode gap, 200 V, sensitive radius 7.8 mm (Roos-like), at the
centre of the field. Field: 10 x 10 cm^2 single-energy raster, 5 mm spot spacing, sigma 3.5 mm,
2 Gy to water, serpentine scanning with 1 ms spot switching and 2 ms line switching. The
nozzle current sets the spot dwell times (dose per spot fixed by the monitor).

For every nozzle current the script prints:

* the time-resolved k_s and the quasi-static Boag estimate (Boag continuous applied to every
  constant segment);
* the peak local dose rate;
* the space-charge parameter eta. The model has a uniform field, so results with eta >~ 1 are
  outside its validity.

Run:  python examples/cyclotron_250MeV.py [--save spots.csv]
"""

import argparse
import time

import numpy as np

from iontracks import Gas, Grid1D, ParallelPlateChamber, Uniform1D, solve_field
from iontracks.analytic import boag_quasi_static_loss, space_charge_parameter
from iontracks.synthetic import (
    PROTON_250MEV_S_AIR_MEV_CM2_G,
    PROTON_250MEV_S_WATER_MEV_CM2_G,
    W_AIR_PROTONS_EV,
    cyclotron_pbs_field,
)

CURRENTS_NA = (1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 300.0, 800.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", metavar="CSV", help="write the 2 nA synthetic spot log to CSV")
    args = parser.parse_args()

    chamber = ParallelPlateChamber(gap_cm=0.2, voltage_V=200.0, sensitive_radius_cm=0.78)
    gas = Gas.air(W_eV=W_AIR_PROTONS_EV)
    model = Uniform1D(Grid1D(n_cells=200))
    points = chamber.sample_points(n_rings=4, n_azimuth=8)
    s_w_air = PROTON_250MEV_S_WATER_MEV_CM2_G / PROTON_250MEV_S_AIR_MEV_CM2_G
    pairs_per_gy_air = gas.ion_pairs_per_cm3_per_Gy()
    t_c = max(chamber.collection_times_s(gas))
    print(
        f"chamber: d = 2 mm, 200 V, t_c(+) = {t_c * 1e6:.0f} us; gas: {gas.name}, W = {gas.W_eV} eV"
    )
    print(f"{len(points)} sample points over r <= 7.8 mm; model: {model.grid.n_cells} cells\n")

    header = (
        f"{'I_nozzle':>9} {'dwell':>9} {'field':>8} {'Dmean':>8} {'Dpeak':>9} {'eta':>6}"
        f" {'k_s':>9} {'k_s(QS)':>9} {'(ks-1)/QS':>9} {'CPU':>6}"
    )
    units = (
        f"{'nA':>9} {'ms':>9} {'s':>8} {'Gy/s':>8} {'Gy/s':>9} {'peak':>6}"
        f" {'':>9} {'Boag':>9} {'':>9} {'s':>6}"
    )
    if args.save:
        _save(cyclotron_pbs_field(beam_current_nA=2.0), args.save)
    print(header + "\n" + units)
    for current in CURRENTS_NA:
        field = cyclotron_pbs_field(beam_current_nA=current)
        t0 = time.perf_counter()
        res = solve_field(field, chamber, gas, model, points=points)
        cpu = time.perf_counter() - t0

        produced = recombined = 0.0
        peak_rate = 0.0
        for x, y, w in zip(points.x_cm, points.y_cm, points.weight_cm2, strict=True):
            src = field.local_source(x, y, gas)
            p, r = boag_quasi_static_loss(src, chamber, gas)
            produced += w * p
            recombined += w * r
            peak_rate = max(peak_rate, src.max_rate)
        ks_qs = produced / (produced - recombined)
        dose_rate_peak_w = peak_rate / pairs_per_gy_air * s_w_air
        dose_rate_mean_w = 2.0 / field.t_end_s
        eta = space_charge_parameter(peak_rate, chamber, gas)
        dwell = field.t_off_s[0] - field.t_on_s[0]
        flag = "" if eta < 1 else "  <- eta > 1: space charge not modelled"
        print(
            f"{current:9.1f} {dwell * 1e3:9.3f} {field.t_end_s:8.3f} {dose_rate_mean_w:8.3f}"
            f" {dose_rate_peak_w:9.1f} {eta:6.2f} {res.ks:9.5f} {ks_qs:9.5f}"
            f" {(res.ks - 1) / (ks_qs - 1):9.3f} {cpu:6.1f}{flag}"
        )
        for w in res.warnings:
            print("   warning:", w)

    print(
        "\nDmean: field-average dose rate to water. Dpeak: highest local dose rate to water in"
        "\nthe chamber (spot centre passing). k_s(QS): Boag continuous applied to each constant"
        "\nsegment. k_s is general recombination only (no track structure, no free electrons,"
        "\nno space charge)."
    )


def _save(field, path):
    header = "t_on_s,t_off_s,x_cm,y_cm,sigma_x_cm,sigma_y_cm,n_protons,S_air_MeV_cm2_g"
    table = np.column_stack(
        [
            field.t_on_s,
            field.t_off_s,
            field.x_cm,
            field.y_cm,
            field.sigma_x_cm,
            field.sigma_y_cm,
            field.n_particles,
            field.stopping_power_MeV_cm2_g,
        ]
    )
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.9g")
    print(f"(synthetic spot log for 2 nA written to {path})")


if __name__ == "__main__":
    main()
