"""Recombination in a plane-parallel chamber irradiated by a scanned proton field.

A 7 x 7 raster of spots is delivered spot by spot. Every lateral sample point of the chamber
sees its own ionisation history (the superposition of the passing Gaussian spots), and the
chamber result is the charge-weighted sum over the points.

Run:  python examples/scanned_field.py
"""

import numpy as np

from iontracks import Field, Gas, Grid1D, ParallelPlateChamber, Uniform1D, plan_field, solve_field

chamber = ParallelPlateChamber(gap_cm=0.2, voltage_V=200.0, sensitive_radius_cm=0.78)  # Roos-like
gas = Gas.air(particle="proton")  # W for protons; mu+ = 1.36, mu- = 2.1 cm^2/Vs; free electrons

# 7 x 7 raster, 5 mm spot spacing, sigma 4 mm, 2 ms dwell + 0.5 ms move per spot
n_side, spacing_cm = 7, 0.5
xs, ys = np.meshgrid(
    (np.arange(n_side) - n_side // 2) * spacing_cm, (np.arange(n_side) - n_side // 2) * spacing_cm
)
t_on = np.arange(n_side * n_side) * 2.5e-3
field = Field.from_spots(
    t_on_s=t_on,
    t_off_s=t_on + 2.0e-3,
    x_cm=xs.ravel(),
    y_cm=ys.ravel(),
    n_particles=3e8,
    sigma_cm=0.4,
    stopping_power_MeV_cm2_g=5.4,  # ~150 MeV protons in air
)

model = Uniform1D(Grid1D(n_cells=100))
points = chamber.sample_points(n_rings=3, n_azimuth=8)

_, plans = plan_field(field, chamber, gas, model, points=points)
print(f"{len(field)} spots, {len(points)} sample points; per point e.g.: {plans[0].summary()}")

result = solve_field(field, chamber, gas, model, points=points)
dose = np.mean([field.dose_to_gas_Gy(x, y) for x, y in zip(points.x_cm, points.y_cm, strict=True)])
print(f"mean dose to air in the chamber: {dose:.3f} Gy over {field.t_end_s * 1e3:.0f} ms")
print(f"k_s = {result.ks:.5f}   collected charge = {result.charge_C * 1e9:.4f} nC")
x, y, ks = result.ks_map()
print(f"k_s across the chamber area: {np.nanmin(ks):.5f} ... {np.nanmax(ks):.5f}")

edges = np.linspace(0.0, field.t_end_s + 1e-3, 200)
current = result.current_A(edges)
print(f"peak chamber current (0.6 ms bins): {current.max() * 1e9:.2f} nA")
for w in result.warnings:
    print("warning:", w)
