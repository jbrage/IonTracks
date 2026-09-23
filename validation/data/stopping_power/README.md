# Proton stopping powers

`stopping_power_air.csv` and `stopping_power_water.csv` are unchanged copies of
`legacy/hadrons/data_LET/` (IonTracks v3). They were generated with libamtrack on 15.03.2022.

- Column `proton_LET_keV_um` is the unrestricted electronic stopping power in keV/µm at the
  medium's density: 1.20479e-3 g/cm³ for air, 1 g/cm³ for water.
- Mass stopping power in MeV cm²/g = value × 10 / density.

They are used by `validation/benchmark_dcpt2019.py`. They are copied here so the validation
doesn't depend on `legacy/`.
