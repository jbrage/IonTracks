# IonTracks v3 single-track results

`single_track_ks.csv` is an unchanged copy of `legacy/tests/refIonTracks.csv`: k_s of one track
computed by the v3 (legacy) single-track solver. It covers protons and carbon ions at 60 and
250 MeV/u, a 1 mm gap, and 200 and 300 V.

- Gas constants: one mobility 1.65 cm²/Vs, D = 0.037 cm²/s, α = 1.6e-6 cm³/s, W = 33.9 eV
  (`Gas.air_legacy()`).
- Track: Gaussian radius from the v3 LET table.
- `LET_keV_um`: the LET in air at air density.

These are model results, used as a code-to-code comparison by `validation/validate_tile3d.py`.
They are copied here so the validation doesn't depend on `legacy/`.
