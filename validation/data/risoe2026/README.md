# Risø 2026 pulsed-beam recombination data

Source: the IonTracks-AMM repository, `main` at commit 538874c, `experimentalData/`. Both files
are copied unchanged. Used by `validation/benchmark_risoe2026.py`.

## `data_extrapolationChamber.csv`

Measured 31.03.2026 on the Varian linac at Risø. From the source README:
- 16 MeV electrons and 10 MV FFF photons, 50 MU per reading;
- an extrapolation chamber 71.4 cm from the virtual source (closer than the standard set-up);
- a 20 × 20 cm² field without applicator, at 1014.38 hPa.

The temperature is not recorded.

- 72 rows: 2 beams × 2 dose rates (16 MeV: 100 and 1000 MU/min; 10 FFF: 400 and 2400 MU/min) × 6
  gaps (0.5, 1, 2, 5, 8, 10 mm) × 3 fields (10, 20, 40 V/mm). There are no repeat readings.
- Raw charges: `Charge_positive_nC ` (note the trailing space in the header) and
  `Charge_negative_nC`, recorded to 0.05 nC.
- Derived columns:
  - `k_pol`;
  - `polarity_corrected_nC`, the mean of the two polarities;
  - `k_di`, an inverse-square correction;
  - `non_uniformity_corrected_nC`;
  - `electrons_s_cm3`, the charge / (e × volume × delivery time). It implies a collecting
    electrode of r = 15.0 mm (7.07 cm²).
- `k_di` is 1 + d/890 mm in this copy. The later, unmerged commit 5b30181 ("uniformity correction
  fix", branch `test-structued-pulses`) changes it to 1 + d/714 mm, which matches the stated
  71.4 cm. The raw charges are identical in both versions. The benchmark uses the raw charges and
  its own 1 + d/714 mm.
- At 16 MeV, 100 MU/min, 10 mm, 200 V, the polarity pair (Q⁺ 178.1, Q⁻ 172.2 nC) is far more
  asymmetric than any other reading at gaps ≥ 5 mm. It is kept.

## `pulses_per_MU.csv`

Measured 09.04.2026, a different session: pulses and mean pulse spacing per delivery for many
beams and dose rates, 20 × 20 cm² field.

The counted repetition rate never exceeds ~122 Hz. Wherever constant dose per pulse would need a
higher rate, the pulses per MU drop to about a third:
- 6 MV above 200 MU/min;
- 4 MV above 50 MU/min;
- 10 FFF above 400 MU/min.

So the counter appears to miss pulses above ~120 Hz. Independent counts on a TrueBeam support
this (White et al. 2026, J. Appl. Clin. Med. Phys. 27 e70670): 35.5 pulses per MU for 6 MV at
400 MU/min, where this file has 11.5, and 7.9 for 10 MV FFF, where it has 7.4.

For the electron beams the drop already appears at 600 MU/min (implied ~100 Hz). The benchmark
compares both readings of the high-dose-rate records (as recorded, and constant dose per pulse).
