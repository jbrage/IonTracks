# DCPT 2019 recombination data

Source: branch `origin/backend-benchmark` (commit e3c2721),
`benchmark/2019_DCPT/experimental-data-200V/`. Measurements are described in
Christensen J B et al 2020 *Mapping initial and general recombination in scanning proton pencil
beams*, Phys. Med. Biol. 65 115003 (Section 2.1.2, DCPT).

- Chamber: **Roos-type** plane-parallel, 2 mm electrode spacing, per the paper. The source README
  says "Advanced Markus", which is inconsistent: the PTW Advanced Markus has a 1 mm gap.
- Beam: Varian ProBeam, 10 × 10 cm² single-energy layers of 41 × 41 spots at 2 cm water depth.
  Energies 70, 150, 226 and 244 MeV; dose-rate factors 0.2, 0.5 and 1.0.

Files:

- `charges.csv`: raw charges (nC), three readings per voltage (50, 67, 83, 125, 200 V), with T and
  p. Parsed from `data_raw/*.dat`. Two lab notes are kept in `note`:
  - one reading (150 MeV, DRF 0.5, 125 V) is excluded following the note "might be wrong";
  - one setting (226 MeV, DRF 1, 200 V) has only two readings.
- `ks_200V.csv`: unchanged copy of `recombination_200V_data.csv`:
  - `time_s` is the spot duration from the machine logs;
  - the `dose_rate_*` columns equal D_field / (≈ 38 × t_spot) and differ from the effective dose
    rates in Fig. 3(a) of the paper;
  - the `k_s` extraction method is not documented. It is reproduced neither by TVM (200/67 V) nor
    by a 5-point 1/V² fit.
