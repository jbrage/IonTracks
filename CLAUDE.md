# CLAUDE.md

IonTracks computes **ion recombination losses (k_s)** in air-filled parallel-plate ionization chambers for beams with an **arbitrary, user-defined time structure**. Examples: pulses, pulse trains, spills, and whole scanned (PBS) fields evaluated at lateral sample points of the chamber.

It solves drift, diffusion and volume recombination (α·n⁺·n⁻) of positive and negative ions across the gap. Free electrons and space charge are included. There are two point models:

- `Uniform1D`: laterally uniform ionisation (general recombination);
- `Tile3D`: resolves individual ion tracks (initial recombination, and tracks overlapping in time).

**Branch `v4` is a rewrite.** The previous code is tagged `v3-legacy` and lives read-only under `legacy/`. Background and decisions:

- `report/code_review_and_refactoring_plan.md`: legacy bugs, numerics, architecture, decisions (§7), whole-field design (§8);
- `report/literature_review.md`: theory, validation data, related models;
- `report/verification/verify_findings.py`: reproduces the legacy findings.

## Commands

```bash
pip install -e ".[dev,plot]"             # Python >= 3.10, Windows or Linux; no compiler needed
python -m pytest                         # ~6 s; the first run adds ~40 s of Numba JIT (cached afterwards)
ruff check src tests examples validation report && ruff format --check src tests examples validation report
python examples/scanned_field.py         # end-to-end field example (~2 s)
python examples/cyclotron_250MeV.py      # 250 MeV cyclotron PBS field, 2 mm / 200 V, current sweep (~12 s)
python validation/run_validation.py      # 29 checks, figs 1-5 and 8 -> validation/VALIDATION.md (~70 s)
python validation/validate_tile3d.py     # 3D track model: Jaffé, IonTracks v3, coupling (fig 9) -> VALIDATION_TILE3D.md (~2.5 min)
python validation/benchmark_dcpt2019.py  # blind benchmark, DCPT 2019 scanned protons -> BENCHMARK_DCPT2019.md (~2 min)
python validation/benchmark_risoe2026.py # blind benchmark, Risø 2026 pulsed e-/photons -> BENCHMARK_RISOE2026.md (~2 min)
python validation/diagnostics.py         # figures and GIFs of the carriers over time for both experiments -> DIAGNOSTICS.md (~2 min)
```

CI (`.github/workflows/main.yml`) runs lint, tests and both examples on ubuntu-latest and windows-latest with Python 3.10 and 3.13. A separate `validation` job (ubuntu, 3.13) runs `run_validation.py` and `validate_tile3d.py` (both exit non-zero if a check fails) and both benchmarks. The validation scripts need the `plot` extra (matplotlib, seaborn).

## Layout

- `src/iontracks/`
  - `gas.py`: `Gas`, with **separate µ⁺, µ⁻ and D⁺, D⁻** and optional free-electron data (electron mobility and attachment rate γ(E)).
    - `Gas.air(T_K, p_hPa, particle=...)` **requires the particle**, which sets W from ICRU 90 (`W_AIR_EV`: electron and photon 33.97, proton 34.44, carbon 34.71 eV). Pass `W_eV` for anything else.
    - It uses Boag's mobilities 1.36 and 2.1 cm²/Vs, density scaling, the Einstein relation, and the Götz free-electron data.
    - `Gas.air_legacy()` holds the v3 constants and has no free-electron data.
  - `chamber.py`: `ParallelPlateChamber` and `sample_points()` (equal-area quadrature over the sensitive disc).
  - `time_structure.py`: `PiecewiseConstant`, the one representation of every time structure. It has exact integrals, `support()`, `window()` and a `shape_key()` for deduplication.
  - `field.py`: `Field` (spot list). `local_source(x, y, gas)` returns the exact ion-pair rate history at a point.
  - `tracks.py`: `GaussianTrack` (N0 ion pairs per cm, Gaussian radius b) and `track_radius_cm(LET)` (the v3 table after Rossomme et al., floor 20 µm).
  - `planner.py`: splits the history into independent **episodes** (gaps > model memory time), groups identical episodes, and estimates cost.
  - `solvers/kernels.py`: the shared Numba kernels:
    - `react`: exact recombination;
    - `transport` / `transport_var`: drift in a uniform or distorted field, plus diffusion;
    - `solve_field`: 1D Gauss;
    - `attach`: free-electron attachment;
    - `outflow_courant`.
  - `solvers/uniform1d.py`: `Uniform1D(grid, free_electrons=True, space_charge=True)`, `Grid1D`, `EpisodeResult` and `Snapshots` (gap profiles at chosen times).
  - `solvers/tile3d.py`: `Tile3D(track, grid, free_electrons, space_charge, seed)`, `Grid3D` and `TrackSnapshots`.
    - It simulates a laterally periodic tile with Poisson tracks perpendicular to the plates.
    - `single_track()` gives initial recombination.
    - Space charge uses the laterally averaged (mean) field.
  - `results.py`: `PointResult` and `FieldResult` (charge-weighted k_s, current I(t), `ks_map`).
  - `api.py`: `solve_point`, `solve_field` and `plan_field` (**`gas` is required**), with a thread pool over points and a shared episode cache.
  - `analytic/boag.py`:
    - Boag pulsed and continuous;
    - `boag_pulsed_free_electron_f` (Fenwick & Kumar 2023 exact solution with free electrons);
    - `boag_quasi_static_loss`;
    - `space_charge_parameter` (continuous) and `space_charge_parameter_pulsed`.
  - `analytic/jaffe.py`: `jaffe_parallel_f` (Jaffé, a column parallel to the field, as in v3).
  - `synthetic.py`: synthetic deliveries (`cyclotron_pbs_field`, `rf_bunched`) and 250 MeV proton stopping powers.
- `tests/`: kernels, conservation and Ramo, analytic limits (Boag, Fenwick–Kumar, Jaffé), free electrons, space charge, the 3D model, planner equivalences, fields.
- `examples/`: runnable scripts (CI runs both).
- `validation/`:
  - `plotstyle.py`: the shared figure style;
  - `run_validation.py` and `validate_tile3d.py`: numerics and physics against theory;
  - `benchmark_*.py`: the blind benchmarks against measurements;
  - `diagnostics.py`: figures and GIFs of the carriers over time.

  Data and provenance READMEs are in `validation/data/`, including the proton stopping-power tables and the eight v3 single-track reference values, so nothing in `validation/` reads `legacy/`. Rerun the validation after any numerics change.
- `legacy/`: the v3 code, for reference and baselines only. Don't import it from `src/`, and don't lint it.

## Physics and units conventions

- Internal units are cm, s, V, eV. Densities are ion pairs per cm³, and gap-integrated tallies are **per cm² of electrode area**. Keep the unit-suffix naming (`gap_cm`, `voltage_V`, `t_on_s`, `stopping_power_MeV_cm2_g`).
- Positive ions drift to z = gap (cathode), negative ions and free electrons to z = 0 (anode). Air's slower species (+) sets the collection time t_c = d/(µ⁺E).
- **The default model has free electrons and space charge.**
  - Comparisons with Boag's classic theories need `free_electrons=False, space_charge=False` (`BOAG` in `tests/conftest.py`, `boag_model()` in `run_validation.py`).
  - Jaffé comparisons use `Gas.air_legacy()`, because Jaffé assumes one mobility and one diffusion coefficient.
- **Free electrons** attach within a step (~10 ns ≪ Δt), using an exact exponential kernel per cell. Those reaching the anode are collected directly. Electron–ion recombination is neglected.
- **Space charge:** 1D Gauss at a fixed voltage. With field reversal, ions can reach the "wrong" electrode (`backflow_*`), and the Ramo signal is then collected_pos − backflow_pos − backflow_neg.
- **Chamber memory** (`Uniform1D.memory_time_s`) is t_c + 8 diffusion widths, ×1.25. Beam-off gaps longer than that separate independent episodes.
- `ks` is P/(P − R), recombination only. `collection_efficiency` is the Ramo–Shockley signal over P.
- Back-diffusion against the field (~kT/eV ≈ 1e-4) is deliberately not modelled: its boundary layer (~0.3 µm) is sub-grid, so the upstream electrode reflects diffusion.
- Field spots and tracks cross the gap perpendicular to the plates. The stopping power is the mass stopping power **in the chamber gas**.
- **Initial recombination** comes from `Tile3D.single_track()`. It is coupled to general recombination by feeding the surviving source q·f_init into `Uniform1D`, with k_s = 1/(f_init f_general). The coupling is validated against full 3D runs (fig 9d); the DCPT benchmark uses it.

## Numerical rules (the "don't blow up" rules)

1. **Plan before running.** `plan_point` / `plan_field` give cell-steps and estimated seconds (≈ 40 ns per cell-step for 1D, ×2.5 with space charge; ≈ 60 ns for 3D; single thread). Pass `budget_s` for big jobs. Never integrate through long beam-off gaps: episodes and deduplication handle them.
2. **Steady-state skipping (1D only).** Inside a constant-source run, once three consecutive steps agree to `Grid1D.steady_state_tolerance` (1e-12), the kernel tallies the rest of the run without stepping. Results are stored as run-length records, so use `EpisodeResult.cumulative()` rather than per-step arrays. `Tile3D` is stochastic and steps every Δt: keep its episodes short (single tracks, pulses, milliseconds).
3. **Sources are integrals over the step** (`PiecewiseConstant.cumulative`), so charge is conserved for any Δt. Never sample rates at step times.
4. **Positivity and stability:**
   - drift is flux-limited (van Leer) upwind, exact at Courant 1 for the faster species in a uniform field;
   - explicit diffusion keeps D·Δt/Δz² ≤ 1/3 (and D·Δt/Δx² ≤ 1/4 laterally in 3D);
   - recombination is the **exact local solution** (Strang-split).

   Don't reintroduce Lax–Wendroff or forward-Euler recombination (legacy N1/N2).
5. **Space charge sub-cycles the drift.**
   - No cell may lose more than 90 % of its content per sub-step. The limiter is off in cells that lose charge through both faces.
   - Each sub-step must also resolve the **dielectric relaxation time** ε₀/σ. Otherwise the explicit drift–Poisson coupling overshoots and reverses the field; at 1e12 cm⁻³ this takes about 30 sub-steps.
   - A 5 % Δt margin avoids sub-cycling for small distortions.
   - Screened plasmas drain slowly, so episodes may run up to 20 memory times.
6. **Conservation is asserted in tests:** produced = recombined + collected + remaining (1e-11). The Ramo identity (1e-10) holds for every model and physics combination. Any new term must keep both.
7. **Accuracy:**
   - 1D: at 400 cells Boag pulsed is reproduced to ~2e-5 and Fenwick–Kumar to ~3e-4 relative in (1 − f). The default is 200 cells.
   - 3D: lateral cells of about b/6 reproduce Jaffé to ~0.3 % (convergence order ~1.7). The automatic tile is 3 track widths after one transit.

   Justify other resolutions by convergence, not rules of thumb.
8. **Parallelise across sample points and episodes.** The kernels are `nogil`. Keep kernels serial.
9. **Dedupe keys** (`shape_key`) are 9-digit rounded. Never loosen them so far that different episodes merge.

## Working rules

- **One Numba backend.** No Cython, CuPy or pure-Python copies of kernels. `NUMBA_DISABLE_JIT=1` is the debug path.
- Keep numerics free of I/O: no printing, plotting or timing in `src/`. Return result objects.
- Tests assert on **k_s − 1** or (1 − f) with relative tolerances, never on k_s ≈ 1 directly.
- **Benchmarks against measurements are blind predictions.** Never fit a model input (spot size, α, mobilities, electron parameters, error terms) to the data it is compared with. Take inputs from delivery records or independent literature, cite them, and propagate their uncertainty as a band. Choose the measurement uncertainties before looking at the residuals. Declare the primary prediction before running; show alternatives (for example the classic physics) as comparisons, not replacements.
- **Plots use native matplotlib with seaborn's theme on top.** Import `validation/plotstyle.py` and call `plotstyle.apply()`. It gives the default colour cycle (C0, C1, …), plain greys, viridis for sequential maps, and matplotlib's native font (DejaVu Sans). Don't write custom palettes or rcParams blocks in figure scripts.
- **Validation oracles:**
  - Boag pulsed (no diffusion, exact) and Boag continuous (first order; the second-order deviation is about −1.8 ξ²);
  - Fenwick & Kumar 2023 (free electrons, exact) and Jaffé (initial recombination, prescribed diffusion);
  - the IonTracks v3 single-track results;
  - equivalence limits, and conservation.
- New point models implement the duck interface: `memory_time_s`, `estimate_cell_steps`, `run_episode`, `seconds_per_cell_step`.
- **Validity:**
  - Continuous or slowly varying beams: `space_charge_parameter` η = e q d⁴ / (ε₀ µ⁺ V²).
  - Short pulses: `space_charge_parameter_pulsed` = e n₀ d² / (ε₀ V) ≈ 3.9 u in air.

  Space charge is modelled (1D mean field), but both benchmarks show its size is uncertain; see the open items.
- Keep formatting-only commits separate from logic changes.

## Open items and flags

- **Physics parameter consistency (top priority).** Across the two blind benchmarks:
  - the free-electron term changes little at their fields;
  - the space-charge term, with Boag's ion constants, adds loss that the measured voltage and field dependence doesn't show (DCPT: η ≈ 3 at spot centres at 50 V; Risø: 5–10 mm gaps);
  - yet space charge is what reproduces the pulse-overlap loss at Risø.

  Candidates: the ion parameter set (Boag's µ⁺ = 1.36 is low; the free-electron literature uses 1.87 cm²/Vs and α = 1.3e-6 cm³/s), and the 1D mean-field approximation. The Götz attachment rate is stated only for ≥ 1000 V/cm. Choose parameters from independent data, never by fitting the benchmarks.
- **Flagged, not modelled yet:**
  - α depends on air density (it is kept constant);
  - humidity;
  - charge multiplication at high fields;
  - the polarity effect;
  - guard-ring and edge fields (relevant for the Risø 10 mm gaps);
  - oblique tracks (Jaffé's angle dependence);
  - the lateral field of single tracks.

  The benchmark bands cover spot size or ionisation only, not the µ±, α or electron parameters.
- **Deferred by the maintainer:** the parameter-consistency decision above.
- **Postponed by the maintainer:** a machine-log / spot-list loader, until a real log is available. Tests for the two solver warnings (gap not emptied, episodes interacting) are on hold.
- **Data:**
  - DCPT: the chamber is a Roos (2 mm), not an Advanced Markus (1 mm); `k_s` and the dose-rate columns are undocumented.
  - Risø: the high-dose-rate pulse counts in `pulses_per_MU.csv` are unreliable above ~120 Hz. Independent TrueBeam counts (White et al. 2026) support this. The temperature is unknown.
  - The maintainer is collecting the missing inputs: DCPT spot sizes and dead time; the Risø temperature, pulse counts and chamber type. A web search found no DCPT-specific spot sizes; Rahman et al. 2020 (Med. Phys., a ProBeam TOPAS model) is a second ProBeam source.
