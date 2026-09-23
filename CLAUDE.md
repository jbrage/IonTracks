# CLAUDE.md

IonTracks computes **ion recombination losses (k_s)** in air-filled parallel-plate ionization chambers for beams with an **arbitrary, user-defined time structure**. Examples: pulses, pulse trains, spills, and whole scanned (PBS) fields evaluated at lateral sample points of the chamber. It solves drift, diffusion and volume recombination (α·n⁺·n⁻) of positive and negative ions across the gap.

**Branch `v4` is a rewrite.** The previous code is tagged `v3-legacy` and lives read-only under `legacy/`. Background and decisions:

- `report/code_review_and_refactoring_plan.md`: legacy bugs, numerics, architecture, decisions (§7), whole-field design (§8);
- `report/literature_review.md`: theory, validation data, related models;
- `report/verification/verify_findings.py`: reproduces the legacy findings.

## Commands

```bash
pip install -e ".[dev]"                 # Python >= 3.10, Windows or Linux; no compiler needed
python -m pytest                        # ~5 s; the first run adds ~7 s of Numba JIT (cached afterwards)
ruff check src tests examples validation report && ruff format --check src tests examples validation report
python examples/scanned_field.py        # end-to-end field example (~2 s)
python examples/cyclotron_250MeV.py     # 250 MeV cyclotron PBS field, 2 mm / 200 V, current sweep (~12 s)
python validation/run_validation.py     # 26 quantitative checks + 5 diagnostic figures -> validation/VALIDATION.md (~2 min)
python validation/benchmark_dcpt2019.py # benchmark vs DCPT 2019 measured charges -> validation/BENCHMARK_DCPT2019.md (~20 s)
python validation/benchmark_risoe2026.py # benchmark vs Risø 2026 pulsed e-/photon extrapolation chamber -> validation/BENCHMARK_RISOE2026.md (~40 s)
```

CI (`.github/workflows/main.yml`) runs lint, tests and both examples on ubuntu-latest and windows-latest with Python 3.10 and 3.13. A separate `validation` job (ubuntu, 3.13) runs `run_validation.py`, which exits non-zero if any check fails, and both benchmarks as smoke tests. The validation scripts need the `plot` extra (matplotlib).

## Layout

- `src/iontracks/`
  - `gas.py`: `Gas`, with **separate µ⁺, µ⁻ and D⁺, D⁻**. `Gas.air(T_K, p_hPa)` uses Boag's mobilities 1.36 and 2.1 cm²/Vs, density scaling and the Einstein relation. `Gas.air_legacy()` holds the v3 constants.
  - `chamber.py`: `ParallelPlateChamber` (uniform field, no space charge) and `sample_points()` (equal-area quadrature over the sensitive disc).
  - `time_structure.py`: `PiecewiseConstant`, the one representation of every time structure. It has exact integrals, `support()`, `window()` and a `shape_key()` for deduplication.
  - `field.py`: `Field` (spot list). `local_source(x, y, gas)` returns the exact ion-pair rate history at a point.
  - `planner.py`: splits the history into independent **episodes** (gaps > model memory time), groups identical episodes, and estimates cost.
  - `solvers/uniform1d.py`: the Numba point model `Uniform1D` / `Grid1D` and the kernels `_react`, `_transport` and `_simulate`.
  - `results.py`: `PointResult` and `FieldResult` (charge-weighted k_s, current I(t), `ks_map`).
  - `api.py`: `solve_point`, `solve_field` and `plan_field`, with a thread pool over points and a shared episode cache.
  - `analytic/boag.py`: Boag pulsed and continuous oracles, `boag_quasi_static_loss` (Boag continuous per constant segment), and the validity checks `space_charge_parameter` (continuous) and `space_charge_parameter_pulsed`.
  - `synthetic.py`: synthetic deliveries (`cyclotron_pbs_field`, `rf_bunched`) and 250 MeV proton constants (S from the libamtrack tables, W_air = 34.44 eV from ICRU 90).
- `tests/`: kernels, conservation, analytic limits and parameter relations, planner equivalences, fields.
- `examples/`: runnable scripts. CI runs `scanned_field.py`.
- `validation/`: `run_validation.py` regenerates `figures/*.png` and `VALIDATION.md`. Rerun it after any numerics change and check that every check still passes. `benchmark_*.py` are the blind benchmarks against measurements. Their data, with provenance READMEs, are in `validation/data/`, including the proton stopping-power tables, so nothing in `validation/` reads `legacy/`.
- `legacy/`: the v3 code, for reference and baselines only. Don't import it from `src/`, and don't lint it.

## Physics and units conventions

- Internal units are cm, s, V, eV. Densities are ion pairs per cm³, and gap-integrated tallies are **per cm² of electrode area**. Keep the unit-suffix naming (`gap_cm`, `voltage_V`, `t_on_s`, `stopping_power_MeV_cm2_g`).
- Positive ions drift to z = gap, negative ions to z = 0. Air's slower species (+) sets the collection time t_c = d/(µ⁺E).
- **Chamber memory** (`Uniform1D.memory_time_s`) is t_c + 8 diffusion widths, ×1.25. Beam-off gaps longer than that separate independent episodes.
- `ks` is P/(P − R), recombination only. `collection_efficiency` is the Ramo–Shockley signal over P. They are equal once the gap is empty, because back-diffusion against the field (~kT/eV ≈ 1e-4) is deliberately not modelled: its boundary layer (~0.3 µm) is sub-grid, so the upstream electrode reflects diffusion.
- Field spots: particles cross the gap perpendicular to the plates, the fluence has a Gaussian lateral profile, and each spot delivers uniformly over [t_on, t_off). The stopping power is the mass stopping power **in the chamber gas**.

## Numerical rules (the "don't blow up" rules)

1. **Plan before running.** `plan_point` / `plan_field` give cell-steps and estimated seconds (`SECONDS_PER_CELL_STEP` ≈ 40 ns, single thread). Pass `budget_s` for big jobs. Never integrate through long beam-off gaps: episodes and deduplication handle them.
2. **Steady-state skipping.** Inside a constant-source run, once three consecutive steps agree to `Grid1D.steady_state_tolerance` (1e-12), the kernel tallies the rest of the run without stepping. It agrees with full stepping to 1e-11. Long spots (for example 128 ms at 1 nA) then cost the same as short ones. Results are stored as run-length records, so use `EpisodeResult.cumulative()` rather than per-step arrays.
3. **Sources are integrals over the step** (`PiecewiseConstant.cumulative`), so charge is conserved for any Δt. Sub-step structure such as RF bunches never needs resolving. Never sample rates at step times.
4. **Positivity and stability in `uniform1d`:**
   - drift is flux-limited (van Leer) upwind, exact at Courant 1 for the faster species;
   - explicit diffusion keeps D·Δt/Δz² ≤ 1/3;
   - recombination is the **exact local solution** (Strang-split), so it has no Δt limit.
   Don't reintroduce Lax–Wendroff or forward-Euler recombination (legacy N1/N2).
5. **Conservation is asserted in tests:** produced = recombined + collected + remaining (1e-11). ∫I dt = collected charge (1e-10). Any new solver or physics term must keep both.
6. **Accuracy:** at 400 cells Boag pulsed is reproduced to ~2e-5 relative in (1 − f). The default is 200 cells, and tests typically use 80–150 for speed. Justify other resolutions by convergence (halve Δz), not rules of thumb.
7. **Parallelise across sample points and episodes.** The kernels are `nogil` and a single-grid kernel is memory-bound. Keep kernels serial.
8. **Dedupe keys** (`shape_key`) are 9-digit rounded. A key mismatch only costs runtime. Never loosen them so far that different episodes merge.

## Working rules

- **One Numba backend.** No Cython, CuPy or pure-Python copies of kernels. `NUMBA_DISABLE_JIT=1` is the debug path.
- Keep numerics free of I/O: no printing, plotting or timing in `src/`. Return result objects.
- Tests assert on **k_s − 1** or (1 − f) with relative tolerances, never on k_s ≈ 1 directly.
- **Benchmarks against measurements are blind predictions.** Never fit a model input (spot size, α, mobilities, error terms) to the data it is compared with. Take inputs from delivery records or independent literature, cite them, and propagate their uncertainty as a band. Choose the measurement uncertainties before looking at the residuals.
- Validation oracles: Boag pulsed (no diffusion, exact), Boag continuous (first order; the second-order deviation is about −1.8 ξ²), equivalence limits, and conservation. Planned: Fenwick & Kumar 2023 f_exp (free electrons), and Jaffé (initial recombination).
- New point models (e.g. a 3D track tile) implement the `Uniform1D` duck interface: `memory_time_s`, `estimate_cell_steps`, `run_episode`, `seconds_per_cell_step`. Plan §8.3.
- **Initial recombination will be added by the maintainer.** The planned coupling feeds the surviving source S·f_init into the general-recombination solver (plan §8.4). Don't port the legacy single-track solver unless asked.
- **Validity:** the field is uniform (no space charge).
  - Continuous or slowly varying beams: check `space_charge_parameter` (η = e q d⁴ / (ε₀ µ⁺ V²)) at the peak rate. Results with η ≳ 1 are outside validity; for 2 mm / 200 V that is above roughly 50 Gy/s local dose rate.
  - Pulses much shorter than the transit time: η at the peak rate overstates the effect. Use `space_charge_parameter_pulsed` (e n₀ d² / (ε₀ V) ≈ 3.9 u in air) instead; it exceeds 1 once Boag's u exceeds about 0.25.
- Open items: free electrons and space charge (1D first; top priority for high-current and FLASH fields), the 3D track tile, stopping-power and LET helpers, and validation against further literature datasets. The DCPT 2019 benchmark is done (`validation/BENCHMARK_DCPT2019.md`); the chamber is a Roos with a 2 mm gap, not an Advanced Markus (1 mm). The Risø 2026 pulsed benchmark (`validation/BENCHMARK_RISOE2026.md`) shows the signatures of free electrons (small gaps) and of under-predicted pulse-overlap loss (large gaps, high rate), so it is the first test for the free-electron and space-charge extension. Its high-dose-rate pulse counts in `pulses_per_MU.csv` are unreliable above ~120 Hz.
- Keep formatting-only commits separate from logic changes.
