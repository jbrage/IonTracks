# IonTracks: code review and refactoring plan

*Review of the working tree on branch `master` (HEAD `5659299`, plus uncommitted changes), 2026-09-23.*
*Companion documents: [literature_review.md](literature_review.md) and [verification/verify_findings.py](verification/verify_findings.py). The script reproduces every number marked **[measured]** in about 30 s with NumPy + Numba and needs no Cython build.*

**Refactor goal:** a user supplies the **temporal (and optionally spatial) structure of an ion beam**, and IonTracks returns *k*<sub>s</sub>, the collected charge and the time-resolved electrode current. That covers the regime where tracks from different times overlap in complex ways: pulses, spills, spot scanning, synchrocyclotron and FLASH beams. The temporal and spatial discretisation must stay bounded so the calculation does not blow up.

---

## 0. Executive summary

1. **The physics core is sound but narrow.** Positive and negative ions drift, diffuse and recombine (α·n⁺·n⁻) on a 3D grid, and ion tracks are Gaussian or Geiss columns parallel to the field. This is the right model for initial + general recombination at conventional and moderately high dose rates. The code realises only three time structures: a single track, a "continuous" Poisson-like beam, and (for uniform density) one instantaneous pulse or a constant rate. There is no user-facing time-structure input.
2. **The same algorithm exists up to 5 times.** Python, Cython, Numba, Numba-parallel and CuPy copies exist for hadrons, and Python, Cython and Numba copies for electrons. The copies have **already diverged in their physics results**: B1 and B3 below change results, and B9 is latent. Consolidating on one kernel implementation is the single most valuable refactoring step.
3. **Correctness issues found (all reproducible, see §2.1):**
   - The Numba and parallel continuous-beam backends score recombination during build-up. With identical seeds this makes *k*<sub>s</sub>−1 **17–49 % larger** than the Cython logic **[measured]**.
   - All continuous-beam backends score production over 2 separation times but recombination over 3, which double-counts the drain tail. That biases *k*<sub>s</sub>−1 **≈ +10 %** **[measured]**.
   - The Python, Numba and parallel single-track solvers drop the lateral drift for tilted tracks (`self.cx = cy`).
   - `Jaffe_theory` crashes with its default `input_is_LET=True`.
   - The electron continuous-beam example cannot be imported.
   - The default random seed is frozen at import time.
4. **Numerics:**
   - The Lax–Wendroff stencil is not positivity preserving. It undershoots to −1.5 % of the column density at every column end **[measured]**.
   - Recombination is an explicit Euler step that becomes unstable as α·n·Δt → 1. That is about 10 Gy/pulse at the default Δt **[measured]**.
   - The time step is always the CFL limit of the finest grid, independent of what the beam is doing.
5. **Why it blows up:**
   - One collection time *t*<sub>c</sub> ≈ 25–150 µs needs 250–560 time steps on a 0.5–2×10⁶-voxel grid at 5 µm.
   - A 1 ms synchrocyclotron period is about 7 *t*<sub>c</sub>, a 5 ms PBS spot about 34 *t*<sub>c</sub>, a 0.5 s PBS layer about 3.4×10³ *t*<sub>c</sub> and a 2 s synchrotron spill about 1.4×10⁴ *t*<sub>c</sub>. That means **10⁴–10⁷ steps** if integrated naively **[measured cost model]**.
   - Refining to 1 µm (for example for a Geiss core) multiplies the cost by about 600.
6. **The way out is structural, not just faster kernels.** The chamber has a **finite memory**: after about one collection time without beam it is empty. So:
   - (a) split the user's timeline into independent **episodes**;
   - (b) simulate one representative of repeated pulses;
   - (c) treat slowly varying intensities **quasi-statically**;
   - (d) run overlapping pulse trains to a **periodic steady state**;
   - (e) pick the **lowest-dimensional model** that is valid (1D uniform, 2D axisymmetric single track, or 3D periodic tile);
   - (f) run a **preflight cost/accuracy plan** before any heavy computation.

   Together these remove 2–3 orders of magnitude from realistic cases before any kernel optimisation (estimates, to be confirmed in Phase 2/3).
7. **The literature confirms the niche** ([literature_review.md](literature_review.md)). Uniform-density 1D models already handle arbitrary pulse structure, free electrons and space charge (Gotz 2017, Kranzer 2021/22, Paz-Martín 2022/25). Track models (IonTracks, Orts 2024) handle only "pulsed" or "continuous". **Ion tracks + arbitrary time structure** is unsolved. Furutani 2022 (short-spot PBS fits neither Boag model) and Gan 2026 (2.5 % error from misclassified time structure) are ready-made motivating cases.

---

## 1. What the code does today

### 1.1 Layout and data flow

```
hadrons/
  functions.py            user API: Jaffe_theory, ks_initial_IonTracks, IonTracks_continuous_beam,
                          E_MeV_u_to_LET_keV_um (PSTAR tables), doserate_to_fluence, calc_b_cm (b(LET) fit)
  solver.py               SolverType enum -> single-track backends
  common_properties.py    constants (W, mu, D, alpha, grid factors), used only by Python/Numba single-track
  geiss_utils.py          Geiss radial dose distribution
  cython_files/           initial_recombination.pyx (single track), continuous_beam.pyx (reference)
  python/ numba_files/ parallel/ python_cupy/   re-implementations of the same two solvers
  data_LET/               stopping power tables (air, water), LET->b table
electrons/
  Boag_theory.py          Boag pulsed / continuous formulas
  common/generic_electron_solver.py   dataclass + abstract "beam" hooks (Python loop)
  common/{continous,pulsed}_e_beam.py factories adding beam hooks
  numba/ python/          backends of the generic solver
  cython/                 two standalone .pyx solvers with their own constants
tests/                    single-track tests vs Jaffé; example-script regression (CSV equality)
documentation/README.md   convergence study of radius_cm, explanation of build-up/steps
```

**Hadron single track** (`single_track_PDEsolver`):
1. A Gaussian or Geiss column is placed on a grid of width 6*b* (*b* = Gaussian track radius), spanning the gap plus a 4-voxel z-buffer.
2. The time step comes from the Dehghan (2004) von Neumann criterion.
3. The solver steps 2 × *d*/(2µ*E*) and returns 1/*f* = *N*<sub>init</sub>/(*N*<sub>init</sub> − *N*<sub>rec</sub>).
4. The track angle is modelled by tilting the *field* (cx = µ*E*Δ*t*/Δ*x*·sin θ) while the electrodes stay z-planes.

**Hadron continuous beam** (`continuous_beam_PDEsolver`):
1. The domain is a 2*r* = 0.24 mm cylinder of interest inside a square grid with a 10-voxel lateral buffer and a 5-voxel z-buffer.
2. *N* = int(Φ̇·*T*·*A*) tracks are placed at uniformly random positions inside the circle.
3. Arrival times are the normalised cumulative sum of U(0,1) variates, histogrammed per step.
4. Scoring:
   - production is scored for steps > *t*<sub>sep</sub> = *d*/(2µ*E*);
   - recombination is scored for steps > *t*<sub>sep</sub> inside the circle and gap;
   - insertion runs for 3 *t*<sub>sep</sub>, followed by 1 *t*<sub>sep</sub> of drain.

**Electron/photon beam** (uniform density):
- the same stencil on a 3D grid with `sx = sy = 0`;
- "continuous" adds *n*·Δ*t* every step, "pulsed" adds *n* at *t* = 0;
- validated against Boag pulsed and continuous in `example_*`.

### 1.2 Backend matrix (who implements what)

| | Python | Cython | Numba | Numba-parallel | CuPy |
|---|---|---|---|---|---|
| Hadron single track | ✓ (bug B3) | ✓ reference | ✓ inherits B3 | ✓ inherits B3 | – |
| Hadron continuous | ✓ | ✓ reference | ✓ (bug B1) | ✓ (bug B1) | ✓ |
| Electron continuous | ✓ (generic) | ✓ own constants | ✓ (generic, B9) | – | – |
| Electron pulsed | ✓ (generic) | ✓ own constants | ✓ (generic, B9) | – | – |

Constants differ between copies:
- *W* = 33.9 (single track) vs 34.2 eV (continuous);
- µ = 1.65 (hadrons) vs 1.73 (electrons);
- grid 5 µm (electron pulsed) vs 6 µm (electron continuous);
- cylinder radius 0.012 vs 0.002 cm.

### 1.3 Working-tree state

The uncommitted changes are almost entirely a Black-style reformat (18 files). The Numba/parallel continuous-beam rewrites (buffer swapping, per-phase timing prints) are already in HEAD; bug B1 predates the working tree. **Commit the formatting as its own commit before refactoring**, so refactor diffs stay reviewable.

---

## 2. Findings

Severity: **H** = changes physics results, **M** = wrong in some configurations or a latent trap, **L** = quality or performance.

### 2.1 Correctness bugs

| ID | Sev | Where | Problem | Evidence | Fix |
|---|---|---|---|---|---|
| B1 | H | [continuous_beam_numba.py:350](../hadrons/numba_files/continuous_beam_numba.py#L350), [continuous_beam_numba_parallel.py:356](../hadrons/parallel/continuous_beam_numba_parallel.py#L356) | `calculate_new_densites` accumulates recombination on **every** step. Production is only counted after build-up (Cython/Python/CuPy gate both). | *k*<sub>s</sub>−1 ratio Numba/gated = 1.39, 1.49, 1.17, 1.20 for four seed/fluence cases **[measured, check A]** | Pass a `score` flag. Better: one shared kernel (§4). |
| B2 | H | [continuous_beam.pyx:163-172, 211, 258](../hadrons/cython_files/continuous_beam.pyx#L163) and all copies | Production is scored over steps (*t*<sub>sep</sub>, 3*t*<sub>sep</sub>], recombination over (*t*<sub>sep</sub>, 4*t*<sub>sep</sub>]. In steady state, recombination over a window already equals the recombination owed by that window's production, so adding the drain double-counts it. | (*k*<sub>s</sub>−1)·10³ = 9.04 ± 0.10 (repo scheme) vs 8.14 ± 0.12, 8.22 ± 0.11, 8.17 ± 0.08 for same-window estimators **[measured, check B]** | Score P and R over the **same** steady-state window, or run each episode until the domain is empty and use *f* = *Q*<sub>collected</sub>/*Q*<sub>produced</sub>. |
| B3 | H | [python/initial_recombination.py:107-108](../hadrons/python/initial_recombination.py#L107) | `self.sx = sy`, `self.cx = cy` sets cx = 0, so tilted tracks (θ ≠ 0) lose their lateral drift in the Python, Numba and parallel single-track solvers. Tests only use θ = 0. | inspection | `self.sx = sx; self.cx = cx`. Add a θ ≠ 0 test against Jaffé's angular formula. |
| B4 | H | [functions.py:42-45, 56, 107-108](../hadrons/functions.py#L42) | `Jaffe_theory(..., input_is_LET=True)` (the default) raises `UnboundLocalError`, because `LET_keV_um` is only set when `input_is_LET=False`. In the angled branch `x` is overwritten and later reported as `E_MeV_u`. | inspection | Rename parameters explicitly (`LET_keV_um=`, `E_MeV_u=`) and never reuse `x`. |
| B5 | M | [example_continuous_electron_beam.py:7](../electrons/example_continuous_electron_beam.py#L7) | Imports `Boag_continuous`, but the function is `Boag_Continuous`, so the script cannot run. Both electron examples also rely on the cwd (`from Boag_theory import`). | inspection | Fix the name and use package imports. |
| B6 | M | [functions.py:255](../hadrons/functions.py#L255) | `myseed=int(np.random.randint(1, 1e7))` is a default argument, evaluated **once at import**. Every call in a session reuses the same track positions and times, so "repeat runs" are not independent, and results are not reproducible across sessions unless a seed is passed. | Python semantics | `seed: int \| None = None` and `np.random.default_rng(seed)`. Record the seed used in the result. |
| B7 | M | all continuous hadron backends | Insertion writes to all lateral voxels, including boundary rows and columns that the update loop never touches. Boundary cells accumulate charge forever and feed it back into the domain through the stencil as a spurious source. In Numba, buffer swapping makes the two arrays alternate stale boundaries. Small with the default 10-voxel buffer (Gaussian tail e⁻⁶ at 50 µm, *b* = 20 µm). | inspection | Periodic lateral boundaries (§4.5) remove the buffer entirely. |
| B8 | M | Geiss RDD, single track | The grid half-width is 3*b* (Gaussian *b*), but Geiss *r*<sub>max</sub> = 4×10⁻⁵ *E*<sup>1.5</sup>/ρ<sub>air</sub> is **1 cm at 10 MeV/u, 33 cm at 100 MeV/u and 1.3 m at 250 MeV/u** in air. Point-sampling on a 60 µm-radius grid keeps only (1 + 2 ln(3*b*/*a*₀))/(1 + 2 ln(*r*<sub>max</sub>/*a*₀)) ≈ **25 %** of the track's charge at 100 MeV/u. *f* is normalised to the on-grid charge, so the low-density penumbra (which barely recombines) is silently dropped and *k*<sub>s</sub> is biased high. | calculation | Size the grid from the RDD. Deposit **cell-averaged** RDD (integrate over the voxel) so charge is conserved. Report the truncated fraction. |
| B9 | L (latent) | [electrons/numba/generic_electron_solver.py:39](../electrons/numba/generic_electron_solver.py#L39) | `sc_neg, sc_pos, sc_center = create_sc_gradients(...)` unpacks in the wrong order (the function returns pos, neg), so both species drift the opposite way to the Python backend. Harmless while µ⁺ = µ⁻ and the fill is uniform. It becomes wrong as soon as mobilities differ or space charge is added. | inspection | Fix the order. The unified kernel removes the duplication. |
| B10 | L | electron backends | The backends return different things: Cython returns DataFrames with a steady-state mean of *per-step* *f* after 2 *t*<sub>sep</sub>, Python/Numba return an array of *cumulative* *f* including build-up, and `cython/run_simulation.py` returns a Series. | inspection | One `Result` type. |
| B11 | L | arrival times | Track times come from `cumsum(U(0,1))`, normalised. Inter-arrival spacing is then uniform-distributed (CV ≈ 0.58), not exponential (CV = 1). A random beam is a Poisson process, and bunched beams are quantised to RF buckets. The time clustering of tracks, which drives inter-track recombination, is underrepresented. | inspection | Sample an inhomogeneous Poisson process from the user's rate(t) (§4.3). |

### 2.2 Numerical-method issues

**N1 – Positivity (Lax–Wendroff + central diffusion).** At the repo's chosen Δ*t* the z-stencil weight on the downstream neighbour is negative (−0.016 for *d* = 2 mm, 200 V, 5 µm; the cell Péclet number µ*E*Δ*x*/*D* = 22–67 ≫ 2). A drifting column end undershoots to **−1.5 % … −1.6 %** of its density **[measured, check C]**. Negative densities in α·n⁺·n⁻ produce spurious *negative* recombination. Fix: donor-cell/upwind at Courant number 1 (exact shift), or a flux-limited second-order scheme (van Leer / MC limiter; Hundsdorfer & Verwer 2003).

**N2 – Explicit recombination.** `recomb = α·n⁺·n⁻·Δt` is forward Euler. It is inaccurate once α·n·Δ*t* ≳ 0.1 and goes negative at ≥ 1. For uniform pulses at the default Δ*t* = 265 ns this is α·*n*·Δ*t* = 0.009 (0.1 Gy/pulse), 0.09 (1 Gy) and 0.94 (10 Gy) **[measured, check D]**. Track cores are fine today (≤ 2×10⁻³ for carbon at 10 MeV/u, Gaussian), but not if the grid is refined or dense cores overlap. Fix: the exact local solution of *ṅ*⁺ = *ṅ*⁻ = −α*n*⁺*n*⁻ over Δ*t*, which is closed form and unconditionally positive (§4.5).

**N3 – Time step selection.**
- `dt /= 1.01` from 1 s until the criterion holds takes about 1500 iterations; there is a closed form.
- The criterion has no reaction limit and no source (pulse-edge) limit.
- The second criterion `cx²cy²cz² ≤ 8 sx sy sz` is vacuous whenever cx = cy = 0 (always in continuous and electron runs).
- The step is **global and fixed**: the solver steps at the finest CFL limit even through beam-off periods.

**N4 – Isotropic grid.** Δ*z* = Δ*x*, although tracks are parallel to *E* and z-gradients exist only at column ends and the electrodes. Most of the 400 z-cells are wasted. Also, the drift-limited Δ*t* ∝ Δ*z* is forced down by the lateral resolution.

**N5 – Lateral domain.** A disc of interest with a buffer and Dirichlet-zero edges makes the result depend on the radius. The `documentation/README.md` convergence study shows that *k*<sub>s</sub> converges only for *r* ≳ 175 µm, versus the 120 µm default. For a laterally uniform broad beam, **periodic** lateral boundaries represent lateral charged-particle equilibrium exactly and remove the edge effect.

**N6 – Electrodes as buffers.** Ions keep drifting inside a 4–5 voxel buffer and can diffuse back into the gap. A proper outflow (absorbing) boundary is simpler, and it gives the **collected charge and electrode current** directly.

**N7 – No statistics.** Stochastic runs return one number. Seed-to-seed scatter of *k*<sub>s</sub>−1 is about 10–20 % at 10⁹ cm⁻²s⁻¹ in the test configuration **[measured, check A]**. Results need mean ± standard error and adaptive seeding.

**N8 – No conservation check.** Production = collected + recombined + remaining is never verified, although it is the cheapest correctness test available.

### 2.3 Physics-model gaps (relative to the target and the literature)

| Gap | Today | Why it matters | Reference |
|---|---|---|---|
| Separate µ⁺, µ⁻, *D*⁺, *D*⁻ | averaged µ = 1.65 (hadrons) / 1.73 (electrons), *D* = 0.037 | Continuous general recombination depends on µ⁺µ⁻ (Boag continuous), and pulsed on µ⁺ + µ⁻. Collection time is set by the slower positive ion (µ⁺ ≈ 1.36). With the Einstein relation, *D* = µ*kT*/*e* gives 0.034–0.053, not 0.037. | Boag; Boutillon 1998 |
| Gas state (*p*, *T*, humidity) | hard-coded constants | CCE shifts about 6–8 % per 100 hPa at UHDR | Flores-Mancera 2026 |
| Free electrons + attachment | absent | dominant loss mechanism above about 10–100 mGy/pulse | Boag 1996; Fenwick & Kumar 2023; Kranzer 2021; Gérard Ortega 2026 |
| Space charge (Poisson) | absent | ≤ 0.03 % at 1 mGy/pulse, ≤ 1.1 % at 10, ≤ 21 % at 100 mGy/pulse | Fenwick et al. 2024; Kranzer 2022 |
| Track geometry | field tilted, electrodes not | a tilted *track* between z-electrodes is the physical configuration, and tracks cross the electrodes at different lateral positions | Jaffé; Kanai 1998 |
| Lateral beam profile | uniform only | small spots and PBS gain drop | Tansho 2016; Christensen 2020 |
| Time structure input | none | the goal of this refactor | Furutani 2022; Gan 2026; Brualla-González 2016 |
| *W* per particle | 33.9 / 34.2 inconsistent | should be one configurable value per particle (ICRU 90) | — |

### 2.4 Engineering issues

- **Duplication and drift.** Five hadron backends and three electron backends, with constants copied into each. Bugs B1, B3 and B9 exist *because* of the copies.
- **Coupled imports.** `hadrons/functions.py` imports the compiled Cython modules at module level, and `tests/utils.py` imports `functions`. So *every* test, including the Numba ones, needs a Cython build and `mpmath`. On this Windows machine neither is available, so no test can run as-is. There is also a `sys.path.append("./cython_files")` hack.
- **Solvers do I/O.** Solvers do plotting (matplotlib inside the kernels), printing (`print(f"step ...")` every step in `python/continuous_beam.py`) and timing. Keep numerics pure and return data.
- **Parameter passing.** Untyped `dict`s (`parameter_dic`, `extra_params_dic`) with inconsistent keys (`d_cm` / `electrode_gap_cm` / `electrode_gap`, `elec_per_cm3` / `electron_density_per_cm3`).
- **Magic numbers.** `r_cm = 0.012`, `buffer_radius = 10`, `n_separation_times = 3`, `no_z_electrode = 4/5`, `1e8` voxel cap, and `number_of_iterations = 1e7`. The last one pre-allocates **160 MB** of random coordinates per call, and in Cython with `boundscheck(False)` an overrun would read out of bounds.
- **Cython directives misapplied.** In [initial_recombination.pyx:19-24](../hadrons/cython_files/initial_recombination.pyx#L19) the `boundscheck/wraparound/cdivision` decorators bind to `Geiss_r_max`, not to `single_track_PDEsolver`. The hot loops also contain `if debug:` branches and a Python closure call per voxel during initialisation.
- **Weak tests.**
  - Single-track tests compare *k*<sub>s</sub> ≈ 1.00001 to Jaffé with `rtol=1e-3`, so an error of up to **100×** in *k*<sub>s</sub>−1 would pass.
  - `test_single_track_numba.py` actually tests `NUMBA_PARALLEL`.
  - The example regression uses exact DataFrame equality (brittle).
  - There are no tests for continuous beams, electrons, angles, conservation or convergence.
- **Packaging.** `setup.py`, `setup.cfg` and a minimal `pyproject.toml` coexist. Classifiers say Python 3.8–3.10, CI tests 3.8–3.10 on Ubuntu only, and the local environment is Python 3.13 on Windows. `requirements.txt` mixes runtime and dev tools (isort, flake8).
- **Stray artefacts.** `times_result*.csv` (timing benchmarks) and a 145 kB notebook sit in the repo root.

---

## 3. Temporal and spatial resolution: why it blows up and how to keep it bounded

### 3.1 Characteristic scales (air, parallel plate)

| Symbol | Definition | Typical value |
|---|---|---|
| *t*<sub>c</sub>± | *d*/(µ±*E*), collection time | 25–150 µs (*d* = 1–2 mm, 100–400 V) |
| *t*<sub>sep</sub> | *d*/((µ⁺+µ⁻)*E*), end of intra-track (column) overlap for field-parallel tracks | ≈ *t*<sub>c</sub>/2 |
| *t*<sub>mem</sub> | chamber memory: time after which a beam-off chamber is empty (≈ *t*<sub>c</sub>⁺ + diffusion tail; longer with space charge) | ≈ 1.1–1.3 *t*<sub>c</sub>⁺ |
| *b* | Gaussian track radius | ≥ 20 µm (repo floor) |
| ℓ<sub>D</sub> | √(4*D t*<sub>sep</sub>), lateral spread during column overlap | 15–50 µm |
| τ<sub>rec</sub> | 1/(α*n*), local recombination time | ≫ *t*<sub>c</sub> conventional; ≪ *t*<sub>c</sub> at UHDR |
| η | Φ<sub>(*t*sep)</sub>·π(*b*² + ℓ<sub>D</sub>²), tracks overlapping one track during its column phase | ≪ 1 isolated · ~1 cross-term · ≫ 1 uniform |
| τ<sub>p</sub>, *T*<sub>p</sub> | pulse width and period of the beam | ns (RF, laser) … µs (linac, synchrocyclotron) … ms (PBS spot) … s (spill) |

### 3.2 Where the cost comes from **[measured cost model, check D]**

The cost is roughly cells × steps + tracks × cells-per-insertion.

| gap | V | Δ*x* | Δ*t* (repo) | steps per *t*<sub>c</sub> | voxels |
|---|---|---|---|---|---|
| 2 mm | 200 | 5 µm | 265 ns | 555 | 1.9×10⁶ |
| 1 mm | 300 | 5 µm | 97 ns | 254 | 9.7×10⁵ |
| 2 mm | 200 | 2 µm | 87 ns | 1 688 | 2.0×10⁷ |
| 2 mm | 200 | 1 µm | 32 ns | 4 557 | 1.4×10⁸ |

Beam window vs chamber memory (2 mm, 200 V, 5 µm):

| Beam structure | window / *t*<sub>c</sub> | naive steps |
|---|---|---|
| cyclotron RF spacing, ~14 ns | 10⁻⁴ | < 1 (sub-step: average it) |
| linac or synchrocyclotron pulse, ~5 µs | 0.03 | 19 |
| synchrocyclotron period, ~1 ms | 7 | 3.8×10³ |
| PBS spot dwell, ~5 ms | 34 | 1.9×10⁴ |
| PBS energy layer, ~0.5 s | 3.4×10³ | 1.9×10⁶ |
| synchrotron spill, ~2 s | 1.4×10⁴ | 7.6×10⁶ |

Two explosion mechanisms multiply. **Time:** a realistic delivery is 10³–10⁴ *t*<sub>c</sub> long. **Space:** cost ∝ Δ*x*⁻⁵ when Δ*t* is diffusion-limited (3 spatial dimensions plus Δ*t* ∝ Δ*x*²). Insertion adds cost too: every track currently evaluates `exp()` on **all** voxels (O(*N*<sub>tracks</sub>·*N*<sub>voxels</sub>)).

### 3.3 Principles for a bounded calculation

1. **Chamber memory ⇒ episodes.** Charge produced at *t* is gone by *t* + *t*<sub>mem</sub>. Merge beam-on intervals separated by gaps < *t*<sub>mem</sub> into *episodes*. Episodes are independent, and their produced, collected and recombined charges simply add. Nothing is ever integrated through a gap longer than *t*<sub>mem</sub>.
2. **Repetition ⇒ simulate once.** Pulse trains with period *T*<sub>p</sub> > *t*<sub>mem</sub> (conventional linacs, synchrocyclotrons at ~kHz, most UHDR e-linacs) become one representative pulse per distinct pulse charge, times *N*. This is Petersson 2017's observation that only DPP matters. For stochastic tracks, simulate *K* realisations for statistics, not all *N* pulses.
3. **Overlapping repetition ⇒ periodic steady state.** If *T*<sub>p</sub> < *t*<sub>mem</sub>, the pulses interact (Brualla-González 2016). Simulate consecutive cycles until the per-cycle *f* changes by less than a tolerance, then extrapolate over the remaining cycles and add start-up and tail transients.
4. **Slow modulation ⇒ quasi-static.** When intensity varies on a time scale *T*<sub>var</sub> ≫ *t*<sub>mem</sub> (spill envelope, long PBS spots), use *f*(*t*) ≈ *f*<sub>ss</sub>(rate(*t*)) from a small table of steady-state runs, charge-weighted. The error is O(*t*<sub>mem</sub>/*T*<sub>var</sub>). Simulate the on/off edges explicitly only when that error exceeds the tolerance.
5. **Sub-step structure ⇒ integrate, never sample.** Source terms are deposited as ∫*S* d*t* over each step, so charge is conserved for any Δ*t*. RF micro-structure (ns) is below any useful Δ*t*: ions move nanometres per bunch, so bunches merge exactly. Only pulses with τ<sub>p</sub> ≳ Δ*t* need resolving, with Δ*t* ≤ τ<sub>p</sub>/10 near edges.
6. **Pick the lowest valid dimension.**
   - **1D (z)** for uniform ionisation: photons/electrons, *or ion beams with η ≫ 1* (dense pulses, laser-driven).
   - **2D axisymmetric (r, z)** for a single field-parallel track (Jaffé / initial recombination). This is about 50× fewer cells than today's 3D single-track grid.
   - **3D periodic tile** only for partially overlapping tracks (η ~ 0.01–10) or tilted tracks.
   - Today the electron solver runs 196 identical filled columns (256 interior, 18×18 grid) for an exactly 1D problem **[measured, check D]**, about 250× wasted work with an identical result.
7. **Lateral decomposition.** Lateral transport during *t*<sub>mem</sub> is ~√(4*Dt*<sub>mem</sub>) ≈ 50–150 µm, far below spot sizes (mm). A PBS delivery therefore becomes a set of **independent lateral sample points**, each with its own local rate history Φ̇(*x*, *y*, *t*) = Σ<sub>spots</sub> *N*<sub>s</sub>(*t*)·*G*(*x* − *x*<sub>s</sub>, *y* − *y*<sub>s</sub>). This generalises Christensen 2020's "mapping". Deduplicate similar histories.
8. **Anisotropic, periodic grids.**
   - Lateral Δ*x* resolves *b*; Δ*z* only needs to resolve column ends and d (for example Δ*z* ≈ 2–5 Δ*x*, subject to convergence tests).
   - Periodic lateral boundaries remove the 10-voxel buffer and the radius convergence problem.
   - Drift-limited Δ*t* = Δ*z*/(µ*E*) grows with Δ*z*.
   - Rough estimate for 5 µm / 20 µm: 4× fewer cells and about 4.5× larger Δ*t* (1.2 µs vs 265 ns; explicit lateral diffusion stays stable up to Δ*x*²/(4*D*) = 1.7 µs), so **~18× fewer cell updates** before any kernel tuning.
9. **Split the operators, integrate reaction exactly.** Drift (upwind/limited), diffusion (explicit, or FFT/ADI if Δ*x* is small), recombination (closed form), source (integral). Stability then depends only on drift and diffusion. Accuracy of the splitting is controlled by keeping α*n*Δ*t* ≲ 0.1, using reaction sub-cycling in dense cores or pulses.
10. **Plan before running.** Compute all scales in §3.1, choose the model, grid and Δ*t*, estimate cells × steps, insertions, memory and runtime, and show the plan. Refuse (or ask for `force=True`) when over budget, and suggest the cheapest change that fits: coarser Δ*z*, fewer seeds, quasi-static, or hybrid.
11. **Converge, don't assume.** Every production setting is justified by a documented Δ*x*/2, Δ*z*/2, Δ*t*/2, *L*×2 and seeds×2 study. Resolution rules of thumb (for example Δ*x* ≤ *b*/4, *L* ≥ 6(*b* + ℓ<sub>D</sub>)) are *outputs* of Phase 2, not assumptions.

---

## 4. Target architecture

### 4.1 Package layout (proposal)

```
src/iontracks/
  units.py                 # cm, s, V, eV conventions + e, eps0, kT/e helpers (keep the _cm/_V/_s suffix style)
  gas.py                   # Gas dataclass: mu_pos, mu_neg, D_pos, D_neg, alpha, W_eV, T_K, p_hPa, (electron params);
                           # presets: AIR_KANAI1998 (legacy, reproduces old results), AIR_BOAG (mu+=1.36, mu-=2.1, Einstein D)
  chamber.py               # ParallelPlateChamber(gap_cm, voltage_V, guard / sensitive area)
  tracks/                  # radial dose models (Gauss(b), Geiss(a0, r_max), libamtrack adapter), b(LET), LET tables
  beam/
    time_structure.py      # Constant, Pulse, PulseTrain, Envelope*Micro, Piecewise, FromLog; integral(), support(), arrivals()
    spatial.py             # Uniform, GaussianSpot, SpotSequence (PBS), lateral sampling
    beam.py                # IonBeam(particle, energy/LET, track model, time, spatial) / UniformBeam(ionisation rate, time)
  analytic/                # jaffe.py, boag.py (pulsed, continuous, 1996 free-electron f', f'', f'''), fenwick_kumar.py (f_exp)
  numerics/
    grid.py                # Grid1D, GridAxi2D, GridTile3D (periodic lateral, anisotropic dz)
    kernels.py             # ONE set of numba kernels: drift (upwind/limited), diffusion, exact reaction, deposit, outflow
    stepping.py            # Strang splitting, dt selection (closed form), reaction sub-cycling
    poisson.py             # (phase 4) 1D cumulative / 3D FFT-DST
  solvers/
    uniform1d.py           # photons/electrons & dense ion pulses; optional free electrons + space charge
    track_axi2d.py         # single field-parallel track
    track_tile3d.py        # multi-track, arbitrary time structure
  planner.py               # scales, episodes, model choice, grid, cost estimate, Plan object
  result.py                # Result: ks, f, sem, Q_prod/coll/rec, I(t), per-episode table, plan, provenance
  api.py                   # solve(chamber, beam, gas=..., accuracy=..., budget_s=..., seed=...)
legacy/                    # frozen copy of today's hadrons/ and electrons/ for regression baselines (deleted after Phase 5)
tests/                     # unit / analytic / regression / convergence (slow) / conservation
```

### 4.2 User-facing API sketch

```python
from iontracks import solve, ParallelPlateChamber, gas, beam as B

chamber = ParallelPlateChamber(gap_cm=0.2, voltage_V=200)

# Synchrocyclotron-like: 7 us pulses at 1 kHz, 200 pulses, 5e7 protons/cm^2 per pulse
time = B.PulseTrain(B.RectPulse(width_s=7e-6), period_s=1e-3, n_pulses=200)
ion_beam = B.IonBeam(particle="proton", energy_MeV_u=228, track=B.GaussianTrack(),
                     time=time, fluence_per_pulse_cm2=5e7)

plan = solve.plan(chamber, ion_beam, gas=gas.AIR_BOAG, accuracy=1e-3, budget_s=600)
print(plan)            # scales, episodes (1 representative pulse x 200), model (3D tile), grid, est. runtime
result = plan.run(seed=1)
result.ks, result.ks_sem, result.current_A(t), result.episodes
```

Other time structures the API must express without expanding them:

```python
B.Constant(rate=...)                                  # continuous (cyclotron, averaged)
B.Envelope(B.Piecewise(t_edges, rates)) * B.RF(f_Hz=72.8e6, duty=0.1)   # synchrotron spill x RF (RF auto-averaged)
B.SpotSequence.from_csv("pbs_log.csv")               # t_on, t_off, x, y, MU per spot + sigma -> local rate(t) per lateral point
B.Piecewise.from_log("machine_log.csv")              # arbitrary measured intensity
```

Every primitive implements:
- `integral(t0, t1)`, exact for charge-conserving deposition;
- `support()`, the beam-on intervals used by the planner;
- `max_rate()`;
- `period()` (optional);
- `arrivals(rng, area_cm2, t0, t1)`, an inhomogeneous Poisson process (or RF-bucket-quantised when requested).

### 4.3 Planner (the "doesn't blow up" guarantee)

```
scales      <- t_c±, t_sep, t_mem, b, l_D, tau_rec, eta, pulse widths/periods          (§3.1)
episodes    <- merge support intervals with gaps < t_mem
for each episode:
    classify: single | independent-repeat (T_p > t_mem) | overlapping-periodic | slow-modulated | general
    choose model: analytic (if allowed & valid) | uniform1d | axi2d | tile3d | hybrid(axi2d x uniform1d)
    choose grid: dx from b (rule table from Phase 2), dz, tile L from (b, l_D, target tracks per t_mem)
    choose dt: min(drift Courant, diffusion, pulse-edge resolution, reaction accuracy)
    estimate: cells x steps x c_kernel + insertions x c_insert + seeds; memory
dedupe identical episodes / lateral histories; total cost vs budget -> Plan (inspectable, editable) or refusal with advice
```

`c_kernel` and `c_insert` are calibrated once per machine by a 1-second micro-benchmark, cached.

### 4.4 Solver hierarchy and when each is used

| Solver | Unknowns | Valid when | Validation oracle |
|---|---|---|---|
| analytic | — | textbook limits | literature |
| `uniform1d` | *n*⁺, *n*⁻ (+ *n*<sub>e</sub>, φ) on z | laterally uniform ionisation, or η ≫ 1 | Boag pulsed/continuous; Fenwick–Kumar f<sub>exp</sub> (1e-5); Paz-Martín *I*(*t*) data |
| `track_axi2d` | *n*⁺, *n*⁻ on (*r*, *z*) | one track, parallel to *E* | Jaffé (θ = 0); legacy single-track |
| `track_tile3d` | *n*⁺, *n*⁻ on periodic *L*²×*d* | anything else (overlap, tilt, time structure) | the three above in their limits; legacy continuous (after B1/B2 fix) |
| hybrid | *k*<sub>s,init</sub>(axi2d) × *k*<sub>s,gen</sub>(uniform1d) | cross term small (η ≪ 1 or LET low) | tile3d spot checks |

### 4.5 Numerical scheme (tile3d; uniform1d and axi2d are special cases)

Strang splitting per step Δ*t*:
1. ½ reaction (exact);
2. drift along *z* (upwind; exact shift at Courant 1 for the faster species, van Leer-limited otherwise; outflow through electrodes is tallied as collected charge);
3. diffusion (explicit 5-point lateral + 3-point z while stable; spectral/FFT lateral with periodic BC when Δ*x* is small);
4. ½ reaction;
5. source (∫*S* d*t* or tracks arriving in [*t*, *t*+Δ*t*)).

- **Exact reaction** for *a* = *n*⁺, *c* = *n*⁻ with δ = *a*₀ − *c*₀ conserved:
  - δ ≠ 0: *a*(Δ*t*) = δ / (1 − (*c*₀/*a*₀)·e^(−αδΔ*t*)) and *c* = *a* − δ;
  - δ = 0: *a* = *a*₀/(1 + α*a*₀Δ*t*).

  Recombined = *a*₀ − *a*(Δ*t*). Always positive, with no Δ*t* limit. When |αδΔ*t*| ≪ 1, evaluate with `expm1` or the δ = 0 form to avoid cancellation.
- **Track insertion:**
  - precompute the **cell-averaged** RDD kernel once (radius *r*<sub>cut</sub> = 4*b* for Gauss; the Geiss tail handled with a cut and the dropped fraction reported);
  - add it periodically wrapped at the track position (sub-voxel offset via bilinear split);
  - replicate along *z* only between the electrodes (or along the tilted path);
  - when many tracks arrive in one step, deposit centres on a 2D grid and FFT-convolve once.

  Cost is O(kernel) per track, or O(*N* log *N*) per step, instead of O(*N*<sub>voxels</sub>) per track.
- **Tallies per step:**
  - *Q*<sub>prod</sub>, *Q*<sub>rec</sub>, *Q*<sub>coll±</sub> and *Q*<sub>content</sub>, with the conservation residual asserted ≤ 1e-10 relative;
  - electrode current (Ramo–Shockley for plates) *I*(*t*) = (*e*/*d*)∫(µ⁺*E*<sub>z</sub>*n*⁺ + µ⁻*E*<sub>z</sub>*n*⁻ [+ *v*<sub>e</sub>*n*<sub>e</sub>]) d*V*.
- **Definitions:**
  - *f* = *Q*<sub>coll</sub>/*Q*<sub>prod</sub> for episodes run to empty (default, unambiguous);
  - *f* = 1 − *R*/*P* over the same window for steady-state runs (quasi-static tables).
- **Statistics:** *K* seeds (independent `np.random.Generator` streams), mean ± SEM of *k*<sub>s</sub> − 1, with adaptive *K* until SEM < target.
- **Phase 4 extensions:**
  - space charge (1D: cumulative sum; 3D: lateral FFT × z DST, updated every *m* steps);
  - free electrons with field-dependent velocity and attachment, integrated with sub-cycling while *n*<sub>e</sub> > tolerance (ns scale), then dropped.

### 4.6 Backends and performance

- **One Numba kernel set** (`parallel=True` on the outer loop, thread-local reductions), with `NUMBA_DISABLE_JIT=1` as the pure-Python debug path. Numba needs no compiler toolchain (Cython on Windows does) and runs on Python 3.10–3.13. In the repo's own benchmark (`times_result_0.csv`) it was about 2.6–2.9× faster than Cython on the larger single-track cases.
- **Retire Cython, the Python loops and CuPy** after their results are frozen as regression baselines (Phase 0).
- **GPU** (CuPy RawKernel or `numba.cuda`) only if Phase 3 profiling shows the tile kernels dominate at the target budgets.
- Default to float64, with float32 optional for GPU and large tiles, checked against float64 in tests.

---

## 5. Validation and test strategy

| Level | Test | Tolerance / criterion | Runtime |
|---|---|---|---|
| unit | exact reaction vs high-resolution ODE; drift exact shift; limiter positivity; periodic insertion conserves charge | 1e-12 / ≥ 0 / exact | ms |
| unit | time-structure `integral()` vs numeric quadrature; Poisson arrivals: mean and variance of counts | 1e-12 / χ² | ms |
| conservation | every solver, every run: *Q*<sub>prod</sub> = *Q*<sub>coll</sub> + *Q*<sub>rec</sub> + *Q*<sub>content</sub> | 1e-10 relative | free |
| analytic | uniform1d pulsed (µ⁺ = µ⁻, no *D*) vs Boag pulsed | on *k*<sub>s</sub>−1: 1e-3 relative, at converged grid | s |
| analytic | uniform1d continuous vs Boag continuous (small-*k*<sub>s</sub> limit) | 1 % of *k*<sub>s</sub>−1 | s |
| analytic | uniform1d + free electrons vs Fenwick–Kumar f<sub>exp</sub> | 1e-4 | s |
| analytic | axi2d / tile3d single track vs Jaffé, θ = 0 and θ = 30° | 2 % of *k*<sub>s</sub>−1 (Jaffé assumes infinite column) | s–min |
| equivalence | train with *T*<sub>p</sub> > *t*<sub>mem</sub> = *N* × single pulse; very short pulse → pulsed limit; long constant → continuous limit; tile3d with η ≫ 1 → uniform1d | within statistical SEM | min |
| regression | legacy Cython reference (Phase 0 baselines) after applying the B2 fix consistently | SEM-aware | min (slow mark) |
| convergence | Δ*x*, Δ*z*, Δ*t*, *L*, seeds halving/doubling → observed order, Richardson estimate | documented table | slow, nightly |
| experimental | Christensen 2020 (PBS), Kanai 1998 (high LET), Paz-Martín 2022 Zenodo *I*(*t*), Furutani 2022 short-spot | report, not gate | manual |

**Replace `assert np.allclose(ks_ref, ks, rtol=1e-3)` with checks on `ks - 1`.** Mark tests `unit`, `analytic`, `slow`, `experimental`. CI runs everything except `slow` and `experimental` on Linux **and Windows**, Python 3.10–3.13.

---

## 6. Phased roadmap

**Phase 0: safety net (small, do first).**
- Commit the reformatting separately.
- Build the Cython reference once, on Linux CI if Windows lacks MSVC, and **freeze baselines**: single track (θ = 0, 30°) and continuous (a few fluences, 10+ seeds, mean ± SEM) for the current and B2-fixed scoring.
- Fix B1, B3, B4, B5 and B6.
- Decouple `functions.py` from the Cython import (lazy import).
- Make the tests assert on *k*<sub>s</sub>−1 and add the conservation test.
- Clean the root (`times_result*.csv`).

*Exit:* tests run on Windows without Cython; baselines stored under `tests/baselines/`.

**Phase 1: skeleton + 1D.**
- `units`, `gas` (with the legacy preset), `chamber`, `analytic` (Jaffé, Boag ×3, BHB 1996, Fenwick–Kumar).
- The time-structure primitives with `integral` and `support`.
- `uniform1d` with upwind drift, exact reaction, electrode outflow and current, and the planner's scale computation.

*Exit:* the analytic tests pass, and the electron examples are reproduced about 250× faster.

**Phase 2: tracks.**
- `track_axi2d`, then `track_tile3d` (periodic, anisotropic, cell-averaged insertion, FFT batch insertion, seeds/SEM).
- The convergence study that produces the resolution rules.

*Exit:* Jaffé and legacy-baseline agreement; documented Δ*x*/Δ*z*/*L* rules; cost per (cell·step) and per insertion measured.

**Phase 3: time structure and planner.**
- Episodes, deduplication, quasi-static tables, periodic steady state, lateral decomposition for `SpotSequence`, preflight budget and `Plan` object.

*Exit:*
- the equivalence tests pass;
- a synchrocyclotron train (200 pulses) runs in minutes;
- a PBS layer from a spot log runs in < 10 min on a laptop at 1e-3 accuracy on *k*<sub>s</sub>−1 (target to confirm);
- the Furutani short-spot scenario is reproduced qualitatively.

**Phase 4: extended physics.**
- Separate mobilities as the new default (legacy preset kept).
- *p*/*T* scaling.
- Free electrons + attachment and space charge in `uniform1d`, then in `track_tile3d` for dense ion pulses.

*Exit:* Fenwick–Kumar to 1e-4; Paz-Martín and Kranzer data reproduced within their reported model–experiment agreement.

**Phase 5: retire legacy code, docs, examples.** Examples: continuous, pulsed, synchrocyclotron, PBS log, UHDR electron. Optional GPU.

---

## 7. Decisions

### 7.1 Decided (2026-09-23)

1. **One backend: Numba, CPU.** Cython, the Python loops and CuPy are retired after their results are frozen as baselines. The measurements behind this choice and the alternatives considered are in §7.3.
2. **Scope: whole fields in time, evaluated at spatial sample points inside the chamber's sensitive volume** (§8).
3. **Initial recombination is added later by the maintainer.** The new code provides a slot for it (§8.4) and does not port the legacy single-track solver.
4. **Layout and platforms:**
   - `src/iontracks/` with a PEP 621 `pyproject.toml` and Python ≥ 3.10;
   - Windows **and** Linux CI, which is feasible because the package has no compiled extension of its own;
   - legacy packaging removed: `setup.py`, `setup.cfg`, `requirements.txt`, `.gitpod.yml`, Cython build, CuPy extras, flake8/isort (replaced by ruff).
5. **Validation data:** all sources listed in the original question 5 will be supplied later. Already available now:
   - `origin/backend-benchmark:benchmark/2019_DCPT/experimental-data-200V/recombination_200V_data.csv` (DCPT, 200 V, 2 mm, 70–244 MeV);
   - the space-charge-screening prototype on `origin/spaceChargeScreening` (M. Orts, 2023) as a reference for Phase 4.

### 7.2 Decided (2026-09-23, second round)

- **Repository:** a new branch `v4` in this repository. The published master is tagged `v3-legacy`, and the legacy code is kept under `legacy/` on the branch.
- **Per-point model:** the 1D uniform-density model first, with tests. The 3D track tile comes later behind the same interface (§8.3, option C).
- **Gas defaults:** separate µ⁺, µ⁻ and *D*⁺, *D*⁻ (Einstein relation). The v3 constants remain as `Gas.air_legacy()`.

### 7.3 Backend choice: measurements and alternatives

Measured on this machine (14 threads), one drift + diffusion + recombination step on the legacy 1.9×10⁶-cell grid:

| Backend | ms/step | Mcell/s | vs NumPy |
|---|---|---|---|
| NumPy (vectorised) | 213 | 9 | 1× |
| Numba, serial | 5.1 | 369 | 41× |
| Numba, `parallel=True` | 2.9 | 652 | 73× |

The team's own `origin/backend-benchmark` run on the 2019 DCPT case found Numba 405 s vs Cython 1155 s vs Python ≈ 2400–7000 s.

**Consequence for fields.** Threading one grid gives only about 1.8× on 14 threads, because the kernel is memory-bandwidth bound. Independent sample points and seeds scale almost linearly. **Parallelise across points and seeds**, with each point running a compact serial kernel that fits in cache. Later, the same pattern runs on a cluster (one job per point batch).

| Alternative | Strength | Why not primary |
|---|---|---|
| CuPy RawKernel / `numba-cuda` (GPU) | likely 10–50× on large 3D tiles (estimate, not measured) | NVIDIA only. The 1D per-point solves are too small to benefit. Add later behind the kernel interface if 3D tiles dominate. |
| JAX | CPU/GPU, **autodiff** (fit α, µ± and free-electron parameters to measured *I*(*t*)) | no native Windows GPU; static shapes make random track insertion and event-driven stepping awkward. It is a candidate for a separate *fitting* layer, not the solver. |
| C++ / Rust extension (nanobind, PyO3) | full control, no JIT warm-up | brings back compiled wheels per platform (cibuildwheel), which is exactly the burden being removed |
| Taichi | stencil DSL, cross-platform GPU | development activity has slowed, so the maintenance risk is high |
| NumPy only | zero extra dependency | 41–73× slower (measured) |

### 7.4 Implementation status (branch `v4`)

Implemented in `src/iontracks/`, with 78 tests passing in about 5 s:
- `Gas`, `ParallelPlateChamber` with sample points, and `PiecewiseConstant` time structures;
- `Field` with exact local sources;
- the episode planner with deduplication and a budget check;
- the `Uniform1D` Numba point model (flux-limited drift, exact recombination, Ramo current);
- `solve_point` and `solve_field` (threads over points, shared episode cache);
- Boag oracles, and space-charge validity parameters for continuous beams and for pulses.

Measured accuracy:
- Boag pulsed: 2×10⁻⁵ relative in (1 − f) at 400 cells;
- Boag continuous: correct to second order;
- conservation and the Ramo identity: 10⁻¹¹.

Validation (`validation/`):
- 26 quantitative checks and 5 diagnostic figures (`VALIDATION.md`);
- two blind benchmarks against measurements, with no model input fitted to the compared data:
  - DCPT 2019 scanned protons: Q(V)/Q(200 V) within 0.35 %;
  - Risø 2026 pulsed electrons and photons: field ratios within 3.1 %. The residuals show the
    expected signatures of free electrons and space charge.

Design notes arising from implementation:
- **Back-diffusion is not modelled.** Diffusion against the field is not resolvable (boundary layer D/v ≈ 0.3 µm), so the upstream electrode reflects diffusion. The effect is about kT/(eV) ≈ 10⁻⁴.
- **Kernel cost** is 24–40 ns per cell-step (single thread). That is about 7 s of CPU per sample point for a 0.5 s PBS layer at 200 cells.
- **Dedupe keys** use 9 significant digits because absolute-time round-off limits microsecond pulses late in a delivery.

Not yet done:
- free electrons and space charge (1D);
- the 3D track tile;
- the initial-recombination slot (maintainer);
- stopping-power helpers;
- further validation datasets (§5).

---

## 8. Whole-field simulation at chamber sample points

### 8.1 Idea

During one collection time, ions move laterally only about √(4*D t*<sub>c</sub>) ≈ 50–150 µm. That is much less than PBS spot sizes and dose gradients (mm). So the sensitive volume splits into **independent lateral sample points *k*** with area weights *w*<sub>k</sub>. Each point experiences its own ionisation history *S*<sub>k</sub>(*t*), is solved independently in time with the episode planner (§3.3, §4.3), and the chamber signal is the weighted sum.

### 8.2 Data flow

```
Field (spot list or machine log):  per spot s: t_on, t_off, x_s, y_s, N_s (or MU + calibration),
                                   particle, E or LET at chamber depth, sigma_x, sigma_y
Chamber:                           sensitive radius, gap d, voltage V, position/rotation in field coords
Sample points (x_k, y_k, w_k):     quadrature over the sensitive area; refine where the local dose gradient is large
Per point k:
    S_k(t) = sum_s  [N_s / (t_off - t_on)] * G_s(x_k - x_s, y_k - y_s) * LET_s / W   on [t_on, t_off)
             -> piecewise-constant rate history (exact integrals, no time sampling)
    planner  -> episodes (gaps > t_mem split them), repeats, quasi-static segments
    solver   -> Q_prod,k, Q_coll,k, Q_rec,k, I_k(t)
Chamber:     Q = sum_k w_k Q_k ;  k_s = sum_k w_k Q_prod,k / sum_k w_k Q_coll,k ;  I(t) = sum_k w_k I_k(t)
```

`k_s` is **charge-weighted**, not an average of per-point *k*<sub>s</sub>. Per-point results are kept, so recombination maps across the chamber (as in Christensen 2020) come for free.

### 8.3 Model per point

| Option | Captures | Cost per point | Notes |
|---|---|---|---|
| A. `uniform1d` | general recombination with the full time structure, and later free electrons and space charge | ms–s | validated against Boag, Fenwick–Kumar. Needs *S*<sub>k</sub>(*t*) only. |
| B. `track_tile3d` | general + intra-track + **inter-track overlap** | s–min × seeds | needs per-spot LET → track radius, Poisson track arrivals from the local fluence rate |
| C. A everywhere, B at a few representative points | A's speed plus a measured correction for the track cross term | | recommended path: build A first, add B behind the same `PointModel` interface |

### 8.4 Slot for initial recombination (to be added by the maintainer)

- **Interface:** `InitialRecombination.surviving_fraction(particle, LET_keV_um, E_V_cm, angle_rad) -> f_init`.
- **Recommended coupling:** feed the general-recombination solver the **surviving** source, *S*<sub>k,eff</sub>(*t*) = Σ<sub>s</sub> *S*<sub>k,s</sub>(*t*)·*f*<sub>init,s</sub>, and report *Q*<sub>init-lost</sub> separately. This is more consistent than multiplying *k*<sub>s,init</sub>·*k*<sub>s,gen</sub>, because charge lost in the track core never takes part in general recombination. It also handles mixed-LET fields (several energy layers) naturally.
- Option B (§8.3) contains initial recombination intrinsically. When B is used, the slot must be switched off for that point to avoid double counting.

### 8.5 Cost control for fields

- Each spot's contribution is truncated at ≥ 4σ.
- Points with a negligible share of the total charge are dropped, and the dropped fraction is reported.
- Episodes and quasi-static segments are shared by the planner per point.
- Points run in parallel (process pool or `prange` over points).
- Preflight reports points × episodes × steps before running.
- Rough target for option A: a clinical layer with about 100 points in minutes on a laptop. This is an estimate, to be confirmed in Phase 3.

---

## Appendix A: reproducing the measured numbers

```
python report/verification/verify_findings.py           # A-D, ~30 s on 14 cores
python report/verification/verify_findings.py --quick   # skips the 24-run scoring study
```

The script patches the repo's Numba continuous-beam solver *in memory* (build-up / scoring window flags) and changes no repository file. Test configuration: *d* = 0.5 mm, 100 V, Δ*x* = 10 µm, LET = 0.02 keV/µm, *b* = 20 µm, Φ̇ = 10⁹ and 2×10¹⁰ cm⁻²s⁻¹. These settings are chosen for speed, not clinical realism, so the magnitudes of B1 and B2 will differ in other configurations while their sign and mechanism stay the same.

## Appendix B: file-by-file notes

- `hadrons/functions.py`:
  - B4, B6;
  - `E_MeV_u_to_LET_keV_um` prints and returns 0 on an unknown material or particle (should raise);
  - `calc_b_cm` refits a polynomial on every call (cache it);
  - `doserate_to_fluence` assumes dose-to-air input (document it or accept dose-to-water plus a stopping-power ratio).
- `hadrons/common_properties.py`: the corrupted comment on line 5 (`recombination constantno_figure_updates = 5`); used by only 2 of 5 backends.
- `hadrons/cython_files/initial_recombination.pyx`:
  - directives bound to the wrong function;
  - `debug` branches in the hot loops;
  - the RDD as a Python closure;
  - `raise` checks `no_x²·no_z` but reports the count with the buffer.
- `hadrons/cython_files/continuous_beam.pyx`:
  - B2, B7, B11;
  - *W* = 34.2 vs 33.9 elsewhere;
  - 160 MB coordinate pre-allocation;
  - plotting inside the solver.
- `hadrons/numba_files/*`, `hadrons/parallel/*`:
  - B1;
  - timing prints in library code;
  - in the parallel version, `MAXVAL` is updated conditionally inside `prange` (a race, plot only), and the single-track path allocates new arrays every step.
- `hadrons/python/continuous_beam.py`: prints every time step.
- `hadrons/python_cupy/continuous_beam_cupy.py`: `.item()` host syncs per track and per step; rejection sampling on GPU scalars.
- `electrons/common/generic_electron_solver.py`:
  - 3D loop for a 1D problem;
  - `ion_mobility` comment says "cm s⁻¹ V⁻¹" (should be cm² V⁻¹ s⁻¹);
  - the `get_initialised_charge_carriers_after_beam` abstract method lacks `self`.
- `electrons/numba/generic_electron_solver.py`: B9.
- `electrons/cython/*.pyx`: their own hard-coded grid (5 vs 6 µm) and cylinder radius; the pulsed solver returns *f*, the continuous one returns DataFrames.
- `electrons/Boag_theory.py`: fine as an analytic module. Move it into `analytic/` and add BHB 1996 and Fenwick–Kumar.
- `tests/`: see §2.4. Also, `ks_initial/generate_expected_result.py` imports `testing_parameters` by cwd.
- `setup.py` / `setup.cfg` / `pyproject.toml` / `requirements.txt` / CI: see §2.4.
