"""1D drift-diffusion-recombination solver for laterally uniform ionisation.

Per time step (Strang splitting): source, 1/2 recombination, drift + diffusion of each species,
1/2 recombination.

* Source: the exact integral of the rate over the step, spread uniformly over the gap.
* Recombination: exact local solution of dn+/dt = dn-/dt = -alpha n+ n-. It is unconditionally
  stable and positive.
* Drift: finite-volume, flux-limited (van Leer) upwind scheme. It is positivity preserving for
  Courant <= 1 and exact at Courant = 1 (the faster species by default).
* Diffusion: explicit. The collecting electrode absorbs and the upstream electrode reflects.
  Diffusion against the field (loss ~ kT/(eV) ~ 1e-4) is not modelled: its boundary layer
  D/v ~ 0.3 um cannot be resolved by the grid. D dt / dz^2 <= 1/3 keeps it positive.
* Signal: the induced (Ramo-Shockley) current, from the displacement of every face flux. After
  complete collection its time integral equals the charge an electrometer reads.

All tallies are per cm^2 of electrode area.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..chamber import ParallelPlateChamber
from ..constants import ELEMENTARY_CHARGE_C
from ..gas import Gas
from ..time_structure import PiecewiseConstant

# single-thread cost per cell and step (both species), measured 24-40 ns on a 2024 laptop;
# used only for the preflight estimate
SECONDS_PER_CELL_STEP = 4e-8


@njit(cache=True, nogil=True)
def _react(a, c, x):
    """Exact recombination over a sub-step with x = alpha * dt. Returns the summed density loss."""
    lost = 0.0
    for j in range(a.shape[0]):
        aj = a[j]
        cj = c[j]
        if aj <= 0.0 or cj <= 0.0:
            continue
        ax = aj * x
        cx = cj * x
        if (
            ax < 1e-6 and cx < 1e-6
        ):  # weak recombination: 2nd-order Taylor, error O((n x)^2) < 1e-12
            loss = aj * cx * (1.0 - 0.5 * (ax + cx))
            lost += loss
            a[j] = aj - loss
            c[j] = cj - loss
            continue
        d = aj - cj  # conserved by the reaction
        if d == 0.0:
            an = aj / (1.0 + aj * x)
        else:
            arg = -d * x
            if abs(arg) < 1e-4:  # series: error ~ arg^3 / 24 relative
                em1 = arg * (1.0 + arg * (0.5 + arg * (1.0 / 6.0)))
                an = aj * d / (d - cj * em1)
            elif arg > 700.0:  # a is exhausted (d < 0, strong recombination)
                an = 0.0
            else:
                an = aj * d / (d - cj * math.expm1(arg))
        if an < 0.0:
            an = 0.0
        elif an > aj:
            an = aj
        cn = an - d
        if cn < 0.0:
            cn = 0.0
        lost += aj - an
        a[j] = an
        c[j] = cn
    return lost


@njit(cache=True, nogil=True)
def _transport(n, courant, lam, flux):
    """Drift towards +index and diffuse one species in place.

    ``flux`` is scratch space of length N + 1. Returns (outflow, displacement): the outflow through
    the collecting electrode (a density; multiply by dz) and the displacement in units of
    dz^2 * density (sum over faces of flux times lever arm, lever 1 inside, 1/2 at the electrode).
    """
    N = n.shape[0]
    displacement = 0.0
    # --- drift: zero inflow at face 0, upwind outflow at face N
    if courant > 0.0:
        flux[0] = 0.0
        for f in range(1, N):
            up = n[f - 1]
            dn = n[f] - up
            limited = 0.0
            if dn != 0.0:
                upup = n[f - 2] if f >= 2 else 0.0
                r = (up - upup) / dn
                if r > 0.0:
                    limited = 2.0 * r / (1.0 + r)  # van Leer
            flux[f] = courant * (up + 0.5 * (1.0 - courant) * limited * dn)
        flux[N] = courant * n[N - 1]
        for f in range(1, N):
            displacement += flux[f]
        displacement += 0.5 * flux[N]
        outflow = flux[N]
        for j in range(N):
            n[j] -= flux[j + 1] - flux[j]
    else:
        outflow = 0.0
    # --- diffusion: reflecting upstream electrode (face 0), absorbing collecting electrode (face N)
    if lam > 0.0:
        flux[0] = 0.0
        for f in range(1, N):
            flux[f] = lam * (n[f - 1] - n[f])
        flux[N] = 2.0 * lam * n[N - 1]
        for f in range(1, N):
            displacement += flux[f]
        displacement += 0.5 * flux[N]
        outflow += flux[N]
        for j in range(N):
            n[j] -= flux[j + 1] - flux[j]
    return outflow, displacement


@njit(cache=True, nogil=True)
def _grow(arr, n_used):
    out = np.empty(2 * arr.shape[0], dtype=arr.dtype)
    out[:n_used] = arr[:n_used]
    return out


@njit(cache=True, nogil=True)
def _simulate(
    run_len,
    run_src,
    n_cells,
    c_pos,
    c_neg,
    lam_pos,
    lam_neg,
    alpha_dt,
    max_steps,
    empty_tol,
    ss_tol,
):
    """Step through source runs (constant source per step within a run) until the gap is empty.

    Steady-state skipping: when, for three consecutive steps of a run with constant source, each
    species' production equals its outflow plus recombination and the per-step tallies are
    unchanged (both to ``ss_tol`` relative), the state is a fixed point of the scheme and the rest
    of the run is tallied without stepping. Records (count, per-step values) keep the time series
    compact: one record per simulated step, one per skipped block.
    """
    n_pos = np.zeros(n_cells)
    n_neg = np.zeros(n_cells)
    n_neg_rev = n_neg[::-1]  # negative ions drift towards index 0: transport them reversed
    flux = np.zeros(n_cells + 1)
    cap = 1024
    rec_count = np.empty(cap, dtype=np.int64)
    rec_disp = np.empty(cap)
    rec_lost = np.empty(cap)
    rec_src = np.empty(cap)
    n_rec = 0
    produced = 0.0
    recombined = 0.0
    coll_pos = 0.0
    coll_neg = 0.0
    half = 0.5 * alpha_dt
    n_runs = run_len.shape[0]
    run = 0
    left = run_len[0] if n_runs > 0 else 0
    prev_lost = -1.0
    prev_o1 = -1.0
    prev_o2 = -1.0
    prev_d = -1.0
    calm = 0
    step = 0
    emptied = False
    while step < max_steps:
        s = run_src[run] if run < n_runs else 0.0
        if s > 0.0:
            for j in range(n_cells):
                n_pos[j] += s
                n_neg[j] += s
            produced += s * n_cells
        lost = _react(n_pos, n_neg, half) if half > 0.0 else 0.0
        out1, d1 = _transport(n_pos, c_pos, lam_pos, flux)
        out2, d2 = _transport(n_neg_rev, c_neg, lam_neg, flux)
        if half > 0.0:
            lost += _react(n_pos, n_neg, half)
        d = d1 + d2
        coll_pos += out1
        coll_neg += out2
        recombined += lost
        if n_rec == rec_count.shape[0]:
            rec_count = _grow(rec_count, n_rec)
            rec_disp = _grow(rec_disp, n_rec)
            rec_lost = _grow(rec_lost, n_rec)
            rec_src = _grow(rec_src, n_rec)
        rec_count[n_rec] = 1
        rec_disp[n_rec] = d
        rec_lost[n_rec] = lost
        rec_src[n_rec] = s * n_cells
        n_rec += 1
        step += 1
        if run < n_runs:
            left -= 1
            scale = s * n_cells
            # steady state = per-species balance (production = outflow + recombination, so the
            # gap content no longer changes) AND unchanged per-step tallies. The balance bounds
            # the conservation error of the skipped block by ss_tol.
            if (
                abs(scale - out1 - lost) <= ss_tol * scale
                and abs(scale - out2 - lost) <= ss_tol * scale
                and abs(lost - prev_lost) <= ss_tol * (abs(lost) + scale)
                and abs(out1 - prev_o1) <= ss_tol * (abs(out1) + scale)
                and abs(out2 - prev_o2) <= ss_tol * (abs(out2) + scale)
                and abs(d - prev_d) <= ss_tol * (abs(d) + scale)
            ):
                calm += 1
            else:
                calm = 0
            prev_lost = lost
            prev_o1 = out1
            prev_o2 = out2
            prev_d = d
            if calm >= 3 and left > 0 and s > 0.0:
                k = min(left, max_steps - step)
                produced += s * n_cells * k
                recombined += lost * k
                coll_pos += out1 * k
                coll_neg += out2 * k
                if n_rec == rec_count.shape[0]:
                    rec_count = _grow(rec_count, n_rec)
                    rec_disp = _grow(rec_disp, n_rec)
                    rec_lost = _grow(rec_lost, n_rec)
                    rec_src = _grow(rec_src, n_rec)
                rec_count[n_rec] = k
                rec_disp[n_rec] = d
                rec_lost[n_rec] = lost
                rec_src[n_rec] = s * n_cells
                n_rec += 1
                step += k
                left -= k
            if left == 0:
                run += 1
                calm = 0
                prev_lost = -1.0
                prev_o1 = -1.0
                prev_o2 = -1.0
                prev_d = -1.0
                if run < n_runs:
                    left = run_len[run]
        else:
            content = 0.0
            for j in range(n_cells):
                content += n_pos[j] + n_neg[j]
            if content <= 2.0 * empty_tol * produced:
                emptied = True
                break
    remaining_pos = 0.0
    remaining_neg = 0.0
    for j in range(n_cells):
        remaining_pos += n_pos[j]
        remaining_neg += n_neg[j]
    return (
        produced, recombined, coll_pos, coll_neg, remaining_pos, remaining_neg, step, emptied,
        rec_count[:n_rec].copy(), rec_disp[:n_rec].copy(), rec_lost[:n_rec].copy(),
        rec_src[:n_rec].copy(),
    )  # fmt: skip


def source_runs(source: PiecewiseConstant, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-step source amounts as runs of equal value, exact in total.

    Steps containing a breakpoint get their exact integral; the steps between two such steps get
    the exact average over the run. Memory scales with the number of breakpoints, not with time.
    """
    n_src = max(1, math.ceil(source.t_end_s / dt_s))
    boundary = np.unique(np.clip(np.floor(source.edges / dt_s).astype(np.int64), 0, n_src - 1))
    # run starts: 0, every boundary step, and the step after each boundary step
    starts = np.unique(np.concatenate(([0], boundary, boundary + 1)))
    starts = starts[starts < n_src].astype(np.int64)
    lengths = np.diff(np.append(starts, n_src)).astype(np.int64)
    amount = source.integral(starts * dt_s, (starts + lengths) * dt_s) / lengths
    return lengths, amount


@dataclass(frozen=True)
class Grid1D:
    """Discretisation of the gap. ``courant`` applies to the faster ion species."""

    n_cells: int = 200
    courant: float = 1.0
    diffusion_number: float = 0.3
    empty_tolerance: float = 1e-12
    steady_state_tolerance: float = 1e-12  # 0 disables steady-state skipping

    def __post_init__(self):
        if self.n_cells < 4:
            raise ValueError("n_cells must be >= 4")
        if not 0 < self.courant <= 1:
            raise ValueError("courant must be in (0, 1]")
        if not 0 < self.diffusion_number <= 1 / 3:
            raise ValueError("diffusion_number must be in (0, 1/3] for positivity")


@dataclass(frozen=True)
class Steps:
    dz_cm: float
    dt_s: float
    courant_pos: float
    courant_neg: float
    lam_pos: float
    lam_neg: float


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one episode (source starting at t = 0) per cm^2 of electrode area.

    The time series is stored as records: ``record_steps[i]`` consecutive steps that all had the
    per-step values ``*_per_step_cm2[i]`` (one record per simulated step, one per skipped
    steady-state block).
    """

    dt_s: float
    n_steps: int
    produced_cm2: float
    recombined_cm2: float
    collected_pos_cm2: float  # positive ions reaching their collecting electrode
    collected_neg_cm2: float
    remaining_cm2: float  # ion pairs still in the gap at the end (0 when emptied)
    emptied: bool
    simulated_steps: int  # steps actually computed (the rest was skipped at steady state)
    record_steps: np.ndarray
    induced_per_step_cm2: np.ndarray  # induced charge per step / e
    recombined_per_step_cm2: np.ndarray
    source_per_step_cm2: np.ndarray

    @property
    def signal_cm2(self) -> float:
        """Electrometer reading / e, i.e. the time integral of the induced current."""
        return float(np.dot(self.record_steps, self.induced_per_step_cm2))

    @property
    def f_recombination(self) -> float:
        """Fraction of ion pairs not lost to recombination, 1 - R/P."""
        return 1.0 - self.recombined_cm2 / self.produced_cm2

    @property
    def ks(self) -> float:
        """Recombination correction factor P / (P - R)."""
        return 1.0 / self.f_recombination

    @property
    def collection_efficiency(self) -> float:
        """Signal / produced. Equals f_recombination once the gap is empty (no back-diffusion)."""
        return self.signal_cm2 / self.produced_cm2

    @property
    def record_edges_s(self) -> np.ndarray:
        """Start of every record and the end of the last one."""
        return np.concatenate(([0], np.cumsum(self.record_steps))) * self.dt_s

    def cumulative(self, quantity: str, t_s):
        """Integral from the episode start to t of 'induced', 'recombined' or 'source' (per cm^2).

        Exact for the stepped solution (rates are constant within each step).
        """
        per_step = {
            "induced": self.induced_per_step_cm2,
            "recombined": self.recombined_per_step_cm2,
            "source": self.source_per_step_cm2,
        }[quantity]
        totals = np.concatenate(([0.0], np.cumsum(self.record_steps * per_step)))
        return np.interp(t_s, self.record_edges_s, totals)

    @property
    def current_density_A_cm2(self) -> np.ndarray:
        """Induced current density during each record (constant within a record)."""
        return ELEMENTARY_CHARGE_C * self.induced_per_step_cm2 / self.dt_s


@dataclass(frozen=True)
class Uniform1D:
    """Point model for laterally uniform ionisation (no track structure, no space charge)."""

    grid: Grid1D = Grid1D()
    seconds_per_cell_step = SECONDS_PER_CELL_STEP  # class constant for the preflight estimate

    def steps(self, chamber: ParallelPlateChamber, gas: Gas) -> Steps:
        dz = chamber.gap_cm / self.grid.n_cells
        v_pos, v_neg = chamber.drift_velocities_cm_s(gas)
        dt = self.grid.courant * dz / max(v_pos, v_neg)
        d_max = max(gas.D_pos_cm2_s, gas.D_neg_cm2_s)
        if d_max > 0:
            dt = min(dt, self.grid.diffusion_number * dz * dz / d_max)
        return Steps(
            dz_cm=dz,
            dt_s=dt,
            courant_pos=v_pos * dt / dz,
            courant_neg=v_neg * dt / dz,
            lam_pos=gas.D_pos_cm2_s * dt / dz**2,
            lam_neg=gas.D_neg_cm2_s * dt / dz**2,
        )

    def memory_time_s(self, chamber: ParallelPlateChamber, gas: Gas) -> float:
        """Beam-off time after which the gap is empty to far below any tolerance.

        Slow-species transit time plus 8 diffusion widths, with a 25 % margin for the numerical
        tail of the limited scheme.
        """
        t_pos, t_neg = chamber.collection_times_s(gas)
        if t_pos >= t_neg:
            t_slow, d_slow = t_pos, gas.D_pos_cm2_s
        else:
            t_slow, d_slow = t_neg, gas.D_neg_cm2_s
        v_slow = chamber.gap_cm / t_slow
        tail = 8.0 * math.sqrt(2.0 * d_slow * t_slow) / v_slow
        return 1.25 * (t_slow + tail)

    def estimate_cell_steps(
        self, source: PiecewiseConstant, chamber: ParallelPlateChamber, gas: Gas
    ) -> float:
        """Steps that will actually be simulated, times cells.

        Within each constant-rate interval, stepping stops about two memory times after the rate
        changed (steady state), so long constant segments cost the same as short ones.
        """
        dt = self.steps(chamber, gas).dt_s
        memory = self.memory_time_s(chamber, gas)
        widths = np.diff(source.edges)
        simulated_s = float(np.minimum(widths, 3.0 * memory).sum()) + memory
        return (math.ceil(simulated_s / dt) + 2 * source.rates.size) * self.grid.n_cells

    def run_episode(
        self, source: PiecewiseConstant, chamber: ParallelPlateChamber, gas: Gas
    ) -> EpisodeResult:
        """Simulate a source that starts at t = 0 until the gap is empty."""
        st = self.steps(chamber, gas)
        run_len, run_src = source_runs(source, st.dt_s)
        n_src = int(run_len.sum())
        max_steps = n_src + math.ceil(3.0 * self.memory_time_s(chamber, gas) / st.dt_s) + 16
        (produced, recombined, coll_pos, coll_neg, rem_pos, rem_neg, n_steps, emptied,
         rec_count, rec_disp, rec_lost, rec_src) = _simulate(
            run_len, run_src, self.grid.n_cells, st.courant_pos, st.courant_neg, st.lam_pos,
            st.lam_neg, gas.alpha_cm3_s * st.dt_s, max_steps, self.grid.empty_tolerance,
            self.grid.steady_state_tolerance,
        )  # fmt: skip
        dz = st.dz_cm
        return EpisodeResult(
            dt_s=st.dt_s,
            n_steps=int(n_steps),
            produced_cm2=produced * dz,
            recombined_cm2=recombined * dz,
            collected_pos_cm2=coll_pos * dz,
            collected_neg_cm2=coll_neg * dz,
            remaining_cm2=0.5 * (rem_pos + rem_neg) * dz,
            emptied=bool(emptied),
            simulated_steps=int(rec_count.size),
            record_steps=rec_count,
            induced_per_step_cm2=rec_disp * dz * dz / chamber.gap_cm,
            recombined_per_step_cm2=rec_lost * dz,
            source_per_step_cm2=rec_src * dz,
        )
