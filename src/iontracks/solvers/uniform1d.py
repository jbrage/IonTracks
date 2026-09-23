"""1D drift-diffusion-recombination solver for laterally uniform ionisation.

Per time step (Strang splitting): source, 1/2 recombination, drift + diffusion of each species,
1/2 recombination.

* Source: the exact integral of the rate over the step, spread uniformly over the gap.
* Free electrons (optional): the electrons of each step drift towards the anode and attach to
  neutral molecules with a field-dependent attachment length. Attachment (~10 ns) is far shorter
  than a time step, so it is applied instantly within the step (exact exponential kernel, per
  cell). Electrons that reach the anode first are collected directly, and electron-ion
  recombination is neglected (Boag et al. 1996, Fenwick & Kumar 2023).
* Recombination: exact local solution of dn+/dt = dn-/dt = -alpha n+ n-. It is unconditionally
  stable and positive.
* Drift: finite-volume, flux-limited (van Leer) upwind scheme. It is positivity preserving for
  Courant <= 1 and exact at Courant = 1 (the faster species by default).
* Space charge (optional): the field follows Gauss's law for the net ion charge at a fixed
  voltage. In a distorted field the drift uses face velocities and sub-cycles so that no cell
  loses more than its content; charge may then leave through either electrode.
* Diffusion: explicit. The collecting electrode absorbs and the upstream electrode reflects.
  Diffusion against the field (loss ~ kT/(eV) ~ 1e-4) is not modelled: its boundary layer
  D/v ~ 0.3 um cannot be resolved by the grid. D dt / dz^2 <= 1/3 keeps it positive.
* Signal: the induced (Ramo-Shockley) current, from the displacement of every face flux,
  including the free electrons. After complete collection its time integral equals the charge
  an electrometer reads.

All tallies are per cm^2 of electrode area.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..chamber import ParallelPlateChamber
from ..constants import ELEMENTARY_CHARGE_C, VACUUM_PERMITTIVITY_F_CM
from ..gas import Gas
from ..time_structure import PiecewiseConstant
from .kernels import (
    attach,
    attachment_cells,
    grow,
    outflow_courant,
    react,
    solve_field,
    transport,
    transport_var,
)

# single-thread cost per cell and step (both species), measured 24-40 ns on a 2024 laptop;
# used only for the preflight estimate
SECONDS_PER_CELL_STEP = 4e-8
SPACE_CHARGE_COST = 2.5  # relative cost of a cell-step with space charge (Poisson, face speeds)
SPACE_CHARGE_MARGIN = 1.05  # time-step margin: field distortions below 5 % need no sub-cycling
FIELD_COEF_V_CM = ELEMENTARY_CHARGE_C / VACUUM_PERMITTIVITY_F_CM  # e / eps0


@njit(cache=True, nogil=True)
def _simulate(
    run_len, run_src, n_cells, dz, dt, e0, mu_pos, mu_neg, lam_pos, lam_neg, alpha_dt,
    fe_on, mu_e, att_a, att_b, att_c, sc_on, sc_coef, max_steps, empty_tol, ss_tol, snap_steps,
):  # fmt: skip
    """Step through source runs (constant source per step within a run) until the gap is empty.

    Steady-state skipping: when, for three consecutive steps of a run with constant source, each
    species' production equals its outflow plus recombination and the per-step tallies are
    unchanged (both to ``ss_tol`` relative), the state is a fixed point of the scheme and the rest
    of the run is tallied without stepping. Records (count, per-step values) keep the time series
    compact: one record per simulated step, one per skipped block.
    """
    N = n_cells
    n_pos = np.zeros(N)
    n_neg = np.zeros(N)
    n_neg_rev = n_neg[::-1]  # negative ions drift towards index 0: transport them reversed
    flux = np.zeros(N + 1)
    ef = np.full(N + 1, e0)
    cf_pos = np.zeros(N + 1)
    cf_neg = np.zeros(N + 1)
    prod = np.zeros(N)
    ell = np.zeros(N)
    ell0 = attachment_cells(e0, mu_e, att_a, att_b, att_c, dz) if fe_on else 0.0
    for k in range(N):
        ell[k] = ell0
    c_pos0 = mu_pos * e0 * dt / dz
    c_neg0 = mu_neg * e0 * dt / dz
    cap = 1024
    rec_count = np.empty(cap, dtype=np.int64)
    rec_disp = np.empty(cap)
    rec_lost = np.empty(cap)
    rec_src = np.empty(cap)
    n_rec = 0
    n_snap = snap_steps.shape[0]
    snap_pos = np.zeros((n_snap, N))
    snap_neg = np.zeros((n_snap, N))
    snap_e = np.zeros((n_snap, N + 1))
    i_snap = 0
    # tallies: production, recombination, collection per species and electrode, free electrons
    produced = 0.0
    recombined = 0.0
    pos_fwd = 0.0
    pos_bwd = 0.0
    neg_fwd = 0.0
    neg_bwd = 0.0
    free_total = 0.0
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
    max_sub = 1
    emptied = False
    while step < max_steps:
        s = run_src[run] if run < n_runs else 0.0
        distorted = False
        if sc_on:
            distorted = solve_field(n_pos, n_neg, e0, sc_coef, dz, ef) > 1e-12
        free = 0.0
        d_e = 0.0
        if s > 0.0:
            for j in range(N):
                n_pos[j] += s
            if fe_on:
                if distorted:
                    for k in range(N):
                        j = N - 1 - k
                        ell[k] = attachment_cells(
                            0.5 * (ef[j] + ef[j + 1]), mu_e, att_a, att_b, att_c, dz
                        )
                else:
                    for k in range(N):
                        ell[k] = ell0
                for k in range(N):
                    prod[k] = s
                free, d_e = attach(n_neg_rev, prod, ell)
            else:
                for j in range(N):
                    n_neg[j] += s
            produced += s * N
        lost = react(n_pos, n_neg, half) if half > 0.0 else 0.0
        o1f = 0.0
        o1b = 0.0
        o2f = 0.0
        o2b = 0.0
        d = d_e
        use_var = False
        n_relax = 1
        if sc_on:
            distortion = solve_field(n_pos, n_neg, e0, sc_coef, dz, ef)
            # dielectric relaxation time eps0 / sigma: an explicit drift-Poisson coupling must
            # resolve it, or it overshoots and reverses the field
            sigma = 0.0
            for j in range(N):
                sj = mu_pos * n_pos[j] + mu_neg * n_neg[j]
                if sj > sigma:
                    sigma = sj
            n_relax = max(1, int(math.ceil(dt * sc_coef * sigma / 0.9)))
            use_var = distortion > 1e-12 or n_relax > 1
        if use_var:
            # sub-cycle the drift: no cell may lose more than 90 % of its content per sub-step,
            # and each sub-step stays within the dielectric relaxation time
            n_sub = 1
            sub = 0
            while sub < n_sub:
                if sub > 0:
                    solve_field(n_pos, n_neg, e0, sc_coef, dz, ef)
                for f in range(N + 1):
                    cf_pos[f] = mu_pos * ef[f] * dt / (dz * n_sub)
                    cf_neg[f] = mu_neg * ef[N - f] * dt / (dz * n_sub)
                if sub == 0:
                    worst = max(outflow_courant(cf_pos), outflow_courant(cf_neg))
                    n_sub = n_relax
                    if worst > 1.0:
                        n_sub = max(n_sub, int(math.ceil(worst / 0.9)))
                    for f in range(N + 1):
                        cf_pos[f] /= n_sub
                        cf_neg[f] /= n_sub
                a, b, c = transport_var(n_pos, cf_pos, lam_pos / n_sub, flux)
                o1f += a
                o1b += b
                d += c
                a, b, c = transport_var(n_neg_rev, cf_neg, lam_neg / n_sub, flux)
                o2f += a
                o2b += b
                d += c
                sub += 1
            if n_sub > max_sub:
                max_sub = n_sub
        else:
            o1f, dd = transport(n_pos, c_pos0, lam_pos, flux)
            d += dd
            o2f, dd = transport(n_neg_rev, c_neg0, lam_neg, flux)
            d += dd
        if half > 0.0:
            lost += react(n_pos, n_neg, half)
        o2f += free
        out1 = o1f + o1b
        out2 = o2f + o2b
        recombined += lost
        pos_fwd += o1f
        pos_bwd += o1b
        neg_fwd += o2f
        neg_bwd += o2b
        free_total += free
        if n_rec == rec_count.shape[0]:
            rec_count = grow(rec_count, n_rec)
            rec_disp = grow(rec_disp, n_rec)
            rec_lost = grow(rec_lost, n_rec)
            rec_src = grow(rec_src, n_rec)
        rec_count[n_rec] = 1
        rec_disp[n_rec] = d
        rec_lost[n_rec] = lost
        rec_src[n_rec] = s * N
        n_rec += 1
        step += 1
        if run < n_runs:
            left -= 1
            scale = s * N
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
                produced += s * N * k
                recombined += lost * k
                pos_fwd += o1f * k
                pos_bwd += o1b * k
                neg_fwd += o2f * k
                neg_bwd += o2b * k
                free_total += free * k
                if n_rec == rec_count.shape[0]:
                    rec_count = grow(rec_count, n_rec)
                    rec_disp = grow(rec_disp, n_rec)
                    rec_lost = grow(rec_lost, n_rec)
                    rec_src = grow(rec_src, n_rec)
                rec_count[n_rec] = k
                rec_disp[n_rec] = d
                rec_lost[n_rec] = lost
                rec_src[n_rec] = s * N
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
        # snapshots of the state after the steps done so far (constant inside a skipped block)
        while i_snap < n_snap and snap_steps[i_snap] < step:
            if sc_on:
                solve_field(n_pos, n_neg, e0, sc_coef, dz, ef)
            snap_pos[i_snap, :] = n_pos
            snap_neg[i_snap, :] = n_neg
            snap_e[i_snap, :] = ef
            i_snap += 1
        if run >= n_runs:
            content = 0.0
            for j in range(N):
                content += n_pos[j] + n_neg[j]
            if content <= 2.0 * empty_tol * produced:
                emptied = True
                break
    remaining_pos = 0.0
    remaining_neg = 0.0
    for j in range(N):
        remaining_pos += n_pos[j]
        remaining_neg += n_neg[j]
    return (
        produced, recombined, pos_fwd, pos_bwd, neg_fwd, neg_bwd, free_total, remaining_pos,
        remaining_neg, step, emptied, max_sub, rec_count[:n_rec].copy(), rec_disp[:n_rec].copy(),
        rec_lost[:n_rec].copy(), rec_src[:n_rec].copy(), snap_pos[:i_snap].copy(),
        snap_neg[:i_snap].copy(), snap_e[:i_snap].copy(),
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
class Snapshots:
    """Gap profiles after selected steps (index 0 of the space axis is z = 0, the anode)."""

    times_s: np.ndarray  # (n,)
    z_cm: np.ndarray  # (N,) cell centres
    n_pos_cm3: np.ndarray  # (n, N)
    n_neg_cm3: np.ndarray  # (n, N)
    field_V_cm: np.ndarray  # (n, N + 1) at the cell faces; + points from anode to cathode


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
    collected_pos_cm2: float  # positive ions reaching an electrode
    collected_neg_cm2: float  # negative charge (ions and free electrons) reaching an electrode
    free_electrons_cm2: float  # electrons collected before attaching
    backflow_pos_cm2: float  # positive ions collected on the anode (field reversal)
    backflow_neg_cm2: float  # negative ions collected on the cathode (field reversal)
    remaining_cm2: float  # ion pairs still in the gap at the end (0 when emptied)
    emptied: bool
    simulated_steps: int  # steps actually computed (the rest was skipped at steady state)
    max_substeps: int  # largest drift sub-cycling of a step (space charge)
    record_steps: np.ndarray
    induced_per_step_cm2: np.ndarray  # induced charge per step / e
    recombined_per_step_cm2: np.ndarray
    source_per_step_cm2: np.ndarray
    snapshots: Snapshots | None = None

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
        """Signal / produced. Equals f_recombination once the gap is empty and no charge was
        collected on the wrong electrode (no back-diffusion is modelled)."""
        return self.signal_cm2 / self.produced_cm2

    @property
    def free_electron_fraction(self) -> float:
        return self.free_electrons_cm2 / self.produced_cm2

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
    """Point model for laterally uniform ionisation (no track structure).

    ``free_electrons`` needs a gas with free-electron data (``Gas.air`` has it).
    ``space_charge`` makes the field follow the ion charge in the gap. Both are on by default;
    switch them off to compare with Boag's classic theories.
    """

    grid: Grid1D = Grid1D()
    free_electrons: bool = True
    space_charge: bool = True
    seconds_per_cell_step = SECONDS_PER_CELL_STEP  # class constant for the preflight estimate

    def steps(self, chamber: ParallelPlateChamber, gas: Gas) -> Steps:
        dz = chamber.gap_cm / self.grid.n_cells
        v_pos, v_neg = chamber.drift_velocities_cm_s(gas)
        margin = SPACE_CHARGE_MARGIN if self.space_charge else 1.0
        dt = self.grid.courant * dz / (margin * max(v_pos, v_neg))
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
        tail of the limited scheme. Strong space charge slows the collection down; the solver then
        warns (episode not emptied, or episodes interacting).
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
        """Steps that will actually be simulated, times cells (weighted by the relative cost).

        Within each constant-rate interval, stepping stops about two memory times after the rate
        changed (steady state), so long constant segments cost the same as short ones.
        """
        dt = self.steps(chamber, gas).dt_s
        memory = self.memory_time_s(chamber, gas)
        widths = np.diff(source.edges)
        simulated_s = float(np.minimum(widths, 3.0 * memory).sum()) + memory
        cost = SPACE_CHARGE_COST if self.space_charge else 1.0
        return cost * (math.ceil(simulated_s / dt) + 2 * source.rates.size) * self.grid.n_cells

    def run_episode(
        self,
        source: PiecewiseConstant,
        chamber: ParallelPlateChamber,
        gas: Gas,
        snapshot_times_s=None,
    ) -> EpisodeResult:
        """Simulate a source that starts at t = 0 until the gap is empty.

        ``snapshot_times_s`` (optional) records the gap profiles at the end of the steps that
        contain these times.
        """
        if self.free_electrons and not gas.has_free_electrons:
            raise ValueError(
                f"gas {gas.name!r} has no free-electron data: use Gas.air(...) or "
                "Uniform1D(free_electrons=False)"
            )
        st = self.steps(chamber, gas)
        run_len, run_src = source_runs(source, st.dt_s)
        n_src = int(run_len.sum())
        # strong space charge screens the field and slows the collection down: allow longer
        drain = 20.0 if self.space_charge else 3.0
        max_steps = n_src + math.ceil(drain * self.memory_time_s(chamber, gas) / st.dt_s) + 16
        if snapshot_times_s is None:
            snap_steps = np.zeros(0, dtype=np.int64)
        else:
            snap_steps = np.unique(
                np.maximum(np.floor(np.asarray(snapshot_times_s, float) / st.dt_s), 0)
            ).astype(np.int64)
        fe = self.free_electrons
        mu_e = gas.electron_mobility_cm2_Vs if fe else 0.0
        att = gas.attachment_rate_s if fe else (1.0, 0.0, 0.0)
        (produced, recombined, pos_fwd, pos_bwd, neg_fwd, neg_bwd, free, rem_pos, rem_neg,
         n_steps, emptied, max_sub, rec_count, rec_disp, rec_lost, rec_src, s_pos, s_neg,
         s_e) = _simulate(
            run_len, run_src, self.grid.n_cells, st.dz_cm, st.dt_s, chamber.field_V_cm,
            gas.mu_pos_cm2_Vs, gas.mu_neg_cm2_Vs, st.lam_pos, st.lam_neg,
            gas.alpha_cm3_s * st.dt_s, fe, mu_e, att[0], att[1], att[2], self.space_charge,
            FIELD_COEF_V_CM, max_steps, self.grid.empty_tolerance,
            self.grid.steady_state_tolerance, snap_steps,
        )  # fmt: skip
        dz = st.dz_cm
        snapshots = None
        if snap_steps.size:
            snapshots = Snapshots(
                times_s=(snap_steps[: s_pos.shape[0]] + 1) * st.dt_s,
                z_cm=(np.arange(self.grid.n_cells) + 0.5) * dz,
                n_pos_cm3=s_pos,
                n_neg_cm3=s_neg,
                field_V_cm=s_e,
            )
        return EpisodeResult(
            dt_s=st.dt_s,
            n_steps=int(n_steps),
            produced_cm2=produced * dz,
            recombined_cm2=recombined * dz,
            collected_pos_cm2=(pos_fwd + pos_bwd) * dz,
            collected_neg_cm2=(neg_fwd + neg_bwd) * dz,
            free_electrons_cm2=free * dz,
            backflow_pos_cm2=pos_bwd * dz,
            backflow_neg_cm2=neg_bwd * dz,
            remaining_cm2=0.5 * (rem_pos + rem_neg) * dz,
            emptied=bool(emptied),
            simulated_steps=int(rec_count.size),
            max_substeps=int(max_sub),
            record_steps=rec_count,
            induced_per_step_cm2=rec_disp * dz * dz / chamber.gap_cm,
            recombined_per_step_cm2=rec_lost * dz,
            source_per_step_cm2=rec_src * dz,
            snapshots=snapshots,
        )
