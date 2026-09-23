"""Track-resolved 3D point model: ion tracks in a laterally periodic tile of the gap.

The tile is an L x L column of the gap (periodic in x and y) that represents one lateral sample
point. Tracks cross the gap perpendicular to the electrodes (along z) at random positions and
times: a Poisson process whose mean follows the ionisation history of the point. Each track
deposits a Gaussian column (``GaussianTrack``), integrated exactly over the cells. The ions then
drift along z, diffuse in 3D and recombine locally. At low fluence rates each track recombines
alone (initial recombination, Jaffe's regime); at high rates the tracks overlap and the result
tends to the uniform model (general recombination, Boag's regime).

Per time step: tracks, 1/2 recombination, lateral diffusion, drift + axial diffusion per column,
1/2 recombination. The column kernels, free electrons and space charge are those of ``Uniform1D``.
Space charge uses the laterally averaged charge (mean field): the lateral field of a single track
(e N0 / (2 pi eps0 r), below 0.1 V/cm for protons) is neglected.

All tallies are per cm^2 of electrode area (tile totals divided by L^2).
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..chamber import ParallelPlateChamber
from ..gas import Gas
from ..time_structure import PiecewiseConstant
from ..tracks import GaussianTrack
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
from .uniform1d import (
    FIELD_COEF_V_CM,
    SPACE_CHARGE_MARGIN,
    EpisodeResult,
    Uniform1D,
    source_runs,
)

SECONDS_PER_CELL_STEP_3D = 6e-8  # single-thread cost per cell-step, both species (measured)


@njit(cache=True, nogil=True)
def _gauss_weights(x0, b, L, n, out):
    """Fractions of a periodic 1D Gaussian exp(-x^2/b^2) / (sqrt(pi) b) centred at x0 per cell."""
    dx = L / n
    for i in range(n):
        out[i] = 0.0
    total = 0.0
    # every periodic image of the Gaussian within 6 b of the tile adds to the tile's n cells
    reach = int(math.ceil(6.0 * b / L)) + 1
    for m in range(-reach, reach + 1):
        c = x0 + m * L
        if c + 6.0 * b < 0.0 or c - 6.0 * b > L:
            continue
        for k in range(n):
            xa = k * dx
            w = 0.5 * (math.erf((xa + dx - c) / b) - math.erf((xa - c) / b))
            out[k] += w
            total += w
    for i in range(n):
        out[i] /= total


@njit(cache=True, nogil=True)
def _mean_profiles(n_pos, n_neg, mean_pos, mean_neg):
    nx, ny, nz = n_pos.shape
    for k in range(nz):
        mean_pos[k] = 0.0
        mean_neg[k] = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                mean_pos[k] += n_pos[i, j, k]
                mean_neg[k] += n_neg[i, j, k]
    inv = 1.0 / (nx * ny)
    for k in range(nz):
        mean_pos[k] *= inv
        mean_neg[k] *= inv


@njit(cache=True, nogil=True)
def _diffuse_lateral(n, lam, tmp):
    """Explicit 2D diffusion in x and y with periodic boundaries (lam = D dt / dx^2 <= 1/4)."""
    nx, ny, nz = n.shape
    for i in range(nx):
        ip = (i + 1) % nx
        im = (i - 1) % nx
        for j in range(ny):
            jp = (j + 1) % ny
            jm = (j - 1) % ny
            for k in range(nz):
                tmp[i, j, k] = n[i, j, k] + lam * (
                    n[ip, j, k] + n[im, j, k] + n[i, jp, k] + n[i, jm, k] - 4.0 * n[i, j, k]
                )
    n[:] = tmp


@njit(cache=True, nogil=True)
def _simulate3d(
    nx, nz, dx, dz, dt, e0, mu_pos, mu_neg, lam_xy_pos, lam_xy_neg, lam_z_pos, lam_z_neg,
    alpha_dt, fe_on, mu_e, att_a, att_b, att_c, sc_on, sc_coef, trk_step, trk_x, trk_y,
    n0_per_cm, b, n_src_steps, max_steps, empty_tol, snap_steps,
):  # fmt: skip
    ny = nx
    L = nx * dx
    n_cells = nx * ny * nz
    n_pos = np.zeros((nx, ny, nz))
    n_neg = np.zeros((nx, ny, nz))
    flat_pos = n_pos.reshape(n_cells)
    flat_neg = n_neg.reshape(n_cells)
    tmp = np.zeros((nx, ny, nz))
    flux = np.zeros(nz + 1)
    ef = np.full(nz + 1, e0)
    cf_pos = np.zeros(nz + 1)
    cf_neg = np.zeros(nz + 1)
    mean_pos = np.zeros(nz)
    mean_neg = np.zeros(nz)
    ell = np.zeros(nz)
    prod = np.zeros(nz)
    dep = np.zeros((nx, ny))
    gx = np.zeros(nx)
    gy = np.zeros(ny)
    ell0 = attachment_cells(e0, mu_e, att_a, att_b, att_c, dz) if fe_on else 0.0
    c_pos0 = mu_pos * e0 * dt / dz
    c_neg0 = mu_neg * e0 * dt / dz
    per_cell = n0_per_cm / (dx * dx)  # density of a whole track cross-section in one cell
    cap = 1024
    rec_disp = np.empty(cap)
    rec_lost = np.empty(cap)
    rec_src = np.empty(cap)
    n_snap = snap_steps.shape[0]
    snap_xz_pos = np.zeros((n_snap, nx, nz))
    snap_xz_neg = np.zeros((n_snap, nx, nz))
    snap_mean_pos = np.zeros((n_snap, nz))
    snap_mean_neg = np.zeros((n_snap, nz))
    snap_e = np.zeros((n_snap, nz + 1))
    i_snap = 0
    j_mid = ny // 2
    produced = 0.0
    recombined = 0.0
    pos_fwd = 0.0
    pos_bwd = 0.0
    neg_fwd = 0.0
    neg_bwd = 0.0
    free_total = 0.0
    half = 0.5 * alpha_dt
    n_trk = trk_step.shape[0]
    i_trk = 0
    max_sub = 1
    step = 0
    emptied = False
    while step < max_steps:
        # --- tracks arriving during this step
        has_src = False
        while i_trk < n_trk and trk_step[i_trk] == step:
            if not has_src:
                for i in range(nx):
                    for j in range(ny):
                        dep[i, j] = 0.0
                has_src = True
            _gauss_weights(trk_x[i_trk], b, L, nx, gx)
            _gauss_weights(trk_y[i_trk], b, L, ny, gy)
            for i in range(nx):
                if gx[i] == 0.0:
                    continue
                for j in range(ny):
                    dep[i, j] += per_cell * gx[i] * gy[j]
            i_trk += 1
        distorted = False
        if sc_on:
            _mean_profiles(n_pos, n_neg, mean_pos, mean_neg)
            distorted = solve_field(mean_pos, mean_neg, e0, sc_coef, dz, ef) > 1e-12
        free = 0.0
        d = 0.0
        src = 0.0
        if has_src:
            if fe_on:
                for k in range(nz):
                    if distorted:
                        kk = nz - 1 - k
                        ell[k] = attachment_cells(
                            0.5 * (ef[kk] + ef[kk + 1]), mu_e, att_a, att_b, att_c, dz
                        )
                    else:
                        ell[k] = ell0
            for i in range(nx):
                for j in range(ny):
                    v = dep[i, j]
                    if v == 0.0:
                        continue
                    src += v * nz
                    for k in range(nz):
                        n_pos[i, j, k] += v
                    if fe_on:
                        for k in range(nz):
                            prod[k] = v
                        fc, de = attach(n_neg[i, j, ::-1], prod, ell)
                        free += fc
                        d += de
                    else:
                        for k in range(nz):
                            n_neg[i, j, k] += v
            produced += src
        lost = react(flat_pos, flat_neg, half) if half > 0.0 else 0.0
        # --- lateral diffusion (once per step)
        if lam_xy_pos > 0.0:
            _diffuse_lateral(n_pos, lam_xy_pos, tmp)
        if lam_xy_neg > 0.0:
            _diffuse_lateral(n_neg, lam_xy_neg, tmp)
        # --- drift and axial diffusion per column
        o1f = 0.0
        o1b = 0.0
        o2f = 0.0
        o2b = 0.0
        use_var = False
        n_relax = 1
        if sc_on:
            _mean_profiles(n_pos, n_neg, mean_pos, mean_neg)
            distortion = solve_field(mean_pos, mean_neg, e0, sc_coef, dz, ef)
            sigma = 0.0
            for k in range(nz):
                sk = mu_pos * mean_pos[k] + mu_neg * mean_neg[k]
                if sk > sigma:
                    sigma = sk
            n_relax = max(1, int(math.ceil(dt * sc_coef * sigma / 0.9)))
            use_var = distortion > 1e-12 or n_relax > 1
        if use_var:
            n_sub = 1
            sub = 0
            while sub < n_sub:
                if sub > 0:
                    _mean_profiles(n_pos, n_neg, mean_pos, mean_neg)
                    solve_field(mean_pos, mean_neg, e0, sc_coef, dz, ef)
                for f in range(nz + 1):
                    cf_pos[f] = mu_pos * ef[f] * dt / (dz * n_sub)
                    cf_neg[f] = mu_neg * ef[nz - f] * dt / (dz * n_sub)
                if sub == 0:
                    worst = max(outflow_courant(cf_pos), outflow_courant(cf_neg))
                    n_sub = n_relax
                    if worst > 1.0:
                        n_sub = max(n_sub, int(math.ceil(worst / 0.9)))
                    for f in range(nz + 1):
                        cf_pos[f] /= n_sub
                        cf_neg[f] /= n_sub
                for i in range(nx):
                    for j in range(ny):
                        a1, b1, c1 = transport_var(n_pos[i, j, :], cf_pos, lam_z_pos / n_sub, flux)
                        a2, b2, c2 = transport_var(
                            n_neg[i, j, ::-1], cf_neg, lam_z_neg / n_sub, flux
                        )
                        o1f += a1
                        o1b += b1
                        o2f += a2
                        o2b += b2
                        d += c1 + c2
                sub += 1
            if n_sub > max_sub:
                max_sub = n_sub
        else:
            for i in range(nx):
                for j in range(ny):
                    a1, c1 = transport(n_pos[i, j, :], c_pos0, lam_z_pos, flux)
                    a2, c2 = transport(n_neg[i, j, ::-1], c_neg0, lam_z_neg, flux)
                    o1f += a1
                    o2f += a2
                    d += c1 + c2
        if half > 0.0:
            lost += react(flat_pos, flat_neg, half)
        recombined += lost
        pos_fwd += o1f
        pos_bwd += o1b
        neg_fwd += o2f + free
        neg_bwd += o2b
        free_total += free
        if step == rec_disp.shape[0]:
            rec_disp = grow(rec_disp, step)
            rec_lost = grow(rec_lost, step)
            rec_src = grow(rec_src, step)
        rec_disp[step] = d
        rec_lost[step] = lost
        rec_src[step] = src
        step += 1
        while i_snap < n_snap and snap_steps[i_snap] < step:
            _mean_profiles(n_pos, n_neg, mean_pos, mean_neg)
            if sc_on:
                solve_field(mean_pos, mean_neg, e0, sc_coef, dz, ef)
            snap_xz_pos[i_snap] = n_pos[:, j_mid, :]
            snap_xz_neg[i_snap] = n_neg[:, j_mid, :]
            snap_mean_pos[i_snap] = mean_pos
            snap_mean_neg[i_snap] = mean_neg
            snap_e[i_snap] = ef
            i_snap += 1
        if step >= n_src_steps and i_trk >= n_trk:
            content = 0.0
            for c in range(n_cells):
                content += flat_pos[c] + flat_neg[c]
            if content <= 2.0 * empty_tol * produced:
                emptied = True
                break
    remaining = 0.0
    for c in range(n_cells):
        remaining += 0.5 * (flat_pos[c] + flat_neg[c])
    return (
        produced, recombined, pos_fwd, pos_bwd, neg_fwd, neg_bwd, free_total, remaining, step,
        emptied, max_sub, rec_disp[:step].copy(), rec_lost[:step].copy(), rec_src[:step].copy(),
        snap_xz_pos[:i_snap].copy(), snap_xz_neg[:i_snap].copy(), snap_mean_pos[:i_snap].copy(),
        snap_mean_neg[:i_snap].copy(), snap_e[:i_snap].copy(),
    )  # fmt: skip


@dataclass(frozen=True)
class Grid3D:
    """Discretisation of the tile.

    ``tile_cm`` None sizes the tile automatically (see ``Tile3D.tile_size_cm``). The lateral cell
    is tile / n_xy; the time step obeys the drift Courant number (faster species), D dt / dz^2 <=
    ``diffusion_number`` and D dt / dx^2 <= ``lateral_diffusion_number`` (explicit schemes).
    """

    n_xy: int = 48
    n_z: int = 100
    tile_cm: float | None = None
    courant: float = 1.0
    diffusion_number: float = 0.3
    lateral_diffusion_number: float = 0.2
    empty_tolerance: float = 1e-12

    def __post_init__(self):
        if self.n_xy < 4 or self.n_z < 4:
            raise ValueError("n_xy and n_z must be >= 4")
        if not 0 < self.courant <= 1:
            raise ValueError("courant must be in (0, 1]")
        if not 0 < self.diffusion_number <= 1 / 3:
            raise ValueError("diffusion_number must be in (0, 1/3]")
        if not 0 < self.lateral_diffusion_number <= 1 / 4:
            raise ValueError("lateral_diffusion_number must be in (0, 1/4]")


@dataclass(frozen=True)
class TrackSnapshots:
    """Tile state after selected steps: an x-z slice through the tile centre and lateral means."""

    times_s: np.ndarray  # (n,)
    x_cm: np.ndarray  # (nx,) cell centres
    z_cm: np.ndarray  # (nz,) cell centres, anode at z = 0
    n_pos_xz_cm3: np.ndarray  # (n, nx, nz)
    n_neg_xz_cm3: np.ndarray
    n_pos_mean_cm3: np.ndarray  # (n, nz) laterally averaged
    n_neg_mean_cm3: np.ndarray
    field_V_cm: np.ndarray  # (n, nz + 1) mean field at the faces


@dataclass(frozen=True)
class Tile3D:
    """Track-resolved point model (duck-compatible with ``Uniform1D``).

    ``track`` fixes the particle: line density N0 and Gaussian radius b. The ionisation history
    of a point, q(t) in ion pairs cm^-3 s^-1, becomes a fluence rate q / N0 of tracks; arrivals are
    Poisson with a reproducible ``seed``. Results are stochastic when tracks overlap; a single
    track (``single_track``) is deterministic.
    """

    track: GaussianTrack
    grid: Grid3D = Grid3D()
    free_electrons: bool = True
    space_charge: bool = True
    seed: int = 0
    seconds_per_cell_step = SECONDS_PER_CELL_STEP_3D

    # ------------------------------------------------------------------ geometry and steps
    def tile_size_cm(self, chamber: ParallelPlateChamber, gas: Gas) -> float:
        """Lateral tile size: the given one, or 3 track widths after one slow-ion transit (at
        least 6 track radii). The track images then meet only when their density has fallen; the
        lateral cell size relative to the track radius sets the accuracy (validation fig. 9c)."""
        if self.grid.tile_cm is not None:
            return self.grid.tile_cm
        t_slow = max(chamber.collection_times_s(gas))
        d_max = max(gas.D_pos_cm2_s, gas.D_neg_cm2_s)
        width = math.sqrt(self.track.radius_cm**2 + 4.0 * d_max * t_slow)
        return max(3.0 * width, 6.0 * self.track.radius_cm)

    def steps(self, chamber: ParallelPlateChamber, gas: Gas) -> tuple[float, float, float]:
        """(dx, dz, dt)."""
        dx = self.tile_size_cm(chamber, gas) / self.grid.n_xy
        dz = chamber.gap_cm / self.grid.n_z
        v_pos, v_neg = chamber.drift_velocities_cm_s(gas)
        margin = SPACE_CHARGE_MARGIN if self.space_charge else 1.0
        dt = self.grid.courant * dz / (margin * max(v_pos, v_neg))
        d_max = max(gas.D_pos_cm2_s, gas.D_neg_cm2_s)
        if d_max > 0:
            dt = min(
                dt,
                self.grid.diffusion_number * dz * dz / d_max,
                self.grid.lateral_diffusion_number * dx * dx / d_max,
            )
        return dx, dz, dt

    def memory_time_s(self, chamber: ParallelPlateChamber, gas: Gas) -> float:
        return Uniform1D().memory_time_s(chamber, gas)

    def estimate_cell_steps(
        self, source: PiecewiseConstant, chamber: ParallelPlateChamber, gas: Gas
    ) -> float:
        """Every step is simulated (no steady-state skipping for random tracks)."""
        _, _, dt = self.steps(chamber, gas)
        steps = math.ceil((source.t_end_s + 3.0 * self.memory_time_s(chamber, gas)) / dt)
        return float(steps * self.grid.n_xy**2 * self.grid.n_z)

    # ------------------------------------------------------------------ runs
    def run_episode(
        self,
        source: PiecewiseConstant,
        chamber: ParallelPlateChamber,
        gas: Gas,
        snapshot_times_s=None,
    ) -> EpisodeResult:
        """Simulate the ionisation history q(t) of a point as Poisson-distributed tracks."""
        _, _, dt = self.steps(chamber, gas)
        L = self.tile_size_cm(chamber, gas)
        run_len, run_src = source_runs(source, dt)
        rng = np.random.default_rng(self.seed)
        steps, n_before = [], 0
        for length, amount in zip(run_len, run_src, strict=True):
            mean = length * amount * L * L / self.track.pairs_per_cm
            n = rng.poisson(mean) if mean > 0 else 0
            steps.append(n_before + rng.integers(0, length, n))
            n_before += int(length)
        trk_step = np.sort(np.concatenate(steps)).astype(np.int64) if steps else np.zeros(0, int)
        trk_x = rng.uniform(0, L, trk_step.size)
        trk_y = rng.uniform(0, L, trk_step.size)
        return self._run(trk_step, trk_x, trk_y, n_before, chamber, gas, snapshot_times_s)

    def single_track(
        self, chamber: ParallelPlateChamber, gas: Gas, snapshot_times_s=None
    ) -> EpisodeResult:
        """One track through the centre of the tile at t = 0 (initial recombination)."""
        dx, _, _ = self.steps(chamber, gas)
        centre = (self.grid.n_xy // 2 + 0.5) * dx
        return self._run(
            np.zeros(1, np.int64),
            np.array([centre]),
            np.array([centre]),
            1,
            chamber,
            gas,
            snapshot_times_s,
        )

    def _run(self, trk_step, trk_x, trk_y, n_src_steps, chamber, gas, snapshot_times_s):
        if self.free_electrons and not gas.has_free_electrons:
            raise ValueError(
                f"gas {gas.name!r} has no free-electron data: use Gas.air(...) or "
                "Tile3D(..., free_electrons=False)"
            )
        dx, dz, dt = self.steps(chamber, gas)
        n = self.grid.n_xy
        drain = 20.0 if self.space_charge else 3.0
        max_steps = n_src_steps + math.ceil(drain * self.memory_time_s(chamber, gas) / dt) + 16
        if snapshot_times_s is None:
            snap_steps = np.zeros(0, dtype=np.int64)
        else:
            snap_steps = np.unique(
                np.maximum(np.floor(np.asarray(snapshot_times_s, float) / dt), 0)
            ).astype(np.int64)
        fe = self.free_electrons
        att = gas.attachment_rate_s if fe else (1.0, 0.0, 0.0)
        (produced, recombined, pos_fwd, pos_bwd, neg_fwd, neg_bwd, free, remaining, n_steps,
         emptied, max_sub, rec_disp, rec_lost, rec_src, s_pos, s_neg, s_mpos, s_mneg,
         s_e) = _simulate3d(
            n, self.grid.n_z, dx, dz, dt, chamber.field_V_cm, gas.mu_pos_cm2_Vs,
            gas.mu_neg_cm2_Vs, gas.D_pos_cm2_s * dt / dx**2, gas.D_neg_cm2_s * dt / dx**2,
            gas.D_pos_cm2_s * dt / dz**2, gas.D_neg_cm2_s * dt / dz**2, gas.alpha_cm3_s * dt,
            fe, gas.electron_mobility_cm2_Vs if fe else 0.0, att[0], att[1], att[2],
            self.space_charge, FIELD_COEF_V_CM, trk_step, trk_x, trk_y,
            self.track.pairs_per_cm, self.track.radius_cm, n_src_steps, max_steps,
            self.grid.empty_tolerance, snap_steps,
        )  # fmt: skip
        per_cm2 = dz / n**2  # cell-density sums -> ion pairs per cm^2 of electrode
        snapshots = None
        if snap_steps.size:
            snapshots = TrackSnapshots(
                times_s=(snap_steps[: s_pos.shape[0]] + 1) * dt,
                x_cm=(np.arange(n) + 0.5) * dx,
                z_cm=(np.arange(self.grid.n_z) + 0.5) * dz,
                n_pos_xz_cm3=s_pos,
                n_neg_xz_cm3=s_neg,
                n_pos_mean_cm3=s_mpos,
                n_neg_mean_cm3=s_mneg,
                field_V_cm=s_e,
            )
        return EpisodeResult(
            dt_s=dt,
            n_steps=int(n_steps),
            produced_cm2=produced * per_cm2,
            recombined_cm2=recombined * per_cm2,
            collected_pos_cm2=(pos_fwd + pos_bwd) * per_cm2,
            collected_neg_cm2=(neg_fwd + neg_bwd) * per_cm2,
            free_electrons_cm2=free * per_cm2,
            backflow_pos_cm2=pos_bwd * per_cm2,
            backflow_neg_cm2=neg_bwd * per_cm2,
            remaining_cm2=remaining * per_cm2,
            emptied=bool(emptied),
            simulated_steps=int(n_steps),
            max_substeps=int(max_sub),
            record_steps=np.ones(rec_disp.size, dtype=np.int64),
            induced_per_step_cm2=rec_disp * per_cm2 * dz / chamber.gap_cm,
            recombined_per_step_cm2=rec_lost * per_cm2,
            source_per_step_cm2=rec_src * per_cm2,
            snapshots=snapshots,
        )
