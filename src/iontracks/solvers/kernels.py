"""Numba kernels shared by the point models.

Conventions: a species is transported in its own "view" of the gap in which it drifts towards
+index in the applied field. Positive ions use the physical order (towards z = gap); negative ions
and free electrons use the reversed order (towards z = 0). Densities are ion pairs per cm^3; face
fluxes are densities moved per step (multiply by dz for per-cm^2 amounts). Displacements are in
units of dz^2 x density: every face crossing counts one cell, and a crossing into an electrode
counts half a cell (cell centre to electrode). The induced (Ramo-Shockley) charge per cm^2 is
displacement x dz^2 / gap.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def react(a, c, x):
    """Exact recombination over a sub-step with x = alpha * dt. Returns the summed density loss."""
    lost = 0.0
    for j in range(a.shape[0]):
        aj = a[j]
        cj = c[j]
        if aj <= 0.0 or cj <= 0.0:
            continue
        ax = aj * x
        cx = cj * x
        # weak recombination: 2nd-order Taylor, error O((n x)^2) < 1e-12
        if ax < 1e-6 and cx < 1e-6:
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
def transport(n, courant, lam, flux):
    """Drift (constant speed, towards +index) and diffuse one species in place.

    ``flux`` is scratch space of length N + 1. Returns (outflow, displacement): the outflow
    through the collecting electrode (a density; multiply by dz) and the displacement.
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
    outflow += diffuse(n, lam, flux)
    for f in range(1, N):
        displacement += flux[f]
    displacement += 0.5 * flux[N]
    return outflow, displacement


@njit(cache=True, nogil=True)
def diffuse(n, lam, flux):
    """Explicit diffusion: reflecting upstream electrode (face 0), absorbing collecting electrode.

    Leaves the face fluxes in ``flux`` (zero if lam == 0) and returns the electrode outflow.
    """
    N = n.shape[0]
    if lam <= 0.0:
        for f in range(N + 1):
            flux[f] = 0.0
        return 0.0
    flux[0] = 0.0
    for f in range(1, N):
        flux[f] = lam * (n[f - 1] - n[f])
    flux[N] = 2.0 * lam * n[N - 1]
    for j in range(N):
        n[j] -= flux[j + 1] - flux[j]
    return flux[N]


@njit(cache=True, nogil=True)
def transport_var(n, cf, lam, flux):
    """Drift with signed face Courant numbers ``cf`` (N + 1, > 0 towards +index), then diffuse.

    Flux-limited (van Leer) upwind with open electrodes: charge leaves through whichever electrode
    the local velocity points to, and nothing enters. The limiter is switched off for the outflow
    of a cell that loses charge through both faces (a field reversal inside the gap), which keeps
    the scheme positive as long as each cell's summed outflow Courant number is <= 1.
    Returns (outflow at face N, outflow at face 0, displacement).
    """
    N = n.shape[0]
    for f in range(N + 1):
        c = cf[f]
        if c > 0.0:
            if f == 0:
                flux[f] = 0.0
                continue
            up = n[f - 1]
            if f == N or (f >= 1 and cf[f - 1] < 0.0):
                flux[f] = c * up
                continue
            dn = n[f] - up
            limited = 0.0
            if dn != 0.0:
                upup = n[f - 2] if f >= 2 else 0.0
                r = (up - upup) / dn
                if r > 0.0:
                    limited = 2.0 * r / (1.0 + r)
            flux[f] = c * (up + 0.5 * (1.0 - c) * limited * dn)
        elif c < 0.0:
            if f == N:
                flux[f] = 0.0
                continue
            up = n[f]
            if f == 0 or cf[f + 1] > 0.0:
                flux[f] = c * up
                continue
            dn = n[f - 1] - up
            limited = 0.0
            if dn != 0.0:
                upup = n[f + 1] if f + 1 < N else 0.0
                r = (up - upup) / dn
                if r > 0.0:
                    limited = 2.0 * r / (1.0 + r)
            flux[f] = c * (up + 0.5 * (1.0 + c) * limited * dn)
        else:
            flux[f] = 0.0
    displacement = 0.5 * (flux[0] + flux[N])
    for f in range(1, N):
        displacement += flux[f]
    out_fwd = flux[N]
    out_bwd = -flux[0]
    for j in range(N):
        n[j] -= flux[j + 1] - flux[j]
    out_fwd += diffuse(n, lam, flux)
    for f in range(1, N):
        displacement += flux[f]
    displacement += 0.5 * flux[N]
    return out_fwd, out_bwd, displacement


@njit(cache=True, nogil=True)
def solve_field(n_pos, n_neg, e0, coef, dz, ef):
    """Field at the faces (physical order, + towards z = gap) for a fixed voltage across the gap.

    Gauss: dE/dz = coef * (n+ - n-) with coef = e / eps0 (V cm), and the mean field stays
    e0 = V / gap. Returns the largest |E - e0| / e0.
    """
    N = n_pos.shape[0]
    q = 0.0
    ef[0] = 0.0
    for f in range(1, N + 1):
        q += (n_pos[f - 1] - n_neg[f - 1]) * dz
        ef[f] = q
    qbar = 0.0
    for j in range(N):
        qbar += 0.5 * (ef[j] + ef[j + 1])
    qbar /= N
    worst = 0.0
    for f in range(N + 1):
        ef[f] = e0 + coef * (ef[f] - qbar)
        dev = abs(ef[f] - e0)
        if dev > worst:
            worst = dev
    return worst / e0


@njit(cache=True, nogil=True)
def attachment_cells(field_V_cm, mu_e, att_a, att_b, att_c, dz):
    """Attachment length in cells for a free electron at the given field magnitude."""
    e = abs(field_V_cm)
    return mu_e * e / (att_a + att_b * math.exp(-att_c * e)) / dz


@njit(cache=True, nogil=True)
def attach(n_rev, prod_rev, ell):
    """Free electrons produced (``prod_rev`` per cell, reversed view) drift towards +index and
    attach with per-cell attachment lengths ``ell`` (in cells), exactly for uniform production and
    constant length within a cell. The attached electrons are added to ``n_rev`` as negative ions.

    Returns (collected, displacement): electrons reaching the anode (a density; multiply by dz)
    and their displacement.
    """
    N = n_rev.shape[0]
    phi = 0.0
    displacement = 0.0
    for k in range(N):
        lk = ell[k]
        p = prod_rev[k]
        if lk <= 1e-300:
            n_rev[k] += phi + p
            phi = 0.0
        else:
            absorb = -math.expm1(-1.0 / lk)  # fraction of the through-going flux attaching here
            escape = lk * absorb  # fraction of the electrons produced here that leave the cell
            n_rev[k] += phi * absorb + p * (1.0 - escape)
            phi = phi * (1.0 - absorb) + p * escape
        displacement += phi if k < N - 1 else 0.5 * phi
    return phi, displacement


@njit(cache=True, nogil=True)
def outflow_courant(cf):
    """Largest summed outflow Courant number of any cell for face Courant numbers ``cf``."""
    N = cf.shape[0] - 1
    worst = 0.0
    for j in range(N):
        out = 0.0
        if cf[j + 1] > 0.0:
            out += cf[j + 1]
        if cf[j] < 0.0:
            out -= cf[j]
        if out > worst:
            worst = out
    return worst


@njit(cache=True, nogil=True)
def grow(arr, n_used):
    out = np.empty(2 * arr.shape[0], dtype=arr.dtype)
    out[:n_used] = arr[:n_used]
    return out
