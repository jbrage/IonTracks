"""Transport and ionisation parameters of the chamber gas."""

import math
from dataclasses import dataclass, replace

from .constants import (
    BOLTZMANN_EV_PER_K,
    DRY_AIR_DENSITY_REF_G_CM3,
    ELEMENTARY_CHARGE_C,
    P_REF_HPA,
    T_REF_K,
)

# Mean energy per ion pair in dry air, ICRU Report 90 (2016). Photons deposit dose through
# secondary electrons, so they share the electron value.
W_AIR_EV = {"electron": 33.97, "photon": 33.97, "proton": 34.44, "carbon": 34.71}

# Free electrons in air (Goetz et al. 2013, as tabulated by Fenwick & Kumar 2023, Phys. Med. Biol.
# 68 015016, table 1): drift mobility 0.083 m^2/Vs, and the attachment rate
# gamma(E) = [1.1 + 11.3 exp(-1.04e-5 E / (V/m))] x 1e7 1/s, stated for E >= 1e5 V/m (1000 V/cm)
# and used unchanged below that (at 500 V/cm the same source quotes 9.0e7 1/s; the formula
# gives 7.8e7 1/s).
AIR_ELECTRON_MOBILITY_CM2_VS = 830.0
AIR_ATTACHMENT_RATE_S = (1.1e7, 1.13e8, 1.04e-3)  # gamma = a + b exp(-c E), E in V/cm


@dataclass(frozen=True)
class Gas:
    """Ion (and optionally free-electron) transport and ionisation parameters.

    Positive and negative ions have separate mobilities and diffusion coefficients. Free
    electrons drift with ``electron_mobility_cm2_Vs`` until they attach to a neutral molecule at
    the field-dependent rate gamma(E) = a + b exp(-c E) with (a, b, c) = ``attachment_rate_s``
    (a and b in 1/s, c in cm/V). Without these two parameters the gas has no free-electron data,
    and every electron is taken to attach where it is produced.
    """

    mu_pos_cm2_Vs: float
    mu_neg_cm2_Vs: float
    D_pos_cm2_s: float
    D_neg_cm2_s: float
    alpha_cm3_s: float
    W_eV: float
    density_g_cm3: float
    name: str = "custom"
    electron_mobility_cm2_Vs: float | None = None
    attachment_rate_s: tuple[float, float, float] | None = None

    def __post_init__(self):
        for attr in ("mu_pos_cm2_Vs", "mu_neg_cm2_Vs", "W_eV", "density_g_cm3"):
            if not getattr(self, attr) > 0:
                raise ValueError(f"{attr} must be positive")
        for attr in ("D_pos_cm2_s", "D_neg_cm2_s", "alpha_cm3_s"):
            if not getattr(self, attr) >= 0:
                raise ValueError(f"{attr} must be non-negative")
        if (self.electron_mobility_cm2_Vs is None) != (self.attachment_rate_s is None):
            raise ValueError("give both electron_mobility_cm2_Vs and attachment_rate_s, or neither")
        if self.electron_mobility_cm2_Vs is not None:
            a, b, c = self.attachment_rate_s
            if not self.electron_mobility_cm2_Vs > 0 or a + b <= 0 or min(a, b, c) < 0:
                raise ValueError("free-electron parameters must be positive")

    @classmethod
    def air(
        cls,
        T_K: float = T_REF_K,
        p_hPa: float = P_REF_HPA,
        *,
        particle: str,
        W_eV: float | None = None,
        mu_pos_ref_cm2_Vs: float = 1.36,
        mu_neg_ref_cm2_Vs: float = 2.1,
        alpha_cm3_s: float = 1.6e-6,
    ) -> "Gas":
        """Dry air for the given radiation: ``particle`` is one of ``W_AIR_EV``.

        ``particle`` selects W (ICRU 90). Pass ``W_eV`` explicitly for any other particle; it
        then overrides the table. Ion mobilities are Boag's values at 20 degC and 1013.25 hPa, and
        they scale inversely with the gas number density (constant reduced mobility). The
        diffusion coefficients follow the Einstein relation D = mu k T / e, and the recombination
        coefficient is kept constant. Free electrons use the parameters of Goetz et al. (2013)
        (see ``AIR_ATTACHMENT_RATE_S``); their mobility scales with density, the attachment rate
        does not.
        """
        if W_eV is None:
            if particle not in W_AIR_EV:
                raise ValueError(
                    f"no tabulated W for particle {particle!r}: use one of {sorted(W_AIR_EV)} "
                    "or pass W_eV"
                )
            W_eV = W_AIR_EV[particle]
        density_scale = (p_hPa / P_REF_HPA) * (T_REF_K / T_K)
        mu_pos = mu_pos_ref_cm2_Vs / density_scale
        mu_neg = mu_neg_ref_cm2_Vs / density_scale
        kT_over_e_V = BOLTZMANN_EV_PER_K * T_K
        return cls(
            mu_pos_cm2_Vs=mu_pos,
            mu_neg_cm2_Vs=mu_neg,
            D_pos_cm2_s=mu_pos * kT_over_e_V,
            D_neg_cm2_s=mu_neg * kT_over_e_V,
            alpha_cm3_s=alpha_cm3_s,
            W_eV=W_eV,
            density_g_cm3=DRY_AIR_DENSITY_REF_G_CM3 * density_scale,
            name=f"air, {particle} ({T_K:.2f} K, {p_hPa:.2f} hPa)",
            electron_mobility_cm2_Vs=AIR_ELECTRON_MOBILITY_CM2_VS / density_scale,
            attachment_rate_s=AIR_ATTACHMENT_RATE_S,
        )

    @classmethod
    def air_legacy(cls) -> "Gas":
        """Constants of IonTracks <= v3 (Kanai 1998): one averaged mobility and diffusion."""
        return cls(
            mu_pos_cm2_Vs=1.65,
            mu_neg_cm2_Vs=1.65,
            D_pos_cm2_s=3.7e-2,
            D_neg_cm2_s=3.7e-2,
            alpha_cm3_s=1.6e-6,
            W_eV=33.9,
            density_g_cm3=DRY_AIR_DENSITY_REF_G_CM3,
            name="air (IonTracks <= v3 / Kanai 1998)",
        )

    @property
    def has_free_electrons(self) -> bool:
        return self.electron_mobility_cm2_Vs is not None

    def without_free_electrons(self) -> "Gas":
        """Copy in which every electron attaches where it is produced (Boag's classic model)."""
        return replace(self, electron_mobility_cm2_Vs=None, attachment_rate_s=None)

    def with_(self, **changes) -> "Gas":
        """Return a copy with some parameters replaced, e.g. ``gas.with_(D_pos_cm2_s=0.0)``."""
        return replace(self, **changes)

    def attachment_length_cm(self, field_V_cm: float) -> float:
        """Mean drift of a free electron before it attaches (0 without free-electron data)."""
        if not self.has_free_electrons:
            return 0.0
        a, b, c = self.attachment_rate_s
        return self.electron_mobility_cm2_Vs * field_V_cm / (a + b * math.exp(-c * field_V_cm))

    def free_electron_fraction(self, gap_cm: float, field_V_cm: float) -> float:
        """Fraction of electrons reaching the anode before attaching, in a uniform field."""
        lam = self.attachment_length_cm(field_V_cm)
        if lam == 0.0:
            return 0.0
        return lam / gap_cm * -math.expm1(-gap_cm / lam)

    def ion_pairs_per_cm3_per_Gy(self) -> float:
        """Ion pairs per cm^3 created by an absorbed dose of 1 Gy in the gas."""
        return self.density_g_cm3 * 1e-3 / (self.W_eV * ELEMENTARY_CHARGE_C)

    def ion_pairs_per_cm(self, stopping_power_MeV_cm2_g):
        """Ion pairs per cm of path for a particle with the given mass stopping power in the gas."""
        return stopping_power_MeV_cm2_g * self.density_g_cm3 * 1e6 / self.W_eV
