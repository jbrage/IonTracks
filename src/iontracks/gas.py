"""Transport and ionisation parameters of the chamber gas."""

from dataclasses import dataclass, replace

from .constants import (
    BOLTZMANN_EV_PER_K,
    DRY_AIR_DENSITY_REF_G_CM3,
    ELEMENTARY_CHARGE_C,
    P_REF_HPA,
    T_REF_K,
)


@dataclass(frozen=True)
class Gas:
    """Ion transport and ionisation parameters.

    Positive and negative ions have separate mobilities and diffusion coefficients.
    """

    mu_pos_cm2_Vs: float
    mu_neg_cm2_Vs: float
    D_pos_cm2_s: float
    D_neg_cm2_s: float
    alpha_cm3_s: float
    W_eV: float
    density_g_cm3: float
    name: str = "custom"

    def __post_init__(self):
        for attr in ("mu_pos_cm2_Vs", "mu_neg_cm2_Vs", "W_eV", "density_g_cm3"):
            if not getattr(self, attr) > 0:
                raise ValueError(f"{attr} must be positive")
        for attr in ("D_pos_cm2_s", "D_neg_cm2_s", "alpha_cm3_s"):
            if not getattr(self, attr) >= 0:
                raise ValueError(f"{attr} must be non-negative")

    @classmethod
    def air(
        cls,
        T_K: float = T_REF_K,
        p_hPa: float = P_REF_HPA,
        *,
        mu_pos_ref_cm2_Vs: float = 1.36,
        mu_neg_ref_cm2_Vs: float = 2.1,
        alpha_cm3_s: float = 1.6e-6,
        W_eV: float = 33.97,
    ) -> "Gas":
        """Dry air with separate ion mobilities (Boag's values at 20 degC, 1013.25 hPa).

        Mobilities scale inversely with gas number density (constant reduced mobility), and the
        diffusion coefficients follow the Einstein relation D = mu k T / e. The recombination
        coefficient is kept constant. W defaults to 33.97 eV (ICRU 90, electrons); set it for the
        radiation quality at hand.
        """
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
            name=f"air ({T_K:.2f} K, {p_hPa:.2f} hPa)",
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

    def with_(self, **changes) -> "Gas":
        """Return a copy with some parameters replaced, e.g. ``gas.with_(D_pos_cm2_s=0.0)``."""
        return replace(self, **changes)

    def ion_pairs_per_cm3_per_Gy(self) -> float:
        """Ion pairs per cm^3 created by an absorbed dose of 1 Gy in the gas."""
        return self.density_g_cm3 * 1e-3 / (self.W_eV * ELEMENTARY_CHARGE_C)

    def ion_pairs_per_cm(self, stopping_power_MeV_cm2_g):
        """Ion pairs per cm of path for a particle with the given mass stopping power in the gas."""
        return stopping_power_MeV_cm2_g * self.density_g_cm3 * 1e6 / self.W_eV
