"""Physical constants and reference conditions.

Internal unit system: cm, s, V, eV. Charge-carrier densities are ion pairs per cm^3, and
quantities integrated over the electrode gap are per cm^2 of electrode area.
"""

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_EV_PER_K = 8.617333262e-5
MEV_TO_J = 1.602176634e-13
VACUUM_PERMITTIVITY_F_CM = 8.8541878128e-14

T_REF_K = 293.15
P_REF_HPA = 1013.25
DRY_AIR_DENSITY_REF_G_CM3 = 1.20479e-3  # dry air at T_REF_K and P_REF_HPA (NIST)
