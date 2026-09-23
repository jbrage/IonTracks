import math

import pytest

from iontracks import Gas
from iontracks.constants import BOLTZMANN_EV_PER_K, P_REF_HPA, T_REF_K


def test_air_has_separate_mobilities_and_einstein_diffusion():
    gas = Gas.air(particle="electron")
    kT_over_e = BOLTZMANN_EV_PER_K * T_REF_K
    assert gas.mu_pos_cm2_Vs == pytest.approx(1.36)
    assert gas.mu_neg_cm2_Vs == pytest.approx(2.1)
    assert gas.D_pos_cm2_s == pytest.approx(1.36 * kT_over_e)
    assert gas.D_neg_cm2_s == pytest.approx(2.1 * kT_over_e)
    assert gas.D_pos_cm2_s != gas.D_neg_cm2_s


def test_pressure_scaling():
    ref, dense = Gas.air(particle="electron"), Gas.air(p_hPa=2 * P_REF_HPA, particle="electron")
    assert dense.mu_pos_cm2_Vs == pytest.approx(ref.mu_pos_cm2_Vs / 2)
    assert dense.D_neg_cm2_s == pytest.approx(ref.D_neg_cm2_s / 2)
    assert dense.density_g_cm3 == pytest.approx(2 * ref.density_g_cm3)
    assert dense.ion_pairs_per_cm3_per_Gy() == pytest.approx(2 * ref.ion_pairs_per_cm3_per_Gy())


def test_temperature_scaling():
    ref, warm = Gas.air(particle="electron"), Gas.air(T_K=1.1 * T_REF_K, particle="electron")
    assert warm.mu_pos_cm2_Vs == pytest.approx(1.1 * ref.mu_pos_cm2_Vs)
    assert warm.D_pos_cm2_s == pytest.approx(1.21 * ref.D_pos_cm2_s)  # mu x T
    assert warm.density_g_cm3 == pytest.approx(ref.density_g_cm3 / 1.1)


def test_ion_pairs_per_gray_in_air():
    assert Gas.air(particle="electron").ion_pairs_per_cm3_per_Gy() == pytest.approx(
        2.21e11, rel=0.01
    )


def test_ion_pairs_per_cm_from_stopping_power():
    gas = Gas.air(particle="electron")
    # 1 MeV cm^2/g -> rho * 1e6 eV/cm / W
    assert gas.ion_pairs_per_cm(1.0) == pytest.approx(gas.density_g_cm3 * 1e6 / gas.W_eV)


def test_legacy_preset():
    gas = Gas.air_legacy()
    assert gas.mu_pos_cm2_Vs == gas.mu_neg_cm2_Vs == 1.65
    assert gas.D_pos_cm2_s == gas.D_neg_cm2_s == 3.7e-2


def test_validation():
    with pytest.raises(ValueError):
        Gas.air(particle="electron").with_(mu_pos_cm2_Vs=0.0)
    with pytest.raises(ValueError):
        Gas.air(particle="electron").with_(D_neg_cm2_s=-1.0)


def test_w_follows_the_particle():
    assert Gas.air(particle="electron").W_eV == Gas.air(particle="photon").W_eV == 33.97
    assert Gas.air(particle="proton").W_eV == 34.44
    assert Gas.air(particle="carbon").W_eV == 34.71
    assert Gas.air(particle="helium", W_eV=34.5).W_eV == 34.5
    with pytest.raises(TypeError):
        Gas.air()  # the particle must be stated
    with pytest.raises(ValueError, match="no tabulated W"):
        Gas.air(particle="helium")


def test_free_electron_data():
    gas = Gas.air(particle="electron")
    assert gas.has_free_electrons and not Gas.air_legacy().has_free_electrons
    assert not gas.without_free_electrons().has_free_electrons
    # Goetz et al. at 1000 V/cm: mobility 830 cm^2/Vs, gamma = (1.1 + 11.3 e^-1.04) 1e7 1/s
    lam = gas.attachment_length_cm(1000.0)
    assert lam == pytest.approx(830 * 1000 / ((1.1 + 11.3 * math.exp(-1.04)) * 1e7), rel=1e-12)
    p = gas.free_electron_fraction(0.2, 1000.0)
    assert p == pytest.approx(lam / 0.2 * (1 - math.exp(-0.2 / lam)), rel=1e-12)
    assert 0.05 < p < 0.12
    with pytest.raises(ValueError):
        gas.with_(attachment_rate_s=None)  # both parameters or neither
