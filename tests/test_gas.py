import pytest

from iontracks import Gas
from iontracks.constants import BOLTZMANN_EV_PER_K, P_REF_HPA, T_REF_K


def test_air_has_separate_mobilities_and_einstein_diffusion():
    gas = Gas.air()
    kT_over_e = BOLTZMANN_EV_PER_K * T_REF_K
    assert gas.mu_pos_cm2_Vs == pytest.approx(1.36)
    assert gas.mu_neg_cm2_Vs == pytest.approx(2.1)
    assert gas.D_pos_cm2_s == pytest.approx(1.36 * kT_over_e)
    assert gas.D_neg_cm2_s == pytest.approx(2.1 * kT_over_e)
    assert gas.D_pos_cm2_s != gas.D_neg_cm2_s


def test_pressure_scaling():
    ref, dense = Gas.air(), Gas.air(p_hPa=2 * P_REF_HPA)
    assert dense.mu_pos_cm2_Vs == pytest.approx(ref.mu_pos_cm2_Vs / 2)
    assert dense.D_neg_cm2_s == pytest.approx(ref.D_neg_cm2_s / 2)
    assert dense.density_g_cm3 == pytest.approx(2 * ref.density_g_cm3)
    assert dense.ion_pairs_per_cm3_per_Gy() == pytest.approx(2 * ref.ion_pairs_per_cm3_per_Gy())


def test_temperature_scaling():
    ref, warm = Gas.air(), Gas.air(T_K=1.1 * T_REF_K)
    assert warm.mu_pos_cm2_Vs == pytest.approx(1.1 * ref.mu_pos_cm2_Vs)
    assert warm.D_pos_cm2_s == pytest.approx(1.21 * ref.D_pos_cm2_s)  # mu x T
    assert warm.density_g_cm3 == pytest.approx(ref.density_g_cm3 / 1.1)


def test_ion_pairs_per_gray_in_air():
    assert Gas.air().ion_pairs_per_cm3_per_Gy() == pytest.approx(2.21e11, rel=0.01)


def test_ion_pairs_per_cm_from_stopping_power():
    gas = Gas.air()
    # 1 MeV cm^2/g -> rho * 1e6 eV/cm / W
    assert gas.ion_pairs_per_cm(1.0) == pytest.approx(gas.density_g_cm3 * 1e6 / gas.W_eV)


def test_legacy_preset():
    gas = Gas.air_legacy()
    assert gas.mu_pos_cm2_Vs == gas.mu_neg_cm2_Vs == 1.65
    assert gas.D_pos_cm2_s == gas.D_neg_cm2_s == 3.7e-2


def test_validation():
    with pytest.raises(ValueError):
        Gas.air().with_(mu_pos_cm2_Vs=0.0)
    with pytest.raises(ValueError):
        Gas.air().with_(D_neg_cm2_s=-1.0)
