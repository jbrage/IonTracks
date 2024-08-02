from hadrons.utils import calculate_track_radius, get_LET_per_um


def get_PDEsolver_input(E_MeV_u, voltage_V, electrode_gap_cm, particle, grid_size_um):
    LET_keV_um = get_LET_per_um(E_MeV_u, particle)
    track_radius_cm = calculate_track_radius(LET_keV_um)

    return dict(
        LET_keV_um=float(LET_keV_um),
        voltage_V=voltage_V,
        IC_angle_rad=0,
        electrode_gap_cm=electrode_gap_cm,
        E_MeV_u=E_MeV_u,
        a0_nm=8.0,
        RDD_model="Gauss",
        unit_length_cm=grid_size_um * 1e-4,
        track_radius_cm=track_radius_cm,
    )
