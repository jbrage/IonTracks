def Geiss_r_max(E_MeV_u, rho_material):
    # https://www.sciencedirect.com/science/article/pii/S1350448710001903
    rho_water = 1.0  # g /cm3
    r_max_cm = 4e-5 * E_MeV_u ** 1.5 * rho_water / rho_material
    return r_max_cm


def Geiss_RRD_cm(r_cm, c, a0_cm, r_max_cm):
    # inner core
    if r_cm < a0_cm:
        return c / (a0_cm * a0_cm)
    # radial fall-off
    elif ((a0_cm <= r_cm) and (r_cm <= r_max_cm)):
        return c / (r_cm * r_cm)
    else:
        return 0
