from math import log, exp
from scipy.integrate import quad
from scipy.optimize import fsolve
from src.pyamtrack import AT_Stopping_Power

MIN_DISTANCE = 5 * 1e-4  # deviates from Kanai!
MAX_DISTANCE = 50 * 1e-4


# Scholz parametrization of maximal radius
def rmax_E(E):
    rho_air = 0.001225
    rmax = 4e-5 * E ** 1.5 / rho_air
    return rmax


# Geiss parametrization of the radial dose distribution
def SK(r, E):
    rMin = 1e-6  # cm
    rMax = rmax_E(E)
    factor = 1. / (0.5 + log(rMax / rMin))

    if r <= rMin:
        return factor * 1. / rMin ** 2
    if rMin < r and r <= rMax:
        return factor * 1. / r ** 2
    else:
        return 0.


# including the Jacobian, i.e. \int_0^infty SK(r,E) r dr = 1.0
def SK_jacobian(r, E):
    return SK(r, E) * r


# integrate from 0 to r
def integrate_SK_0_to_r(r, E):
    return quad(SK_jacobian, 0, r, args=(E))[0]


# solve the equation int_0^{distance} SK(E) r dr = m
def solve_SK_percent_distance(percent, E):
    find_min = lambda r: percent - integrate_SK_0_to_r(r, E)
    b_guess = 0.001  # cm
    distance = fsolve(find_min, b_guess)[0]
    if distance < MIN_DISTANCE or E < 20:
        return MIN_DISTANCE
    if distance > MAX_DISTANCE:
        return MAX_DISTANCE
    else:
        return distance


# analytical expression, i.e. solve_SK_percent_distance analytically
def calc_b(m, E):
    rMin = 1e-6  # cm
    rMax = rmax_E(E)

    b = rMin * exp(m / 2. - 0.5 + m * log(rMax / rMin))
    if b < MIN_DISTANCE:
        return MIN_DISTANCE
    if b > MAX_DISTANCE:
        return MAX_DISTANCE
    else:
        return b


# calculate the LET and Gaussian track radius b from the energy
def get_Gaussian_track_radii(E, write):
    ## reference values, (E,b) pair from Kanai.
    b_Kanai = 10.5 * 1e-4  # cm
    E_Kanai = 90  # MeV/u
    volume_percent = integrate_SK_0_to_r(b_Kanai, E_Kanai)

    b = calc_b(volume_percent, E)
    stopping_power_source = "Bethe"
    E_MeV_u = [E]
    particle_no_C12 = [6012]  # 12^C
    material_no_air = 7  # air
    emp = []
    LET = AT_Stopping_Power(stopping_power_source, 1, E_MeV_u, particle_no_C12, material_no_air, emp)[1][0]

    if write:
        print("#E [MeV/u] \tLET [keV/um] \tb [cm]")
        print("%g\t\t%0.4g\t\t%0.4g" % (E, LET, b))
    return E, LET, b
