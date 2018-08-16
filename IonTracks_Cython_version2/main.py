import sys, os
# sys.path.append("/home/jeppe/usr/SkandionRecombination")
# from use_pgf_plot import newfig, savefig
import numpy as np
from math import exp, pi, sqrt, log, erf
from scipy.integrate import quad, dblquad


def energy_to_FWHM(energy_MeV):
    # implement function
    FWHM_cm = 0.59 # 100 MeV, in Nice
    return FWHM_cm


def energy_MeV_to_LET(energy_MeV):
    if int(energy_MeV) == 70:
        LET_keV_um = 1.017e-3
    elif int(energy_MeV) == 150:
        LET_keV_um = 5.801-4
    elif int(energy_MeV) == 226:
        LET_keV_um = 4.434e-4
    else:
        raise ValueError("Wrong Energy: %s MeV" % energy_MeV)

    # FOR 100 MEV
    LET_keV_um = 7.760e-4

    return LET_keV_um


def MU_to_charge(MU, energy_MeV, w_dir):
    fname = "%s/input_data/MU_files/%iMeV_%0.1gMU.txt" % (w_dir, energy_MeV, MU)
    with open(fname, 'r') as MUfile:
        for line in MUfile:
            if "FullSpot" in line:
                full_MU = float(line.split(" ")[1])
            if "MaxSpot" in line:
                max_MU = float(line.split(" ")[1])
    K_TP = 1.0
    GainCorrection = 1.0 # Center spot
    chargePerMU = 3e-9 # Coloumb
    charge_per_s_per_MU = MU * chargePerMU * K_TP * GainCorrection
    charge_per_s = charge_per_s_per_MU * max_MU
    return charge_per_s



def my_pdf(x, x_center, FWHM, IC_radius_cm):
    sigma = FWHM / 2.35482
    # int_0^sigma funktion dr = 0.68, ^2*sigma = 0.95 etc IF C = 1
    r_max = IC_radius_cm # [cm] maximum radius of ionization chamber
    C = erf(r_max / (sqrt(2)*sigma)) # normalization as the integration is not to infty
    norm_factor = 2.0 /(np.sqrt(2*np.pi)*sigma* C)

    if x_center  > IC_radius_cm:
        x_center = 0.99*IC_radius_cm
    if x > r_max:
        return 0.0
    return norm_factor* np.exp( - ( - x - x_center)**2 /(2*sigma**2))


def calc_b_cm(LET_keV_um, w_dir):
    data = np.genfromtxt(w_dir + "/input_data/LET_b.dat", delimiter=",", dtype=float)
    scale = 1e-3
    LET = data[:,0]*scale
    b = data[:,1]
    logLET = np.log10(LET)
    z = np.polyfit(logLET, b, 2)
    p = np.poly1d(z)

    b_cm = p(np.log10(LET_keV_um)) *1e-3
    threshold = 2e-3
    if b_cm < threshold:
        b_cm = threshold
    # b_cm = 1.05*1e-3 # cm   ... Kanai, avoids ridges
    return b_cm


def initialise(parameters):
    w_dir, center_cm, r_cm, voltage_V, simulation_time_s, energy_MeV, IC_radius_cm, MU = parameters

    sys.path.append(w_dir + '/cython')
    from distribute_and_evolve_pulsed import PDEsolver

    myseed = int(np.random.randint(1, 1e7))

    FWHM = energy_to_FWHM(energy_MeV)
    # MU = 1.0
    # charge_per_second = MU_to_charge(MU, energy_MeV, w_dir)
    charge_per_second = 2.35*1e-9
    n_tracks_per_second = charge_per_second/1.602e-19

    # compute the total number of tracks in the given simulation time
    total_number_of_tracks = simulation_time_s * n_tracks_per_second

    LET_keV_um = energy_MeV_to_LET(energy_MeV)
    LET_eV_cm = LET_keV_um*1e7

    track_radius_cm = calc_b_cm(LET_keV_um, w_dir)

    electrode_gap_cm = 0.06             # electrode gap [cm]
    IC_angle_rad = 0.              # angle between track and electric field [rad]

    electric_field_V_cm = voltage_V/electrode_gap_cm

    unit_length_cm = 5e-4      # cm, size of every voxel length
    SHOW_PLOT = True
    SHOW_PLOT = False

    # compute the fraction of the beam covered by the given spot
    area_fraction = quad(my_pdf, 0.0, r_cm, args=(center_cm, FWHM, IC_radius_cm, ))[0]
    # scale the total number of tracks with the probabilty of being in the spot
    # to approximate the number of tracks in the said spot
    number_of_tracks = int(total_number_of_tracks*area_fraction)

    if number_of_tracks < 1:
        raise ValueError("\n!! No tracks: Check integration domain\n")

    print("Center = %0.4g cm" % center_cm)
    print(r"Density fraction = {:05.3f} %".format(area_fraction*100))
    print("Number of tracks = %s" % number_of_tracks)
    print("Voltage = %s V" % voltage_V)

    simulation_parameters = [
                                number_of_tracks,
                                LET_eV_cm,
                                track_radius_cm,
                                simulation_time_s,
                                electric_field_V_cm,
                                unit_length_cm,
                                IC_angle_rad,
                                r_cm,
                                electrode_gap_cm,
                                SHOW_PLOT,
                                myseed
                            ]

    no_initialised, no_recombined = PDEsolver(simulation_parameters)
    f_IonTracks = (no_initialised - no_recombined) / no_initialised

    print("\nCollection efficiency = %0.6g\n" % f_IonTracks)


    result_dir = "%s/results/E_%i_MeV/%s_MU/%i_volt" % (w_dir, energy_MeV, MU, voltage_V)
    os.system("mkdir -p %s" % result_dir)
    fname = "/center_{:0.6f}_cm.dat".format(center_cm)

    header = "# f, energy [MeV], r_max [cm], simul. time [s], voltage [V], center [cm], MU\n"
    output = [f_IonTracks, energy_MeV, r_cm, simulation_time_s, voltage_V, center_cm, MU]

    with open(result_dir + fname, 'w') as outfile:
        outfile.write(header)

        text = "{:0.6f},{:3d},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f}\n".format(*output)
        outfile.write(text)

    return output



if __name__ == "__main__":

    w_dir = os.getcwd()
    center_cm = 0.01
    r_cm = 0.001
    voltage_V = 200
    simulation_time_s = 200e-6
    energy_MeV = 226
    IC_radius_cm = 0.5
    MU = 1.0

    input_parameters = [w_dir, center_cm, r_cm, voltage_V, simulation_time_s, energy_MeV, IC_radius_cm, MU]
    output = initialise(input_parameters)
