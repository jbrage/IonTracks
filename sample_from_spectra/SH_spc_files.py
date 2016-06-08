import src.pyamtrack
import src.pyamtrack_SPC
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from get_b_LET import get_Gaussian_track_radii
import os.path


def get_spectrum_at_range(chosen_range_g_cm2):
    range_g_cm2 = src.pyamtrack_SPC.extract_range(fpath)
    n_bins = src.pyamtrack.AT_SPC_number_of_bins_at_range(spcdirpath, range_g_cm2)
    all_outputs = src.pyamtrack.AT_SPC_read_from_filename_fast(fpath, n_bins, 0., 0., 0, 0, 0., 0, [0] * n_bins,
                                                               [0] * n_bins, [0] * n_bins, [0] * n_bins, [0] * n_bins,
                                                               [0] * n_bins)
    output = all_outputs[0]  # should be 0
    assert (output == 0)

    E_MeV_u_initial = all_outputs[6]            # (float) initial energy
    peak_position_g_cm2 = all_outputs[7]        # (float)
    particle_number_initial = all_outputs[8]    # (int) see AT_particle_name_from_particle_no_single
    material_number = all_outputs[9]            # (int) see AT_material_name_from_number
    normalisation = all_outputs[10]             # (float) normalisation of the fluence

    if normalisation != 1.0:
        print("normalisation: " + str(normalisation))

    material_name = src.pyamtrack.AT_material_name_from_number(material_number, " ")[1]

    print("\nSpectrum of a %03i MeV/u beam in %s" % (E_MeV_u_initial, material_name))
    print("Peak position:\t\t %0.5g [g/cm^2]" % peak_position_g_cm2)

    print("Spectrum at depth:\t %0.5g [g/cm^2]\n" % chosen_range_g_cm2)

    spectrum = src.pyamtrack_SPC.extract_spectrum_at_range(spcdirpath,range_g_cm2)
    spec_dic = src.pyamtrack_SPC.get_spectrum_dictionary_depth(spectrum)
    spectrum_at_depth = src.pyamtrack_SPC.spectrum_at_depth(chosen_range_g_cm2,spec_dic)

    fluence_dic = {}
    energy_dic = {}

    for particle in set(spectrum_at_depth[1][:]):
        fluence_dic[particle] = []
        energy_dic[particle] = []

    n = len(spectrum_at_depth[0][:])

    for i in range(n):
        particle = spectrum_at_depth[1][i]
        fluence_dic[particle].append(spectrum_at_depth[2][i])
        energy_dic[particle].append(spectrum_at_depth[0][i])

    return energy_dic, fluence_dic, E_MeV_u_initial


def sample_points(incident_number_of_particles,energy_dic,fluence_dic,E_MeV_u_initial):

    particle_dic = {1001: '1H', 2004: '4He', 3007: '7Li', 4009:'9Be', 5011: '11B', 6012: '12C'}
    particle_names = fluence_dic.keys()

    collect_Gaussian_radii = []
    collect_LETs = []
    mean_C12 = E_MeV_u_initial*2 # range of the x-axis

    res_file = open("radii_and_LETs.dat", "w")
    for particle_name in particle_names:
        print("\n# ==============================================================")

        particle_energy = energy_dic[particle_name]
        particle_fluence = fluence_dic[particle_name]
        relative_number_of_particles = int(sum(particle_fluence) * incident_number_of_particles)
        print("# "+str(particle_dic[particle_name])+", number of particles: " + str(relative_number_of_particles))

        if relative_number_of_particles > 0:
            # sample points from a discrete distribution
            normalised_fluence = np.array(particle_fluence)/sum(particle_fluence)
            sampled_energy = np.random.choice(particle_energy, size=relative_number_of_particles, replace=True,
                                              p=normalised_fluence)

            # for plotting purposes
            if str(particle_dic[particle_name]) == '12C':
                mean_C12 = np.mean(sampled_energy)
                if mean_C12 < 30:
                    mean_C12 = 50

            occurences = Counter(sampled_energy)

            print("# E [MeV/u] \tLET [keV/um] \tradius, b [um]\t# of occurences")
            for E in set(sampled_energy):
                E, LET, b = get_Gaussian_track_radii(E, print_bs_LETs)
                print("  %g\t\t%.4f\t\t%.4f\t\t%i" % (E, LET, b * 1e4, occurences[E]))

                collect_Gaussian_radii.append(b)
                collect_LETs.append(LET)

                text = "%g,%g,%g\n" % (b,LET,occurences[E])
                res_file.write(text)

            fluence = []
            for i in sampled_energy:
                ind = particle_energy.index(i)
                fluence.append(particle_fluence[ind])
            plt.plot(sampled_energy, fluence, '-*')
    plt.xlim([0, mean_C12 * 2])

    plt.savefig("spectra_w_sampled_points.png")
    res_file.close()

    return collect_Gaussian_radii, collect_LETs


if __name__ == "__main__":

    #path to the directory and the spc file respectively
    spcdirpath = "/home/jeppebrage/PycharmProjects/spectra-scripts"
    fpath = spcdirpath + "/aps.spc"

    assert os.path.isfile(fpath)

    SHOW_PLOT=True # show plots
    print_bs_LETs=False # print the calculated results from the subroutine

    chosen_distance_g_cm2 = 1.1 #spectrum at a given distance [g/cm^2]
    incident_number_of_particles = 50  # calculate this from the incident fluence

    energy_dic,fluence_dic,E_MeV_u_initial = get_spectrum_at_range(chosen_distance_g_cm2)
    Gaussian_radii, LETs = sample_points(incident_number_of_particles, energy_dic,fluence_dic,E_MeV_u_initial)