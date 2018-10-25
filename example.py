from scanned_beam import IonTracks_total_recombination
from single_track import IonTracks_initial_recombination
from functions import Jaffe_theory


'''
Calculate the ion recombination correction factors ks as a function of energy,
electrode gap, polarization voltage and dose-rate for a parallel-plate
ionziation chamber
'''

voltage_V = 200
electrode_gap_cm = 0.2
energy_MeV = 226
doserate_Gy_min = 10


print("\n# Parallal-place IC with electrode gap {} cm".format(electrode_gap_cm))
print("# Using {} MeV protons at {} V\n".format(energy_MeV, voltage_V))

# INITIAL RECOMBINATION

ks_IonTracks = IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm)
ks_Jaffe = Jaffe_theory(energy_MeV, voltage_V, electrode_gap_cm)

print("# Initial recombination:")
print("ks = {:0.6f} (Jaffe theory)".format(ks_Jaffe))
print("ks = {:0.6f} (IonTracks)".format(ks_IonTracks))

# INITIAL AND GENERAL RECOMBINATION

input_parameters = [voltage_V, energy_MeV, doserate_Gy_min, electrode_gap_cm]

ks_IonTracks = IonTracks_total_recombination(input_parameters)
print("\n# Initial + general recombination:")
print("# with a {} Gy/min dose-rate\nks = {:0.6f}".format(doserate_Gy_min, ks_IonTracks))
