import sys; sys.path.append('./cython')

from Geiss_test import Geiss_PDEsolver
from initial_recombination import initial_PDEsolver
from functions import E_MeV_to_LET_keV_um, calc_b_cm, IC_angle_rad, Jaffe_theory
import pyamtrack.libAT as libam




s = """
    stopping_power_source_name = libam.stoppingPowerSource_no["PSTAR"].value
    particle_no = libam.AT_particle_no_from_Z_and_A_single(6, 12)
    material_no = libam.AT_material_number_from_name("Air")
    print(material_no, particle_no)
    
    E_MeV_u = 100.0
    stopping_power_keV_um = [0.0]
    stopping_power_keV_um = [0]
    libam.AT_Stopping_Power(p_stopping_power_source=stopping_power_source_name,
                            p_E_MeV_u=[E_MeV_u],
                            p_particle_no=[particle_no],
                            p_material_no=material_no,
                            p_stopping_power_keV_um=stopping_power_keV_um)

  
"""


def IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm,
                                    PRINT_parameters=False, SHOW_PLOT=False):
    '''
    Calculate the stopping power and track radius as a function of proton energy

    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    '''


    beta = libam.AT_beta_from_E_single(energy_MeV)   
    print("beta = {:0.3f}".format(beta))
    
    rho_material = 1.2e-3 # g/cm3
    density_ratio = 1.0 / rho_material
    a0_nm = 20 * beta
    a0_nm = 1.5    
    a0_cm = a0_nm * 1e-7  * density_ratio    
    
    unit_length_cm = 2.5e-4
    
    # unit_length_cm = a0_cm
    
    # print(unit_length_cm / a0_cm)

    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    track_radius_cm = calc_b_cm(LET_keV_um)

    LET_eV_cm = LET_keV_um*1e7
    simulation_parameters = [
                                LET_eV_cm,
                                track_radius_cm,
                                voltage_V,
                                IC_angle_rad,
                                electrode_gap_cm,
                                SHOW_PLOT,
                                PRINT_parameters, 
                                energy_MeV,
                                a0_cm,
                                unit_length_cm
                            ]

    # f_IonTracks = initial_PDEsolver(simulation_parameters)
    f_IonTracks = Geiss_PDEsolver(simulation_parameters)    
    return 1./f_IonTracks


if __name__ == "__main__":

    electrode_gap_cm = 0.1
    voltage_V = 200
    energy_MeV = 20.0
    
    # LET in air
    LET_keV_um = E_MeV_to_LET_keV_um(energy_MeV)
    
    # calculate the collection effciency with IonTracks and the Jaffe theory
    ks_IonTracks = IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm)
    ks_Jaffe = Jaffe_theory(energy_MeV, voltage_V, electrode_gap_cm)

    results = [energy_MeV, LET_keV_um, ks_Jaffe, ks_IonTracks]

    header = "# E [MeV], LET [keV/um],  ks_Jaffe, ks_IonTracks\n"
    text = "     {},\t{:0.5f},   {:0.6f},\t{:0.6f}".format(*results)
    print(header, text)



    
    
