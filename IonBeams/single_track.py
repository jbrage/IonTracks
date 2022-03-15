import sys; sys.path.append('./cython')

from Geiss_test import Geiss_PDEsolver
from initial_recombination import initial_PDEsolver
from functions import E_MeV_u_to_LET_keV_um, calc_b_cm, IC_angle_rad, Jaffe_theory
import pyamtrack.libAT as libam

import pandas as pd


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


def IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm, a0_nm, use_beta, scale, 
                                    PRINT_parameters=False, SHOW_PLOT=False):
    '''
    Calculate the stopping power and track radius as a function of proton energy

    The function returns the inverse collection efficiency, i.e. the recombination
    correction factor k_s
    '''


    beta = libam.AT_beta_from_E_single(energy_MeV)   
    # print("beta = {:0.3f}".format(beta))
    
    rho_material = 1.2e-3 # g/cm3
    density_ratio = 1.0 / rho_material
    #a0_nm = 20 * beta
    #a0_nm = 1.5    
    a0_cm = a0_nm * 1e-7  * density_ratio    
    
    if use_beta:
        a0_cm *= beta
    
    unit_length_cm = scale*1e-4
    
    # unit_length_cm = a0_cm
    
    # print(unit_length_cm / a0_cm)

    LET_keV_um = E_MeV_u_to_LET_keV_um(energy_MeV)
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

    electrode_gap_cm = 0.2
    
    voltages = [50, 100, 150, 200, 250, 300]
    
    E_MeV_u_list = range(5, 300, 1)
    
    df_J = pd.DataFrame()
    for particle in ["proton", "carbon", "neon", "iron"]:
        LET_keV_um_list = E_MeV_u_to_LET_keV_um(E_MeV_u_list, particle=particle)
        for voltage_V in voltages:
            for E_MeV_u, LET_keV_um in zip(E_MeV_u_list, LET_keV_um_list):    
                
                ks_Jaffe = Jaffe_theory(LET_keV_um, voltage_V, electrode_gap_cm)
	     
                row = {"E_MeV_u": E_MeV_u, "LET_keV_um": LET_keV_um, 
	            "ks_Jaffe": ks_Jaffe, "voltage_V": voltage_V, "particle": particle}

                df_J = df_J.append(row, ignore_index=True)
	
    df_J.to_csv("data_Jaffe.csv", index=False)
    
        
    print("JAFFE FINISHED")	


    
    
