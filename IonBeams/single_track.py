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

    f_Gauss = initial_PDEsolver(simulation_parameters)
    f_Geiss = Geiss_PDEsolver(simulation_parameters)    
    return [1./f_Gauss, 1./f_Geiss]


def calculate_Jaffe_theory(electrode_gap_cm, voltages):

    print("calculating Jaffe's correction factors")

    E_MeV_u_list = range(1, 490, 2)
    
    df_J = pd.DataFrame()
    for particle in ["carbon", "neon", "argon", "iron", "proton"]:
        print(particle)
        LET_keV_um_list = E_MeV_u_to_LET_keV_um(E_MeV_u_list, particle=particle)
        water_LET_keV_um_list = E_MeV_u_to_LET_keV_um(E_MeV_u_list, particle=particle, material="water")
	
        for voltage_V in voltages:
            for E_MeV_u, LET_keV_um_air, LET_keV_um_water in zip(E_MeV_u_list, LET_keV_um_list, water_LET_keV_um_list):    
                
                ks_Jaffe = Jaffe_theory(LET_keV_um_air, voltage_V, electrode_gap_cm)
	     
                row = {"E_MeV_u": E_MeV_u, "LET_keV_um": LET_keV_um_air, "water_LET_keV_um": LET_keV_um_water, 
	            "ks_Jaffe": ks_Jaffe, "voltage_V": voltage_V, "particle": particle}

                df_J = df_J.append(row, ignore_index=True)
	
    df_J.to_csv("data_Jaffe.csv", index=False)        
    print("\nJaffe finished")	



if __name__ == "__main__":

    electrode_gap_cm = 0.2
    
    voltages = [50, 150, 175, 200, 225, 250, 400]
    voltages = range(50, 400, 25)
    

    calculate_Jaffe_theory(electrode_gap_cm, voltages)

    import sys
    sys.exit()
    

    df = pd.DataFrame()

    for scale in [3]:
        for particle in ["carbon", "neon", "argon", "iron", "proton"]:
            for voltage_V in voltages:
                for energy_MeV in range(5, 320, 20):
                    for use_beta in [False]:
                        if use_beta:            
                            a0_nm_list = range(10, 50, 5)
                        else:
                            a0_nm_list = [0.8, 0.9, 1, 1.25, 1.5]

                        for a0_nm in a0_nm_list:
 
                            LET_keV_um = E_MeV_u_to_LET_keV_um(energy_MeV, particle=particle)
                            water_LET_keV_um = E_MeV_u_to_LET_keV_um(energy_MeV, particle=particle, material="water")
                            # calculate the collection effciency with IonTracks and the Jaffe theory

                            ks_Gauss, ks_Geiss = IonTracks_initial_recombination(voltage_V, energy_MeV, electrode_gap_cm, a0_nm, use_beta, scale)
                            # ks_Jaffe = Jaffe_theory(energy_MeV, voltage_V, electrode_gap_cm)

                            row = {"E_MeV_u": energy_MeV, "LET_keV_um": LET_keV_um, "particle": particle,
	                            "a0_nm": a0_nm, "ks_Gauss": ks_Gauss, "ks_Geiss": ks_Geiss, "beta": use_beta,
			              "scale": scale,  "voltage_V": voltage_V, "water_LET_keV_um": water_LET_keV_um,
                               }
                            print(row)
                            df = df.append(row, ignore_index=True)

            df.to_csv("result.csv", index=False)
    
    

