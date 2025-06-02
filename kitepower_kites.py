from qsm import SystemProperties
from math import pi
import json5

def init_sys_props(hardware_database_filename):
    """
    Build system properties for the kite system based on the hardware database and experimental data.
    Inputs:
        hardware_database_filename (str): Path to the hardware database JSON file.
        experimental_aero_coeff_filename (str): Path to the experimental aero coefficients pickle file.
    
    Returns:
        dict: A dictionary containing system properties.
    """
  
    # LOAD FROM hardware_database.json
    with open(hardware_database_filename) as f:
        hardware_data = json5.load(f)

    # Access kite values
    kite_data = hardware_data["Kite"]["KiteV9"]["values"]
    kcu_data = hardware_data["KCU"]["KCU2"]["values"]
    gs_data = hardware_data["GS"]["GS3"]["values"]
    tether_id = gs_data["tether"]['value']
    tether_data = hardware_data["Tether"][tether_id[0:-2]]["values"]

    sys_props = {
        'kite_projected_area': kite_data['kite_surface']['value'],  # [m^2]
        'kite_mass': kite_data['kite_mass']['value'] + kcu_data['kcu_mass']['value'],  # [kg]
        'tether_density': (tether_data["tether_density"]["value"]/
                    (pi/4*tether_data["tether_diameter"]["value"]**2)),  # [kg/m^3] - 0.85 GPa
        'tether_diameter': tether_data["tether_diameter"]["value"],  # [m]
        'tether_force_max_limit': 50000,  # ~ max_wing_loading*projected_area [N] 
        'tether_force_min_limit': 1000,  # ~ min_wing_loading * projected_area [N]
        'kite_lift_coefficient_powered': None,  # [-] - in the range of .9 - 1.0
        'kite_drag_coefficient_powered': None,  # [-]
        'kite_lift_coefficient_depowered': None,  # [-]
        'kite_drag_coefficient_depowered': None,  # [-] - in the range of .1 - .2
        'reeling_speed_min_limit': 0.5,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
        'reeling_speed_max_limit': 10.5,  # [m/s] 
        'tether_drag_coefficient': tether_data["cd_tether"]["value"],  # [-]
    }
    return SystemProperties(sys_props)



# V2 - As presented in Van der Vlugt et al.
projected_area = 19.8  # [m^2]
max_wing_loading = 450  # [N/m^2]
min_wing_loading = 10  # [N/m^2] - min limit seems only active below cut-in
c_l_powered = .59
l_to_d_powered = 3.6  # [-]
c_l_depowered = .15
l_to_d_depowered = 3.5  # [-]
sys_props_v2 = {
    'kite_projected_area': projected_area,  # [m^2] - 25 m^2 total flat area
    'kite_mass': 19.6,  # [kg]
    'tether_density': 724.,  # [kg/m^3]
    'tether_diameter': 0.004,  # [m]
    'tether_force_max_limit': 10000,  # ~ max_wing_loading*projected_area [N]
    'tether_force_min_limit': 200,  # ~ min_wing_loading * projected_area [N]
    'kite_lift_coefficient_powered': c_l_powered,  # [-]
    'kite_drag_coefficient_powered': c_l_powered/l_to_d_powered,  # [-]
    'kite_lift_coefficient_depowered': c_l_depowered,  # [-]
    'kite_drag_coefficient_depowered': c_l_depowered/l_to_d_depowered,  # [-]
    'reeling_speed_min_limit': 0,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
}
sys_props_v2 = SystemProperties(sys_props_v2)

# Kitepower's V3.
projected_area = 19.75  # [m^2]
max_wing_loading = 500  # [N/m^2]
min_wing_loading = 10  # [N/m^2] - min limit seems only active below cut-in
c_l_powered = .9
l_to_d_powered = 4.2  # [-]
c_l_depowered = .2
l_to_d_depowered = 3.2  # [-]
sys_props_v3 = {
    'kite_projected_area': projected_area,  # [m^2] - 25 m^2 total flat area
    'kite_mass': 22.8,  # [kg] - 12 kg kite + 8 kg KCU
    'tether_density': 724.,  # [kg/m^3] - 0.85 GPa
    'tether_diameter': 0.004,  # [m]
    'tether_force_max_limit': 5000,  # ~ max_wing_loading*projected_area [N]
    'tether_force_min_limit': 300,  # ~ min_wing_loading * projected_area [N]
    'kite_lift_coefficient_powered': .9,  # [-] - in the range of .9 - 1.0
    'kite_drag_coefficient_powered': .2,  # [-]
    'kite_lift_coefficient_depowered': .2,  # [-]
    'kite_drag_coefficient_depowered': .1,  # [-] - in the range of .1 - .2
    'reeling_speed_min_limit': 0.1,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
    'reeling_speed_max_limit': 10,  # [m/s]
    'tether_drag_coefficient': 1.1,  # [-]
    'rel_elevation_min_limit': 4*pi/180,
    'rel_elevation_max_limit': 10*pi/180,
    'max_azimuth_min_limit': 10*pi/180,
    'max_azimuth_max_limit': 20*pi/180
}
sys_props_v3 = SystemProperties(sys_props_v3)



# --- Load kite system properties --- 
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/'
sys_props_v9 = init_sys_props(data_path + 'hardware_database.json') # Instance of SystemProperties class
# Update aerodynamic coefficients
sys_props_v9.kite_lift_coefficient_powered = 0.78  # [-]
sys_props_v9.kite_drag_coefficient_powered = 0.14  # [-]
sys_props_v9.kite_lift_coefficient_depowered = 0.35  # [-]
sys_props_v9.kite_drag_coefficient_depowered =  0.08 # [-] 


sys_props_v9.rel_elevation_min_limit = 4*pi/180
sys_props_v9.rel_elevation_max_limit = 8*pi/180
sys_props_v9.max_azimuth_min_limit = 10*pi/180
sys_props_v9.max_azimuth_max_limit = 30*pi/180