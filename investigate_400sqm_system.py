import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from power_curve_single_profile import load_power_curve_single_profile_results_and_plot_trajectories
from utils import *
from qsm import Cycle, LogProfile, SystemProperties, TractionPhasePattern
from cycle_optimizer import OptimizerCycle

def qsm_settings_from_opt_sol(opt_sol):
    settings = {
        'cycle': {
            'traction_phase':                   TractionPhasePattern,
            'include_transition_energy':        True,
            'tether_length_start_retraction':   opt_sol[6],      
            'tether_length_end_retraction':     opt_sol[5],  
            'elevation_angle_traction':         opt_sol[2],
        },
        'retraction': {
            'control': ('tether_force_ground', opt_sol[1]),
        },
        'transition': {
            'control': ('reeling_speed', 0.),
        },
        'traction': {
            'control': ('tether_force_ground', opt_sol[0]),
            'pattern': {'rel_elevation_angle': opt_sol[3], 'azimuth_angle': opt_sol[4]},
            'time_step': .05,
        },
    }
    return settings

# Create environment object.
env_state = LogProfile()
# Set reference height and wind speed
env_state.set_reference_height(100)
env_state.set_reference_wind_speed(15)

# Load system properties from config.yaml
with open("config/config_400.yaml") as f:
    config = yaml.safe_load(f)

sys_props = parse_system_properties_and_bounds(config)
sys_props = SystemProperties(sys_props)

sim_sets = parse_sim_settings(config)

settings = {
        'cycle': {
            'traction_phase':                   TractionPhasePattern,
            'include_transition_energy':        True,
            'tether_length_start_retraction':   465.,      
            'tether_length_end_retraction':     280.,  
            'elevation_angle_traction':         20*np.pi/180
        },
        'retraction': {
            'control': ('tether_force_ground', 30000),
        },
        'transition': {
            'control': ('reeling_speed', 0.),
        },
        'traction': {
            'control': ('tether_force_ground', 215000),
            #'control': ('reeling_speed', 2.6),
            'pattern': {'rel_elevation_angle': 10*np.pi/180, 'azimuth_angle': 37*np.pi/180},
            'time_step': .2,
        },
    }


enabled_opt_var, init_values_array = parse_opt_variables(config)
enabled_cons, param_values_array = parse_constraints(config)

opt_set = parse_opt_settings(config)

oc = OptimizerCycle(settings, sys_props, env_state, enabled_opt_var, enabled_cons,  
                    param_values_array, 'force')

oc.x0_real_scale = np.array([1.97483363e+05*1.5, 2.56563826e+04*1.5, 5.23598776e-01, 1.67633781e-01,\
 6.28844674e-01, 2.80000000e+02, 4.70000000e+02])
oc.scaling_x = np.array([1e-6, 1e-5, 1, 1, 1, 1e-3, 1e-3])
          
oc.optimize(maxiter = opt_set['maxiter'],
             iprint  = opt_set['iprint'],
             eps     = opt_set['eps'],
             ftol    = opt_set['ftol'])

print(oc.x_opt_real_scale)
oc.plot_opt_evolution()
oc.eval_point(True, False, oc.x_opt_real_scale)
plt.show()


opt_sol = [1.97483363e+05, 2.56563826e+04, 5.23598776e-01, 1.67633781e-01,\
 6.28844674e-01, 2.80000000e+02, 4.70000000e+02]


#oc.eval_point(True, False, opt_sol)

tent_sol = np.array([1.97483363e+05*1.5, 2.56563826e+04*1.5, 5.23598776e-01, 1.67633781e-01,\
 6.28844674e-01, 2.80000000e+02, 4.70000000e+02])
settings = qsm_settings_from_opt_sol(tent_sol)

# Create pumping cycle simulation object, run simulation, and plot results.
cycle = Cycle(settings)
cycle.run_simulation(sys_props, env_state, print_summary=True)

# Plotting different quantities of interests
cycle.time_plot(('reeling_speed', 'power_ground', 'apparent_wind_speed', 'tether_force_ground'),
                    ('Reeling speed [m/s]', 'Power [W]', 'Apparent wind speed [m/s]', 'Tether force [N]'))

# Plotting other quantitities, and scaling the angles to have them in degrees
cycle.time_plot(('straight_tether_length', 'elevation_angle', 'azimuth_angle', 'course_angle'),
                    ('Tether length [m]', 'Elevation [deg]', 'Azimuth [deg]', 'Course [deg]'),
                    (None, 180./np.pi, 180./np.pi, 180./np.pi))

# Plotting the trajectory
cycle.trajectory_plot3d()
plt.show()

time = [t for t in cycle.traction_phase.time]
course_angle = [k.course_angle for k in cycle.traction_phase.kinematics]
course_rate = np.gradient(course_angle, settings['traction']['time_step']) 

#plt.plot(time, course_angle)
#plt.plot(time, course_rate)
#plt.plot(time, np.ones_like(time)*course_max)
plt.show()




