import sys, os
from typing import Tuple
sys.path.append(os.path.abspath('./'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from awe_pe.utils import *
from awe_pe.cycle_optimizer import OptimizerCycle
from awe_pe.qsm import SystemProperties, LogProfile, TractionPhasePattern
from awe_pe.power_curve_constructor import PowerCurveConstructor


def get_wind_speed_and_solution_from_power_curve(idx:int = 0):
    labels = ['v_RO [m/s]',
                'F_RI [N]',
                'theta_avg_RO [rad]',
                'theta_rel_RO [rad]',
                'phi_max_RO [rad]',
                'stroke_tether [m]',
                'min_length_tether [m]']
    x_opt = [power_curve.iloc[idx][lab] for lab in labels]
    v_100m = power_curve.iloc[idx]['v_100m [m/s]']
    print(f'Wind speed at 100m = {v_100m}')
    return v_100m, x_opt

def get_cycle_optimizer(env_state:LogProfile, config_filename:str="config/config_V3_speed.yaml"):

    with open(config_filename) as f:
        config = yaml.safe_load(f)

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    cycle_sim_settings = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': True,
        },
        'retraction': {'time_step': 0.25
                        },
        'transition': {'time_step': 0.25,
        },
        'traction': {'time_step': 0.25
        },
    }

    opt_var_enabled_idx, _ = parse_opt_variables(config)
    _, cons_param_vals = parse_constraints(config)

    cycle_optimizer = OptimizerCycle(cycle_sim_settings, sys_props, env_state, reduce_x = opt_var_enabled_idx,
                            reduce_ineq_cons=range(11), parametric_cons_values=cons_param_vals,
                            force_or_speed_control='hybrid')
    
    return cycle_optimizer

def run_simulation(x_vec, v_100m:float, config_filename:str, plot:bool=False) -> Tuple:
    env_state = LogProfile()
    env_state.set_reference_height(100)
    env_state.set_roughness_length(0.1)

    env_state.set_reference_wind_speed(v_100m)

    cycle_optimizer = get_cycle_optimizer(env_state, config_filename)

    cons, kpis = cycle_optimizer.eval_point(plot, x_real_scale=x_vec)

    return cons, kpis, cycle_optimizer

def perform_simulation_from_power_curve(idx:int=0, config_filename:str="config/config_V3_speed.yaml", plot:bool=True):

    v_100m, x_opt = get_wind_speed_and_solution_from_power_curve(idx)
    _ = run_simulation(x_opt, v_100m, config_filename, plot)

def export_to_csv_single_profile(v, v_cut_out, p, x_opts, n_cwp, opt_details, control_mode):
    if control_mode == 'force':
        df = {
            'v_100m [m/s]': v,
            'v/v_cut-out [-]': v/v_cut_out,
            'P_cycle [W]': p,
            'F_RO [N]': [x[0] for x in x_opts],
            'F_RI [N]': [x[1] for x in x_opts],
            'theta_avg_RO [rad]': [x[2] for x in x_opts],        
            'theta_rel_RO [rad]': [x[3] for x in x_opts],
            'phi_max_RO [rad]': [x[4] for x in x_opts],
            'stroke_tether [m]': [x[5] for x in x_opts],
            'min_length_tether [m]': [x[6] for x in x_opts],
            'n_crosswind_patterns [-]': n_cwp,
            'success [-]': [od['success'] for od in opt_details],
        }
    elif control_mode == 'hybrid':
        df = {
            'v_100m [m/s]': v,
            'v/v_cut-out [-]': v/v_cut_out,
            'P_cycle [W]': p,
            'v_RO [m/s]': [x[0] for x in x_opts],
            'F_RI [N]': [x[1] for x in x_opts],
            'theta_avg_RO [rad]': [x[2] for x in x_opts],        
            'theta_rel_RO [rad]': [x[3] for x in x_opts],
            'phi_max_RO [rad]': [x[4] for x in x_opts],
            'stroke_tether [m]': [x[5] for x in x_opts],
            'min_length_tether [m]': [x[6] for x in x_opts],
            'n_crosswind_patterns [-]': n_cwp,
            'success [-]': [od['success'] for od in opt_details],
        }   
    df = pd.DataFrame(df)
    df.to_csv('output/power_curve_log_profile.csv', index=False, sep=";")

# power_curve = PowerCurveConstructor(None)
# power_curve.import_results('output/power_curve_log_profile_hybrid.pickle')

# # Values to remove
# values_to_remove = [11.8, 12.7, 13.2, 13.2, 13.3, 13.4, 13.5, 13.6, 13.9, 14.1, 14.1, 14.2]

# # Loop through each value to remove
# for value in values_to_remove:
#     # Find index and remove the value from power_curve.wind_speeds
#     index_to_remove = np.where(np.isclose(power_curve.wind_speeds, value))[0]
#     if index_to_remove.size > 0:  # Ensure the value exists
#         power_curve.wind_speeds = np.delete(power_curve.wind_speeds, index_to_remove[0])

# real_wind_speed = []
# assigned = []
# ii = 0
# jj = 0

# while ii < len(power_curve.wind_speeds) and len(real_wind_speed) <= len(power_curve.x_opts):
#     wind_speed = power_curve.wind_speeds[ii]
#     x_to_check = power_curve.x_opts[jj]
    
#         # Simula il sistema
#     try:
#         cons, kpis, cycle_opt = run_simulation(x_to_check, wind_speed, "config/config_V3_speed.yaml", False)
        
#         # Stampa per monitorare i valori
#         print(f"Iterazione ii={ii}, jj={jj}, wind_speed={wind_speed}")
#         print(f"Valore KPI simulato: {kpis['average_power']['cycle']}")
#         print(f"Valore KPI target: {power_curve.performance_indicators[jj]['average_power']['cycle']}")
        
        
#         # Confronta i KPI con una tolleranza
#         if np.abs(kpis['average_power']['cycle'] - power_curve.performance_indicators[jj]['average_power']['cycle']) <= 1e-2:
#             real_wind_speed.append(wind_speed)
#             print(f"Correlazione trovata per la velocità del vento: {wind_speed}")
#             print("\n")
#             assigned.append(jj)
#             ii += 1  # Incrementa ii, proviamo con la velocità del vento successiva
#             #jj += 1  # Incrementa jj, passiamo al prossimo risultato ottimo
#         else:
#             print(f"Nessuna corrispondenza per wind_speed={wind_speed}. Passando al prossimo risultato ottimo...")
#             print("\n")
#             jj += 1  # Incrementa jj, proviamo con il prossimo risultato ottimo
#     except:
#         jj += 1

# print(assigned)
# # Stampa l'associazione delle velocità del vento reali
# print(f"Associazione completata. Velocità del vento reali: {real_wind_speed}")

# power_curve.wind_speeds = real_wind_speed

# print(len(power_curve.wind_speeds), len(power_curve.x_opts))

# with open("config/config_V3_speed.yaml") as f:
#     config = yaml.safe_load(f)
# # Parse system properties and bounds

# sys_props = parse_system_properties_and_bounds(config)
# sys_props = SystemProperties(sys_props)

# # Parse sim settings
# control_mode, time_step_RO, time_step_RO, time_step_RIRO = parse_sim_settings(config)

# # Parse optimisation settings, free variables, constraints
# opt_settings = parse_opt_settings(config)
# otp_var_enabled_idx, init_vals = parse_opt_variables(config)
# cons_enabled_idx, cons_param_vals = parse_constraints(config)
# cons_enabled_idx = range(12)

# profile, roughness_length, ref_height, ref_windspeeds = parse_environment(config)

# # Cycle simulation settings for different phases of the power curves.
# cycle_sim_settings_pc = {
#     'cycle': {
#         'traction_phase': TractionPhasePattern,
#         'include_transition_energy': True,
#     },
#     'retraction': {
#         'time_step': time_step_RO},

#     'transition': {
#         'time_step': time_step_RIRO,
#     },
#     'traction': {
#         'time_step': time_step_RO,
#     },
# }

# limits_refined = {'vw_100m_cut_in': [], 'vw_100m_cut_out': []}

# # Pre-configure environment object for optimizations by setting normalized wind profile.
# if profile == 'logarithmic':
#     env = LogProfile()
#     env.set_reference_height(ref_height)
#     env.set_roughness_length(roughness_length)
# else:
#     NotImplementedError('Only logarithmic profiles are supported at the moment!')



# # Optimization variables: Force RO, Force RI, Avg. elevation [rad], Rel. elevation [rad],
# #                          Max. azimuth [rad], Reel-in tether length [m], Minimum tether length [m]
# op_cycle_pc = OptimizerCycle(cycle_sim_settings_pc, sys_props, env, otp_var_enabled_idx,
#                             cons_enabled_idx, cons_param_vals, force_or_speed_control=control_mode)

# p_cycle = [kpis['average_power']['cycle'] for kpis in power_curve.performance_indicators]


# power_curve.plot_optimal_trajectories(circle_radius=sys_props.min_tether_length_min_limit,
#                                 elevation_line=sys_props.avg_elevation_min_limit)

# power_curve.plot_optimization_results(op_cycle_pc.opt_variable_labels, op_cycle_pc.bounds_real_scale,
#                                 [sys_props.tether_force_min_limit, sys_props.tether_force_max_limit],
#                                 [sys_props.reeling_speed_min_limit, sys_props.reeling_speed_max_limit])

# n_cwp = [kpis['n_crosswind_patterns'] for kpis in power_curve.performance_indicators]

# export_to_csv_single_profile(power_curve.wind_speeds, max(power_curve.wind_speeds), p_cycle, power_curve.x_opts, n_cwp, power_curve.optimization_details, control_mode)

filename = 'output/power_curve_log_profile.csv'
power_curve = pd.read_csv(filename, sep=';')
success_mask = power_curve['success [-]'] == True

print(power_curve.columns)

wind_speeds = power_curve['v_100m [m/s]']
cycle_powers = power_curve['P_cycle [W]']
reelout_speeds = power_curve['v_RO [m/s]']
reelin_forces = power_curve['F_RI [N]']
rel_tethas = np.rad2deg(power_curve['theta_rel_RO [rad]'])
avg_tethas = np.rad2deg(power_curve['theta_avg_RO [rad]'])
max_phis = np.rad2deg(power_curve['phi_max_RO [rad]'])
min_tether_lengths = power_curve['min_length_tether [m]']
tether_strokes = power_curve['stroke_tether [m]']

fig, axs = plt.subplots(2, 4)

for ax in axs.flatten():
    ax.grid(visible=True)

axs[0,0].plot(wind_speeds, reelout_speeds)
axs[0,0].set_ylabel(r'$v_{RO}$ (m/s)')

axs[0,1].plot(wind_speeds, reelin_forces/1000)
axs[0,1].set_ylabel(r'$F_{RI}$ (kN)')

axs[0,2].plot(wind_speeds, avg_tethas)
axs[0,2].set_ylabel(r'$\theta_{avg}$ (deg)')

axs[0,3].plot(wind_speeds, rel_tethas)
axs[0,3].set_ylabel(r'$\theta_{rel}$ (deg)')

axs[1,0].plot(wind_speeds, max_phis)
axs[1,0].set_ylabel(r'$\phi_{max}$ (deg)')

axs[1,1].plot(wind_speeds, min_tether_lengths)
axs[1,1].set_ylabel(r'$L_{min}$ (m)')

axs[1,2].plot(wind_speeds, tether_strokes)
axs[1,2].set_ylabel(r'$\Delta L$ (m)')

axs[1,3].plot(wind_speeds, cycle_powers/1000)
axs[1,3].set_ylabel(r'$P_{m}$ (kW)')
plt.tight_layout()

# perform_simulation_from_power_curve(idx = -1)

plt.show()    
