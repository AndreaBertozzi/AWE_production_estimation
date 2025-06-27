import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import *
from qsm import *
from cycle_optimizer import OptimizerCycle
from power_curve_constructor import PowerCurveConstructor

def export_to_csv_single_profile(v, v_cut_out, p, x_opts, n_cwp, opt_details):
    
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
    df = pd.DataFrame(df)
    df.to_csv('output/power_curve_log_profile.csv', index=False, sep=";")

def generate_power_curves_single_profile(config_filename):
    """Determine power curves - requires estimates of the cut-in and cut-out wind speed to be available."""
    with open(config_filename) as f:
        config = yaml.safe_load(f)
    # Parse system properties and bounds

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    # Parse sim settings
    control_mode, time_step_RO, time_step_RO, time_step_RIRO = parse_sim_settings(config)

    # Parse optimisation settings, free variables, constraints
    opt_settings = parse_opt_settings(config)
    otp_var_enabled_idx, init_vals = parse_opt_variables(config)
    cons_enabled_idx, cons_param_vals = parse_constraints(config)

    profile, roughness_length, ref_height, ref_windspeeds = parse_environment(config)

    # Cycle simulation settings for different phases of the power curves.
    cycle_sim_settings_pc = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': True,
        },
        'retraction': {
            'time_step': time_step_RO},

        'transition': {
            'time_step': time_step_RIRO,
        },
        'traction': {
            'time_step': time_step_RO,
        },
    }

    limits_refined = {'vw_100m_cut_in': [], 'vw_100m_cut_out': []}

    # Pre-configure environment object for optimizations by setting normalized wind profile.
    if profile == 'logarithmic':
        env = LogProfile()
        env.set_reference_height(ref_height)
        env.set_roughness_length(roughness_length)
    else:
        NotImplementedError('Only logarithmic profiles are supported at the moment!')

    # Optimizations are performed sequentially with increased wind speed. The solution of the previous optimization
    # is used to initialise the next. With trial and error the lower configuration, a reasonably robust approach is
    # obtained. The power curves may however still exhibit discontinuities and therefore need to be checked and
    # possibly post-processed.

    # The optimization incessantly fails for the estimated cut-out wind speed. Therefore, the highest wind speed for
    # which the optimization is performed is somewhat lower than the estimated cut-out wind speed.
    wind_speeds = np.array(ref_windspeeds)
    vw_cut_in = wind_speeds[0]
    vw_cut_out = wind_speeds[-1]
    if vw_cut_out >= 15:
        wind_speeds[-1] = vw_cut_out - 1    
        wind_speeds = np.concatenate((wind_speeds, np.linspace(vw_cut_out-1, vw_cut_out-0.01, 6)))
    
    
    # Optimization variables: Force RO, Force RI, Avg. elevation [rad], Rel. elevation [rad],
    #                          Max. azimuth [rad], Reel-in tether length [m], Minimum tether length [m]
    op_cycle_pc = OptimizerCycle(cycle_sim_settings_pc, sys_props, env, otp_var_enabled_idx,
                                cons_enabled_idx, cons_param_vals, force_or_speed_control=control_mode)
        
    # Configuration of the sequential optimizations for which is differentiated between the wind speed ranges
    # bounded above by the wind speed of the dictionary key. If dx0 does not contain only zeros, the starting point
    # of the new optimization is not the solution of the preceding optimization.
    op_seq = {
        15.: {'power_optimizer': op_cycle_pc, 'dx0': np.array([0., 0., 0., 0., 0., 0., 0.])},
        np.inf: {'power_optimizer': op_cycle_pc, 'dx0': np.array([0., 0., 0., 0., 0., 0., 0.])}
        }

    # Start optimizations.
    
    pc = PowerCurveConstructor(wind_speeds)
    pc.optimization_settings = opt_settings 
    pc.run_predefined_sequence(op_seq, init_vals)
    pc.export_results('output/power_curve_log_profile.pickle')


    # Refine the wind speed operational limits to wind speeds for which optimal solutions are found.
    limits_refined['vw_100m_cut_in'].append(pc.wind_speeds[0])
    limits_refined['vw_100m_cut_out'].append(pc.wind_speeds[-1])

    print("Cut-in and -out speeds changed from [{:.3f}, {:.3f}] to "
            "[{:.3f}, {:.3f}].".format(vw_cut_in, vw_cut_out, pc.wind_speeds[0], pc.wind_speeds[-1]))
    
    p_cycle = [kpis['average_power']['cycle'] for kpis in pc.performance_indicators]


    pc.plot_optimal_trajectories(circle_radius=sys_props.min_tether_length_min_limit,
                                  elevation_line=sys_props.avg_elevation_min_limit)
    
    pc.plot_optimization_results(op_cycle_pc.opt_variable_labels, op_cycle_pc.bounds_real_scale,
                                    [sys_props.tether_force_min_limit, sys_props.tether_force_max_limit],
                                    [sys_props.reeling_speed_min_limit, sys_props.reeling_speed_max_limit])

    n_cwp = [kpis['n_crosswind_patterns'] for kpis in pc.performance_indicators]
    
    export_to_csv_single_profile(pc.wind_speeds, vw_cut_out, p_cycle, pc.x_opts, n_cwp, pc.optimization_details)

    df = pd.DataFrame(limits_refined)
    print(df)
    if not os.path.exists('output/wind_limits_refined.csv'):
        df.to_csv('output/wind_limits_refined.csv')
    else:
        print("Skipping exporting operational limits.")

    return pc

def load_power_curve_single_profile_results_and_plot_trajectories(config_filename):
    """Plot trajectories from previously generated power curve."""
    with open(config_filename) as f:
        config = yaml.safe_load(f)
    # Parse system properties and bounds
    control_mode, _, _, _ = parse_sim_settings(config) 
    if control_mode == 'force':
        labels = [r'$F_{T, RO}$', r'$F_{T, RI}$', r'$\theta_{avg}$', r'$\theta_{rel}$', r'$\phi_{max}$',\
                                r'$\Delta L$', r'$L_{min}$']

    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)

    pc = PowerCurveConstructor(None)
    pc.import_results('output/power_curve_log_profile.pickle')
    pc.plot_optimal_trajectories()
    plt.gcf().set_size_inches(5.5, 3.5)
    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
    pc.plot_optimization_results(opt_variable_labels=labels, tether_force_limits=[sys_props.tether_force_min_limit, sys_props.tether_force_max_limit],
                                reeling_speed_limits=[sys_props.reeling_speed_min_limit, sys_props.reeling_speed_max_limit])

    return pc

def compare_kpis(power_curves):
    """Plot how performance indicators change with wind speed for all generated power curves."""
    fig_nums = [plt.figure().number for _ in range(7)]
    for pc in power_curves:
        plt.figure(fig_nums[0])
        f_out_min = [kpis['min_tether_force']['out'] for kpis in pc.performance_indicators]
        f_out_max = [kpis['max_tether_force']['out'] for kpis in pc.performance_indicators]
        f_out = [x[0] for x in pc.x_opts]
        p = plt.plot(pc.wind_speeds, f_out)
        clr = p[-1].get_color()
        plt.plot(pc.wind_speeds, f_out_min, linestyle='None', marker=6, color=clr, markersize=7, markerfacecolor="None")
        plt.plot(pc.wind_speeds, f_out_max, linestyle='None', marker=7, color=clr, markerfacecolor="None")
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-out force [N]')

        plt.figure(fig_nums[1])
        f_in_min = [kpis['min_tether_force']['in'] for kpis in pc.performance_indicators]
        f_in_max = [kpis['max_tether_force']['in'] for kpis in pc.performance_indicators]
        f_in = [x[1] for x in pc.x_opts]
        p = plt.plot(pc.wind_speeds, f_in)
        clr = p[-1].get_color()
        plt.plot(pc.wind_speeds, f_in_min, linestyle='None', marker=6, color=clr, markersize=7, markerfacecolor="None")
        plt.plot(pc.wind_speeds, f_in_max, linestyle='None', marker=7, color=clr, markerfacecolor="None")
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-in force [N]')

        plt.figure(fig_nums[2])
        f_in_min = [kpis['min_reeling_speed']['out'] for kpis in pc.performance_indicators]
        f_in_max = [kpis['max_reeling_speed']['out'] for kpis in pc.performance_indicators]
        p = plt.plot(pc.wind_speeds, f_in_min)
        clr = p[-1].get_color()
        plt.plot(pc.wind_speeds, f_in_max, linestyle='None', marker=7, color=clr, markerfacecolor="None")
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-out speed [m/s]')

        plt.figure(fig_nums[3])
        n_cwp = [kpis['n_crosswind_patterns'] for kpis in pc.performance_indicators]
        plt.plot(pc.wind_speeds, n_cwp)
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Number of cross-wind patterns [-]')

        plt.figure(fig_nums[4])
        elev_angles = [x_opt[2]*180./np.pi for x_opt in pc.x_opts]
        plt.plot(pc.wind_speeds, elev_angles)
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Reel-out elevation angle [deg]')

        plt.figure(fig_nums[5])
        power = [kpis['average_power']['cycle'] for kpis in pc.performance_indicators]
        plt.plot(pc.wind_speeds, power)
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Average cycle power [W]')

        plt.figure(fig_nums[6])
        power = [kpis['average_power']['out'] for kpis in pc.performance_indicators]
        plt.plot(pc.wind_speeds, power)
        plt.grid(True)
        plt.xlabel('$v_{w,100m}$ [m/s]')
        plt.ylabel('Average reel-out power [W]')

if __name__ == "__main__":
    pc = generate_power_curves_single_profile('config/config.yaml')
    #pc = load_power_curve_single_profile_results_and_plot_trajectories('config.yaml')
    plt.show()





