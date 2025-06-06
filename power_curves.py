import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from qsm import *
from cycle_optimizer import OptimizerCycle
from power_curve_constructor import PowerCurveConstructor, WindSpeedLimitsEstimator




def generate_power_curves(loc='mmc', n_clusters=8, sys_props = SystemProperties({})):
    """Determine power curves - requires estimates of the cut-in and cut-out wind speed to be available."""
    suffix = '_{}{}'.format(n_clusters, loc)

    wind_speed_limits_estimator = WindSpeedLimitsEstimator(sys_props, l_min = 150, l_max = 250)
    wind_speed_limits_estimator.estimate_wind_speed_operational_limits(n_clusters=1)

    limit_estimates = pd.read_csv('output/wind_limits_estimate{}.csv'.format(suffix))

    # Cycle simulation settings for different phases of the power curves.
    cycle_sim_settings_pc = {
        'cycle': {
            'traction_phase': TractionPhasePattern,
            'include_transition_energy': False,
        },
        'retraction': {
            'time_step': 0.25,},

        'transition': {
            'time_step': 0.25,
        },
        'traction': {
            'time_step': 0.25,
        },
    }

    ax_pcs = plt.subplots(2, 1)[1]
    for a in ax_pcs: a.grid()

    limits_refined = {'vw_100m_cut_in': [], 'vw_100m_cut_out': []}
    res_pcs = []
    for i_profile in range(1, n_clusters+1):
        # Pre-configure environment object for optimizations by setting normalized wind profile.
        env = wind_speed_limits_estimator.create_environment(suffix, i_profile)

        # Optimizations are performed sequentially with increased wind speed. The solution of the previous optimization
        # is used to initialise the next. With trial and error the lower configuration, a reasonably robust approach is
        # obtained. The power curves may however still exhibit discontinuities and therefore need to be checked and
        # possibly post-processed.

        # The optimization incessantly fails for the estimated cut-out wind speed. Therefore, the highest wind speed for
        # which the optimization is performed is somewhat lower than the estimated cut-out wind speed.
        vw_cut_in = limit_estimates.iloc[i_profile-1]['vw_100m_cut_in']
        vw_cut_out = limit_estimates.iloc[i_profile-1]['vw_100m_cut_out']
        wind_speeds = np.arange(vw_cut_in, vw_cut_out-1, 1)
        wind_speeds = np.concatenate((wind_speeds, np.linspace(vw_cut_out-1, vw_cut_out-0.01, 8)))

        # Optimization variables: Force RO, Force RI, Speed RIRO, Avg. elevation [rad], Rel. elevation [rad],
        #                          Max. azimuth [rad], Reel-in tether length [m], Minimum tether length [m]
        op_cycle_pc = OptimizerCycle(cycle_sim_settings_pc, sys_props, env, reduce_x = np.array([0, 1, 2, 3, 5, 6, 7]))        

        
        # Configuration of the sequential optimizations for which is differentiated between the wind speed ranges
        # bounded above by the wind speed of the dictionary key. If dx0 does not contain only zeros, the starting point
        # of the new optimization is not the solution of the preceding optimization.
        op_seq = {
            17.: {'power_optimizer': op_cycle_pc, 'dx0': np.array([0., 0., 0., 0., 0., 0., 0.])},
            np.inf: {'power_optimizer': op_cycle_pc, 'dx0': np.array([0., 0., 0.1, 0., 0., 0., 0.])}
        }

        # Define starting point for the very first optimization at the cut-in wind speed.
        critical_force = limit_estimates.iloc[i_profile-1]['tether_force_cut_in']

        # Optimization variables: Force RO, Force RI, Speed RIRO, Avg. elevation [rad], Rel. elevation [rad],
        #                          Max. azimuth [rad], Reel-in tether length [m], Minimum tether length [m]
        x0 = np.array([critical_force, 1000, -0.5, 30*np.pi/180, 6*np.pi/180,
                        35*np.pi/180,  150., 150.])

        # Start optimizations.
        pc = PowerCurveConstructor(wind_speeds)
        pc.run_predefined_sequence(op_seq, x0)
        pc.export_results('output/power_curve{}{}.pickle'.format(suffix, i_profile))
        res_pcs.append(pc)

        # Refine the wind speed operational limits to wind speeds for which optimal solutions are found.
        limits_refined['vw_100m_cut_in'].append(pc.wind_speeds[0])
        limits_refined['vw_100m_cut_out'].append(pc.wind_speeds[-1])

        print("Cut-in and -out speeds changed from [{:.3f}, {:.3f}] to "
              "[{:.3f}, {:.3f}].".format(vw_cut_in, vw_cut_out, pc.wind_speeds[0], pc.wind_speeds[-1]))

        # Plot power curve together with that of the other wind profile shapes.
        p_cycle = [kpis['average_power']['cycle'] for kpis in pc.performance_indicators]
        ax_pcs[0].plot(pc.wind_speeds, p_cycle, label=i_profile)
        ax_pcs[1].plot(pc.wind_speeds/vw_cut_out, p_cycle, label=i_profile)

        pc.plot_optimal_trajectories()
        pc.plot_optimization_results(op_cycle_pc.OPT_VARIABLE_LABELS, op_cycle_pc.bounds_real_scale,
                                     [sys_props.tether_force_min_limit, sys_props.tether_force_max_limit],
                                     [sys_props.reeling_speed_min_limit, sys_props.reeling_speed_max_limit])

        n_cwp = [kpis['n_crosswind_patterns'] for kpis in pc.performance_indicators]
        export_to_csv(pc.wind_speeds, vw_cut_out, p_cycle, pc.x_opts, n_cwp, i_profile, suffix)
    ax_pcs[1].legend()

    df = pd.DataFrame(limits_refined)
    print(df)
    if not os.path.exists('output/wind_limits_refined{}.csv'.format(suffix)):
        df.to_csv('output/wind_limits_refined{}.csv'.format(suffix))
    else:
        print("Skipping exporting operational limits.")

    return res_pcs


def load_power_curve_results_and_plot_trajectories(loc='mmc', n_clusters=8, i_profile=1):
    """Plot trajectories from previously generated power curve."""
    pc = PowerCurveConstructor(None)
    suffix = '_{}{}{}'.format(n_clusters, loc, i_profile)
    pc.import_results('output/power_curve{}.pickle'.format(suffix))
    pc.plot_optimal_trajectories(wind_speed_ids=[0, 9, 18, 33, 50])
    plt.gcf().set_size_inches(5.5, 3.5)
    plt.subplots_adjust(top=0.99, bottom=0.1, left=0.12, right=0.65)
    pc.plot_optimization_results()

    return pc


def compare_kpis(power_curves):
    """Plot how performance indicators change with wind speed for all generated power curves."""
    fig_nums = [plt.figure().number for _ in range(5)]
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


def export_to_csv(v, v_cut_out, p, x_opts, n_cwp, i_profile, suffix):
    df = {
        'v_100m [m/s]': v,
        'v/v_cut-out [-]': v/v_cut_out,
        'P_cycle [W]': p,
        'F_RO [N]': [x[0] for x in x_opts],
        'F_RI [N]': [x[1] for x in x_opts],
        'v_RIRO [m/s]': [x[2] for x in x_opts],
        'theta_avg_RO [rad]': [x[3] for x in x_opts],        
        'theta_rel_RO [rad]': [x[4] for x in x_opts],
        'phi_max_RO [rad]': [x[5] for x in x_opts],
        'stroke_tether [m]': [x[6] for x in x_opts],
        'min_length_tether [m]': [x[7] for x in x_opts],
        'n_crosswind_patterns [-]': n_cwp,
    }
    df = pd.DataFrame(df)
    df.to_csv('output/power_curve{}{}.csv'.format(suffix, i_profile), index=False, sep=";")

if __name__ == "__main__":
    #estimate_wind_speed_operational_limits(n_clusters=1, loc='mmc')
    #pcs = generate_power_curves(loc='mmc', n_clusters=1, sys_props=sys_props_v9)
    pcs = []
    for idx in range(1, 9):
        pcs.append(load_power_curve_results_and_plot_trajectories('mmc', n_clusters=8, i_profile=idx))


    compare_kpis(pcs)
    plt.show()
   

