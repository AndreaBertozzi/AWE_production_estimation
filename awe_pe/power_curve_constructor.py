import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import pickle

from qsm import *
from awe_pe.utils import flatten_dict

class PowerCurveConstructor:
    def __init__(self, wind_speeds):
        self.wind_speeds = wind_speeds
        self.optimization_settings = {'max_iter': 50, 'i_print': 0, 'ftol': 1e-6, 'eps': 1e-6}

        self.x_opts = []
        self.x0 = []
        self.optimization_details = []
        self.constraints = []
        self.performance_indicators = []

    def run_optimization(self, wind_speed, power_optimizer, x0):
        #TODO: evaluate if robustness can be improved by running multiple optimizations using different starting points
        power_optimizer.environment_state.set_reference_wind_speed(wind_speed)

        print("x0:", x0)
        power_optimizer.x0_real_scale = x0
        x_opt = power_optimizer.optimize(maxiter = self.optimization_settings['maxiter'],
                                         iprint  = self.optimization_settings['iprint'],
                                         eps      = self.optimization_settings['eps'],
                                         ftol     = self.optimization_settings['ftol'])
        self.x0.append(x0)
        self.x_opts.append(x_opt)
        self.optimization_details.append(power_optimizer.op_res)

        try:
            cons, kpis = power_optimizer.eval_point()
            sim_successful = True
        except (SteadyStateError, OperationalLimitViolation, PhaseError) as e:
            print("Error occurred while evaluating the resulting optimal point: {}".format(e))
            cons, kpis = power_optimizer.eval_point(relax_errors=True)
            sim_successful = False

        self.constraints.append(cons)
        kpis['sim_successful'] = sim_successful
        self.performance_indicators.append(kpis)
        return x_opt, sim_successful

    def run_predefined_sequence(self, seq, x0_start):
        wind_speed_tresholds = iter(sorted(list(seq)))
        vw_switch = next(wind_speed_tresholds)

        x_opt_last, vw_last = None, None
        for i, vw in enumerate(self.wind_speeds):
            if vw > vw_switch:
                vw_switch = next(wind_speed_tresholds)

            power_optimizer = seq[vw_switch]['power_optimizer']
            dx0 = seq[vw_switch].get('dx0', None)

            if x_opt_last is None:
                x0_next = x0_start
            else:
                x0_next = x_opt_last + dx0*(vw - vw_last)

            print("[{}] Processing v={:.2f}m/s".format(i, vw))
            try:
                x_opt, sim_successful = self.run_optimization(vw, power_optimizer, x0_next)
            except (OperationalLimitViolation, SteadyStateError, PhaseError):
                try:  # Retry for a slightly different wind speed.
                    x_opt, sim_successful = self.run_optimization(vw+1e-2, power_optimizer, x0_next)
                    self.wind_speeds[i] = vw+1e-2
                except (OperationalLimitViolation, SteadyStateError, PhaseError):
                    self.wind_speeds = self.wind_speeds[:i]
                    print("Optimization sequence stopped prematurely due to failed optimization. {:.1f} m/s is the "
                          "highest wind speed for which the optimization was successful.".format(self.wind_speeds[-1]))
                    break

            if sim_successful:
                x_opt_last = x_opt
                vw_last = vw

    def plot_optimal_trajectories(self, wind_speed_ids=None, ax=None, circle_radius=200, elevation_line=25*np.pi/180):
        if ax is None:
            plt.figure(figsize=(6, 3.5))
            plt.subplots_adjust(right=0.65)
            ax = plt.gca()

        if wind_speed_ids is None:
            if len(self.wind_speeds) > 8:
                wind_speed_ids = [int(a) for a in np.linspace(0, len(self.wind_speeds)-1, 6)]
            else:
                wind_speed_ids = range(len(self.wind_speeds))

        for i in wind_speed_ids:
            v = self.wind_speeds[i]
            kpis = self.performance_indicators[i]
            if kpis is None:
                print("No trajectory available for {} m/s wind speed.".format(v))
                continue

            x_kite, z_kite = zip(*[(kp.x, kp.z) for kp in kpis['kinematics']])

            ax.plot(x_kite, z_kite, label="$v_{100m}$="+"{:.1f} ".format(v) + "m s$^{-1}$")

        # Plot semi-circle at constant tether length bound.
        phi = np.linspace(0, 2*np.pi/3, 40)
        x_circle = np.cos(phi) * circle_radius
        z_circle = np.sin(phi) * circle_radius
        ax.plot(x_circle, z_circle, 'k--', linewidth=1)

        # Plot elevation line.
        x_elev = np.linspace(0, 400, 40)
        z_elev = np.tan(elevation_line)*x_elev
        ax.plot(x_elev, z_elev, 'k--', linewidth=1)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.set_xlim([-70, None])
        ax.set_ylim([0, None])
        ax.grid()
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    def plot_optimization_results(self, opt_variable_labels=None, opt_variable_bounds=None, tether_force_limits=None,
                                  reeling_speed_limits=None):
        assert self.x_opts, "No optimization results available for plotting."
        xf, x0 = self.x_opts, self.x0
        cons = self.constraints
        kpis, opt_details = self.performance_indicators, self.optimization_details
        try:
            performance_indicators = next(list(flatten_dict(kpi)) for kpi in kpis if kpi is not None)
        except StopIteration:
            performance_indicators = []

        n_opt_vars = len(xf[0])
        fig, ax = plt.subplots(max([n_opt_vars, 6]), 2, sharex=True)
        # In the left column plot each optimization variable against the wind speed.
        scl = [1e-3, 1e-3, 180/np.pi, 180/np.pi, 180/np.pi, 1, 1]
        for i in range(n_opt_vars):
            # Plot optimal and starting points.
            ax[i, 0].plot(self.wind_speeds, [a[i]*scl[i] for a in xf], label='x_opt')
            ax[i, 0].plot(self.wind_speeds, [a[i]*scl[i] for a in x0], 'o', markerfacecolor='None', label='x0')

            if opt_variable_labels:
                label = opt_variable_labels[i]
                ax[i, 0].set_ylabel(label)
            else:
                ax[i, 0].set_ylabel("x[{}]".format(i))

            if opt_variable_bounds is not None:
                ax[i, 0].axhline(opt_variable_bounds[i, 0]*scl[i], linestyle='--', color='k')
                ax[i, 0].axhline(opt_variable_bounds[i, 1]*scl[i], linestyle='--', color='k')

            ax[i, 0].grid()
        ax[0, 0].legend()

        # In the right column plot the number of iterations in the upper panel.
        nits = np.array([od['nit'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds, nits)
        mask_opt_failed = np.array([~od['success'] for od in opt_details])
        ax[0, 1].plot(self.wind_speeds[mask_opt_failed], nits[mask_opt_failed], 'x', label='opt fail')
        mask_sim_failed = np.array([~kpi['sim_successful'] for kpi in kpis])
        ax[0, 1].plot(self.wind_speeds[mask_sim_failed], nits[mask_sim_failed], 'x', label='sim fail')
        ax[0, 1].grid()
        ax[0, 1].legend()
        ax[0, 1].set_ylabel('# iter [-]')

        # In the second panel, plot the optimal power.
        cons_treshold = -.1
        mask_cons_adhered = np.array([all([c >= cons_treshold for c in con]) for con in cons])
        mask_plot_power = ~mask_sim_failed & mask_cons_adhered
        power = np.array([kpi['average_power']['cycle'] for kpi in kpis])
        power[~mask_plot_power] = np.nan
        ax[1, 1].plot(self.wind_speeds, power/1000)
        ax[1, 1].grid()
        ax[1, 1].set_ylabel(r'$P_{mech}$ [kW]')

        # In the third panel, plot the tether force related performance indicators.
        max_force_in = np.array([kpi['max_tether_force']['in'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_in/1000, label=r'$F_{T, RI, max}$')
        max_force_out = np.array([kpi['max_tether_force']['out'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_out/1000, label=r'$F_{T, RO, max}$')
        max_force_trans = np.array([kpi['max_tether_force']['trans'] for kpi in kpis])
        ax[2, 1].plot(self.wind_speeds, max_force_trans/1000, label=r'$F_{T, RIRO, max}$')
        if tether_force_limits:
            ax[2, 1].axhline(tether_force_limits[0]/1000, linestyle='--', color='k')
            ax[2, 1].axhline(tether_force_limits[1]/1000, linestyle='--', color='k')

        ax[2, 1].grid()
        ax[2, 1].set_ylabel(r'$F_T$ [kN]')
        ax[2, 1].legend(loc=2)
        
        # Plot reeling speed related performance indicators.
        max_speed_in = np.array([kpi['max_reeling_speed']['in'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_in, label=r'$v_{T, RI, max}$')
        max_speed_out = np.array([kpi['max_reeling_speed']['out'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, max_speed_out, label=r'$v_{T, RO, max}$')
        min_speed_in = np.array([kpi['min_reeling_speed']['in'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_in, label=r'$v_{T, RI, mim}$')
        min_speed_out = np.array([kpi['min_reeling_speed']['out'] for kpi in kpis])
        ax[3, 1].plot(self.wind_speeds, min_speed_out, label=r'$v_{T, RO, min}$')
        if reeling_speed_limits:
            ax[3, 1].axhline(reeling_speed_limits[0], linestyle='--', color='k')
            ax[3, 1].axhline(reeling_speed_limits[1], linestyle='--', color='k')
        ax[3, 1].grid()
        ax[3, 1].set_ylabel(r'$v_T$ [m/s]')
        ax[3, 1].legend(loc=2)

        # Plot constraint matrix.
        cons_matrix = np.array(cons).transpose()
        n_cons = cons_matrix.shape[0]

        cons_treshold_magenta = -.1
        cons_treshold_red = -1e-6

        # Assign color codes based on the constraint values.
        color_code_matrix = np.where(cons_matrix < cons_treshold_magenta, -2, 0)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_magenta) & (cons_matrix < cons_treshold_red), -1,
                                     color_code_matrix)
        color_code_matrix = np.where((cons_matrix >= cons_treshold_red) & (cons_matrix < 1e-3), 1, color_code_matrix)
        color_code_matrix = np.where(cons_matrix == 0., 0, color_code_matrix)
        color_code_matrix = np.where(cons_matrix >= 1e-3, 2, color_code_matrix)

        fig.delaxes(ax[5, 1])
        # Plot color code matrix.
        ax_pos = ax[4, 1].get_position()
        w = ax_pos.x1 - ax_pos.x0
        h = ax_pos.y1 - ax_pos.y0
        ax[4, 1].set_position([ax_pos.x0, ax_pos.y0-0.1, w, h])
        ax_pos = ax[4, 1].get_position()
        cmap = mpl.colors.ListedColormap(['r', 'm', 'y', 'g', 'b'])
        bounds = [-2, -1, 0, 1, 2]
        mpl.colors.BoundaryNorm(bounds, cmap.N)
        im1 = ax[4, 1].matshow(color_code_matrix, cmap=cmap, vmin=bounds[0], vmax=bounds[-1],
                                    extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
        
        ax[4, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[4, 1].set_yticklabels(range(n_cons))
        ax[4, 1].set_ylabel('Cons. ID\'s')

        # Add colorbar.
        h_cbar = ax_pos.y1 - ax_pos.y0
        w_cbar = .012
        cbar_ax = fig.add_axes([ax_pos.x1 , ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = np.arange(-2+4/10., 2., 4/5.)
        cbar_ticks_labels = ['<-.1', '<0', '0', '~0', '>0']
        cbar = fig.colorbar(im1, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        # Plot constraint matrix with linear mapping the colors from data values between plot_cons_range.
        plot_cons_range = [-.1, .1]
        im2 = ax[6, 1].matshow(cons_matrix, vmin=plot_cons_range[0], vmax=plot_cons_range[1], cmap=mpl.cm.YlGnBu_r,
                               extent=[self.wind_speeds[0], self.wind_speeds[-1], n_cons, 0])
        ax[6, 1].set_yticks(np.array(range(n_cons))+.5)
        ax[6, 1].set_yticklabels(range(n_cons))
        ax[6, 1].set_ylabel('Cons. ID\'s')

        # Add colorbar.
        ax_pos = ax[6, 1].get_position()
        cbar_ax = fig.add_axes([ax_pos.x1 + 0.101, ax_pos.y0, w_cbar, h_cbar])
        cbar_ticks = plot_cons_range[:]
        cbar_ticks_labels = [str(v) for v in cbar_ticks]
        if plot_cons_range[0] < np.min(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(1, np.min(cons_matrix))
            cbar_ticks_labels.insert(1, 'min: {:.2E}'.format(np.min(cons_matrix)))
        if plot_cons_range[0] < np.max(cons_matrix) < plot_cons_range[1]:
            cbar_ticks.insert(-1, np.max(cons_matrix))
            cbar_ticks_labels.insert(-1, 'max: {:.2E}'.format(np.max(cons_matrix)))
        cbar = fig.colorbar(im2, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks_labels)

        ax[-1, 0].set_xlabel('Wind speeds at 100 m [m/s]')
        ax[3, 1].set_xlabel('Wind speeds at 100 m [m/s]')
        
        ax[0, 0].set_xlim([self.wind_speeds[0], self.wind_speeds[-1]])


    def export_results(self, file_name):
        export_dict = self.__dict__
        with open(file_name, 'wb') as f:
            pickle.dump(export_dict, f)

    def import_results(self, file_name):
        with open(file_name, 'rb') as f:
            import_dict = pickle.load(f)
        for k, v in import_dict.items():
            setattr(self, k, v)


class WindSpeedLimitsEstimator:
    # TODO: Understand why cut-out wind speed does not converge
    def __init__(self, sys_props):
        self.sys_props = sys_props
        self.l_min = 200
        self.l_max = 350 
        self.max_azimuth_RO_cutin = 35*np.pi/180
        self.avg_elevation_RO_cutin = 30*np.pi/180

        self.rel_elevation_RO_cutout = 9*np.pi/180
        self.max_azimuth_RO_cutout = 35*np.pi/180
        

    def calc_tether_force_traction(self, env_state, straight_tether_length, azimuth, elevation, course = np.pi/2):
        """"Calculate tether force for the minimum allowable reel-out speed and given wind conditions and tether length."""
        kinematics = KiteKinematics(straight_tether_length, azimuth, elevation, course)
        env_state.calculate(kinematics.z)
        self.sys_props.update(kinematics.straight_tether_length, True)

        ss = SteadyState({'enable_steady_state_errors': True})
        ss.control_settings = ('reeling_speed', self.sys_props.reeling_speed_min_limit)
        ss.find_state(self.sys_props, env_state, kinematics)

        return ss.tether_force_ground

    def get_cut_in_wind_speed(self, env = LogProfile()):
        """Iteratively determine lowest wind speed for which, along the entire reel-out path, feasible steady flight states
        with the minimum allowable reel-out speed are found."""
        dv = 1e-2  # Step size [m/s].
        v0 = 3.6  # Lowest wind speed [m/s] with which the iteration is started.
        
        v = v0
        while True:
            env.set_reference_wind_speed(v)
            try:
                # Setting tether force as setpoint in qsm yields infeasible region
                tether_force_start = self.calc_tether_force_traction(env, self.l_min, self.max_azimuth_RO_cutin, self.avg_elevation_RO_cutin)
                tether_force_end = self.calc_tether_force_traction(env, self.l_max, self.max_azimuth_RO_cutin, self.avg_elevation_RO_cutin)

                start_critical = tether_force_end > tether_force_start
                if start_critical:
                    critical_force = tether_force_start
                else:
                    critical_force = tether_force_end

                if tether_force_start > self.sys_props.tether_force_min_limit and \
                        tether_force_end > self.sys_props.tether_force_min_limit:
                    if v == v0:
                        raise ValueError("Starting speed is too high.")
                    return v, start_critical, critical_force
            except SteadyStateError:
                pass

            v += dv
   
    def calc_n_cw_patterns(self, env, traction_force, avg_elevation, rel_elevation, max_azimuth, l_min, l_max):
        """Calculate the number of cross-wind manoeuvres flown."""
        trac = TractionPhasePattern({
                'control': ('tether_force_ground', traction_force),
                'pattern': {'azimuth_angle': max_azimuth, 'rel_elevation_angle': rel_elevation}
            })
        trac.enable_limit_violation_error = True
        # Start and stop conditions of traction phase. Note that the traction phase uses an azimuth angle in contrast to
        # the other phases, which results in jumps of the kite position.
        trac.tether_length_start = l_min
        trac.elevation_angle = TractionConstantElevation(avg_elevation)
        trac.tether_length_end = l_max
        trac.finalize_start_and_end_kite_obj()
        print(trac.kinematics_start.__dict__)
        print(trac.position_end.__dict__)
        
        trac.run_simulation(self.sys_props, env, {'enable_steady_state_errors': True})
        return trac.n_crosswind_patterns


    def get_max_wind_speed_at_elevation(self, env=LogProfile(), avg_elevation = 60*np.pi/180):
        """Iteratively determine maximum wind speed allowing at least one cross-wind manoeuvre during the reel-out phase for
        provided elevation angle."""
        dv = 1e-1  # Step size [m/s].
        v0 = 20.  # Lowest wind speed [m/s] with which the iteration is started.

        # Check if the starting wind speed gives a feasible solution.
        env.set_reference_wind_speed(v0)
        try:
            n_cw_patterns = self.calc_n_cw_patterns(env, self.sys_props.tether_force_max_limit, avg_elevation,
                                                     self.rel_elevation_RO_cutout,
                                                         self.max_azimuth_RO_cutout, self.l_min, self.l_max)
            
        except SteadyStateError as e:
            if e.code != 8:
                raise ValueError("No feasible solution found for first assessed cut out wind speed.")

        # Increase wind speed until number of cross-wind manoeuvres subceeds one.
        v = v0 + dv
        while True:
            env.set_reference_wind_speed(v)
            try:
                n_cw_patterns = self.calc_n_cw_patterns(env, self.sys_props.tether_force_max_limit, avg_elevation, self.rel_elevation_RO_cutout,
                                                     self.max_azimuth_RO_cutout, self.l_min, self.l_max)
                if n_cw_patterns < 1.:
                    return v
            except SteadyStateError as e:
                if e.code != 8:  # Speed is too low to yield a solution when e.code == 8.
                    raise
                    # return None

            if v > 30.:
                raise ValueError("Iteration did not find feasible cut-out speed.")
            v += dv


    def get_cut_out_wind_speed(self, env=LogProfile()):
        """In general, the elevation angle is increased with wind speed as a last means of de-powering the kite. In that
        case, the wind speed at which the elevation angle reaches its upper limit is the cut-out wind speed. This
        procedure verifies if this is indeed the case. Iteratively the elevation angle is determined giving the highest
        wind speed allowing at least one cross-wind manoeuvre during the reel-out phase."""
        beta = 60*np.pi/180.
        dbeta = 1*np.pi/180.
        vw_last = 0.
        while True:
            vw = self.get_max_wind_speed_at_elevation(env, beta)
            if vw is not None:
                if vw <= vw_last:
                    return vw_last, np.rad2deg(beta+dbeta)
                vw_last = vw
            beta -= dbeta

    def create_environment(self, suffix, i_profile):
        """Flatten wind profile shapes resulting from the clustering and use to create the environment object."""
        df = pd.read_csv('wind_resource/'+'profile{}{}.csv'.format(suffix, i_profile), sep=";")
        env = NormalisedWindTable1D()
        env.heights = list(df['h [m]'])
        env.normalised_wind_speeds = list((df['u1 [-]']**2 + df['v1 [-]']**2)**.5)
        return env

    def estimate_wind_speed_operational_limits(self, loc='mmc', n_clusters=8, profile_clustering = False):
        """Estimate the cut-in and cut-out wind speeds for each wind profile shape. These wind speeds are refined when
        determining the power curves."""
        if profile_clustering: suffix = '_{}{}'.format(n_clusters, loc)
        else: 
            suffix = ''
            n_clusters = 1

        fig, ax = plt.subplots(1, 2, figsize=(5.5, 3), sharey=True)
        plt.subplots_adjust(top=0.92, bottom=0.164, left=0.11, right=0.788, wspace=0.13)

        res = {'vw_100m_cut_in': [], 'vw_100m_cut_out': [], 'tether_force_cut_in': []}
        for i_profile in range(1, n_clusters+1):
            if profile_clustering: env = self.create_environment(suffix, i_profile)
            else: env = LogProfile()


            # Get cut-in wind speed.
            env.set_reference_height(200)
            print('Calculating cut-in wind speed...')
            vw_cut_in, _, tether_force_cut_in = self.get_cut_in_wind_speed(env)
            res['vw_100m_cut_in'].append(vw_cut_in)
            res['tether_force_cut_in'].append(tether_force_cut_in)
            print('Cut-in wind speed: ' + str(vw_cut_in))

            # Get cut-out wind speed, which proved to work better when using 250m reference height.
            print('Calculating cut-out wind speed...')
            env.set_reference_height(200)
            vw_cut_out250m, elev = self.get_cut_out_wind_speed(env)
            env.set_reference_wind_speed(vw_cut_out250m)
            vw_cut_out = env.calculate_wind(100.)
            res['vw_100m_cut_out'].append(vw_cut_out)
            print('Cut-out wind speed: ' + str(vw_cut_out))

            # Plot the wind profiles corresponding to the wind speed operational limits and the profile shapes.
            env.set_reference_height(100.)
            env.set_reference_wind_speed(vw_cut_in)
            plt.sca(ax[0])
            env.plot_wind_profile()

            env.set_reference_wind_speed(vw_cut_out)
            plt.sca(ax[1])
            env.plot_wind_profile("{}-{}".format(loc.upper(), i_profile))
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.ylabel('')
            df = pd.DataFrame(res)

        if not os.path.exists('output/wind_limits_estimate{}.csv'.format(suffix)):
            df.to_csv('output/wind_limits_estimate{}.csv'.format(suffix))
        else:
            print("Skipping exporting operational limits...")

        ax[0].set_title("Cut-in")
        ax[0].set_xlim([0, None])
        ax[0].set_ylim([0, 400])
        ax[1].set_title("Cut-out")
        ax[1].set_xlim([0, None])
        ax[1].set_ylim([0, 400])
    
if __name__ == "__main__": 
    from awe_pe.utils import load_config
    sys_props_v9 = SystemProperties(load_config('config.yaml'))
    we = WindSpeedLimitsEstimator(sys_props_v9)
    we.estimate_wind_speed_operational_limits(profile_clustering=False)