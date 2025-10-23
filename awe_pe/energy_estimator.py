import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import calendar
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import Normalize

from qsm import * 
from awe_pe.utils import *
from awe_pe.cycle_optimizer import *

class ElectricalPowerCurve:
    def __init__(self, output_file_path, config_file_path):
        if os.path.exists(output_file_path):
            self.dataframe = pd.read_csv(output_file_path, sep=';')
        else:
            raise(FileNotFoundError('No mechanical power curve found! Use power_curve_single_profile.py to generate one first!'))
    
        if os.path.exists(config_file_path):
            with open(config_file_path) as f:
                self.config = yaml.safe_load(f)
        else:
            raise(FileNotFoundError('No config file found!'))
        
        # Parse electrical efficiencies from config
        el_eff = parse_electrical_etas(self.config)
        self.external_battery_connected = el_eff['external_battery']['connected']

        eta_flight_trajectory = parse_trajectory_etas(self.config)
        self.eta_flight_trajectory = eta_flight_trajectory

        self.self_cons_mot_control = el_eff['motor_controller']['self_consumption']
        self.eta_mot_control = el_eff['motor_controller']['efficiency']

        self.self_cons_DC_AC = el_eff['DC_AC_converter']['self_consumption']
        self.eta_DC_AC = el_eff['DC_AC_converter']['efficiency']

        if self.external_battery_connected:
            self.self_cons_ext_batt = el_eff['external_battery']['self_consumption']
            self.eta_ext_batt = el_eff['external_battery']['efficiency']

        only_successful_opts, smooth, plot_results, fit_settings = parse_power_curve_smoothing(self.config)

        if not smooth and plot_results:
            raise(UserWarning('Enable smoothing if you want to see smoothing results!'))
        
        self.only_successful_opts = only_successful_opts
        self.smooth = smooth
        self.plot_smoothing_results = plot_results
        self.fit_settings = fit_settings

        self.calculate()

    def smooth_opt_results(self):
        if self.only_successful_opts:
            self.dataframe = self.dataframe[self.dataframe['success [-]'] == True]
            self.dataframe.reset_index(inplace = True)     
        
        if self.fit_settings['end_index'] is not None:
            self.dataframe = self.dataframe.iloc[:self.fit_settings['end_index']]

        v_w = self.dataframe['v_100m [m/s]'].to_numpy()
        F_RO = self.dataframe['F_RO [N]'].to_numpy()
        F_RI = self.dataframe['F_RI [N]'].to_numpy()
        theta_avg_RO = self.dataframe['theta_avg_RO [rad]'].to_numpy()
        theta_rel_RO = self.dataframe['theta_rel_RO [rad]'].to_numpy()
        phi_max_RO = self.dataframe['phi_max_RO [rad]'].to_numpy()
        stroke_tether = self.dataframe['stroke_tether [m]'].to_numpy()
        min_tether_length = self.dataframe['min_length_tether [m]'].to_numpy()       

        idx = np.argmax(F_RO) + self.fit_settings['index_offset']['F_RO']
        F_RO_quad = np.polyval(np.polyfit(v_w[:idx], F_RO[:idx], self.fit_settings['fit_order']['F_RO']), v_w[:idx])
        F_RO_smooth = np.hstack((F_RO_quad, F_RO[idx:]))
        
        idx = np.argmax(F_RI >= np.min(F_RI) + self.fit_settings['ineq_tols']['F_RI'])
        F_RI_quad = np.polyval(np.polyfit(v_w[idx:], F_RI[idx:], self.fit_settings['fit_order']['F_RI']), v_w[idx:])
        F_RI_smooth = np.hstack((F_RI[:idx], F_RI_quad))
        
        idx = np.argmax(theta_avg_RO >= np.min(theta_avg_RO) +\
                        self.fit_settings['ineq_tols']['average_elevation']) +\
                        self.fit_settings['index_offset']['average_elevation']
        theta_avg_RO_quad = np.polyval(np.polyfit(v_w[idx:], theta_avg_RO[idx:], self.fit_settings['fit_order']['average_elevation']), v_w[idx:])
        theta_avg_RO_cons = theta_avg_RO[0]*np.ones_like(theta_avg_RO[:idx])
        theta_avg_RO_smooth = np.hstack((theta_avg_RO_cons, theta_avg_RO_quad))
      
        idx = np.argmin(min_tether_length) + self.fit_settings['index_offset']['min_tether_length']
        min_tether_length_quad = np.polyval(np.polyfit(v_w[idx:], min_tether_length[idx:], self.fit_settings['fit_order']['min_tether_length']), v_w[idx:])
        min_tether_length_cons = np.min(min_tether_length)*np.ones_like(min_tether_length[:idx])
        min_tether_length_smooth = np.hstack((min_tether_length_cons, min_tether_length_quad))

        total_tether_length = parse_system_properties_and_bounds(self.config)['total_tether_length']
        stroke_tether_smooth = total_tether_length - min_tether_length_smooth

        self.dataframe['F_RO fit [N]'] = F_RO_smooth
        self.dataframe['F_RI fit [N]'] = F_RI_smooth
        self.dataframe['theta_avg_RO fit [rad]'] = theta_avg_RO_smooth
        self.dataframe['theta_rel_RO fit [rad]'] = self.dataframe['theta_rel_RO [rad]']
        self.dataframe['phi_max_RO fit [rad]'] = self.dataframe['phi_max_RO [rad]']
        self.dataframe['stroke_tether fit [m]'] = stroke_tether_smooth
        self.dataframe['min_length_tether fit [m]'] = min_tether_length_smooth

        if self.plot_smoothing_results:
            _, axs = plt.subplots(nrows = 4, ncols= 2, squeeze=True, sharex=True)
            axs = axs.flatten()
            axs[0].scatter(v_w, F_RO, zorder=1, label='Original', marker ='x', c='gray')
            axs[0].plot(v_w, F_RO_smooth, zorder=2, label='Fitted')

            axs[1].scatter(v_w, F_RI, zorder=1, label='Original', marker ='x', c='gray')
            axs[1].plot(v_w, F_RI_smooth, zorder=2, label='Fitted')

            axs[2].scatter(v_w, theta_avg_RO, zorder=1, label='Original', marker ='x', c='gray')
            axs[2].plot(v_w, theta_avg_RO_smooth, zorder=2, label='Fitted')

            axs[3].scatter(v_w, theta_rel_RO, zorder=1, label='Original', marker ='x', c='gray')
            
            axs[4].scatter(v_w, phi_max_RO, zorder=1, label='Original', marker ='x', c='gray')

            axs[5].scatter(v_w, stroke_tether, zorder=1, label='Original', marker ='x', c='gray')
            axs[5].plot(v_w, stroke_tether_smooth, zorder=2, label='Fitted')

            
            axs[6].scatter(v_w, min_tether_length, zorder=1, label='Original', marker ='x', c='gray')
            axs[6].plot(v_w, min_tether_length_smooth, zorder=2, label='Fitted')

            axs[7].set_axis_off()
        

            axs[0].set_ylabel('Reel-out force [N]')
            axs[1].set_ylabel('Reel-in force [N]')
            axs[2].set_ylabel('Avg. elevation angle [rad]')
            axs[3].set_ylabel('Rel. elevation angle [rad]')
            axs[4].set_ylabel('Max. azimuth angle [rad]')
            axs[5].set_ylabel('Tether stroke [m]')
            axs[6].set_ylabel('Min. tether length [m]')
            

            axs[-2].set_xlabel('Wind speed at 100 m [m/s]')
            axs[-1].set_xlabel('Wind speed at 100 m [m/s]')

            axs[0].legend()   
            plt.tight_layout()  
            plt.show()

    def smooth_power_curve(self):      
        # Parse system properties and bounds
        sys_props = parse_system_properties_and_bounds(self.config) # type: ignore
        sys_props = SystemProperties(sys_props) # type: ignore
        # Parse sim settings
        control_mode, time_step_RO, time_step_RO, time_step_RIRO = parse_sim_settings(self.config) # type: ignore
        otp_var_enabled_idx, x = parse_opt_variables(self.config) # type: ignore
        cons_enabled_idx, cons_param_vals = parse_constraints(self.config) # type: ignore

        profile, roughness_length, ref_height, _ = parse_environment(self.config) # type: ignore

        # Cycle simulation settings for different phases of the power curves.
        cycle_sim_settings_pc = {
            'cycle': {
                'traction_phase': TractionPhasePattern, # type: ignore
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

        # Pre-self.configure environment object for optimizations by setting normalized wind profile.
        if profile == 'logarithmic':
            env = LogProfile() # type: ignore
            env.set_reference_height(ref_height)
            env.set_roughness_length(roughness_length)
        else:
            NotImplementedError('Only logarithmic profiles are supported at the moment!')

        power_results = []
        v_w = self.dataframe['v_100m [m/s]'].to_numpy()

        for i in range(len(self.dataframe['F_RO fit [N]'])):
            env.set_reference_wind_speed(v_w[i])
            op_cycle_pc = OptimizerCycle(cycle_sim_settings_pc, sys_props, env, otp_var_enabled_idx, # type: ignore
                                        cons_enabled_idx, cons_param_vals, force_or_speed_control=control_mode) 
            x[0] = self.dataframe['F_RO fit [N]'].iloc[i]
            x[1] = self.dataframe['F_RI fit [N]'].iloc[i]
            x[2] = self.dataframe['theta_avg_RO fit [rad]'].iloc[i]
            x[3] = self.dataframe['theta_rel_RO fit [rad]'].iloc[i]
            x[4] = self.dataframe['phi_max_RO fit [rad]'].iloc[i]
            x[5] = self.dataframe['stroke_tether fit [m]'].iloc[i]
            x[6] = self.dataframe['min_length_tether fit [m]'].iloc[i]
            
            res = op_cycle_pc.eval_performance_indicators(plot_result = False, x_real_scale = x)

            power_results.append(res['average_power']['cycle'])

        self.dataframe['P_cycle fit [W]'] = power_results

        if self.plot_smoothing_results:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4), sharex=True)
            ax.plot(self.dataframe['v_100m [m/s]'], self.dataframe['P_cycle [W]'] * 1e-3, linewidth = 2, c='#11ad3a')
            ax.plot(self.dataframe['v_100m [m/s]'], self.dataframe['P_cycle fit [W]'] * 1e-3, linewidth = 2, c='#2373cd', linestyle='--')
            ax.set_ylabel(r'$P_{mech}$ [kW]')
            ax.set_xlabel('Windspeed at 100 m [m/s]')
            ax.legend(['Original', 'Fitted'])
            plt.grid()
            plt.tight_layout()
            plt.show()
    
    def calculate(self):
        if self.smooth:
            self.smooth_opt_results()
            self.smooth_power_curve()

            power_series = self.eta_flight_trajectory*self.dataframe['P_cycle fit [W]']
        else:
            power_series = self.eta_flight_trajectory*self.dataframe['P_cycle [W]']

        self.dataframe['P_DC [W]'] = self.eta_mot_control*power_series - self.self_cons_mot_control
        self.dataframe['P_AC [W]'] = self.eta_DC_AC*self.dataframe['P_DC [W]'] - self.self_cons_DC_AC

        if self.external_battery_connected:
            self.dataframe['P_ext_batt [W]'] = self.eta_ext_batt*self.dataframe['P_AC [W]'] - self.self_cons_ext_batt

    
    def plot_power_curves(self): 
        colors = ['#11ad3a', '#2373cd', '#f9b80d', '#c51f05']
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4), sharex=True)
        if self.smooth:
            ax.plot(self.dataframe['v_100m [m/s]'], self.eta_flight_trajectory*self.dataframe['P_cycle fit [W]'] * 1e-3, linewidth = 3,
                     c=colors[0], label= 'Mechanical power curve (fitted)')
        else:
            ax.plot(self.dataframe['v_100m [m/s]'], self.eta_flight_trajectory*self.dataframe['P_cycle [W]'] * 1e-3, linewidth = 3,
                     c=colors[0], label = 'Mechanical power curve (original)'), 

        ax.plot(self.dataframe['v_100m [m/s]'], self.dataframe['P_DC [W]'] * 1e-3, linewidth = 3,
                 c=colors[1], label = 'DC electrical power curve')

        ax.plot(self.dataframe['v_100m [m/s]'], self.dataframe['P_AC [W]'] * 1e-3, linewidth = 3,
                 c=colors[2], label = 'AC electrical power curve')
        
        if self.external_battery_connected:
            ax.plot(self.dataframe['v_100m [m/s]'], self.dataframe['P_ext_batt [W]'] * 1e-3,
                     linewidth = 3, c=colors[3], label = 'External battery power curve')

        ax.set_ylabel(r'$P$ [kW]')
        ax.set_xlabel('Windspeed at 100 m [m/s]')
        ax.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

class EnergyProductionEstimator:     
    def __init__(self, grib_filename, pkl_filename):
        self.grib_filename = grib_filename
        self.pkl_filename = pkl_filename
        self.dataframe = None
        self.flights_dataframe = None
        self.project = 'Mobilis'
        self.start_month = 'September'
        self.end_month = 'December'
    
        self.power_curve = None

        self.min_flight_time = 2.0
        self.max_flight_time = 6.0
        self.launch_min = 11
        self.landing_max = 15 

        self.weekdays_to_exclude = ['Saturday', 'Sunday']
        self.packing_energy = 14
        self.average_packing_time = 2

        self.maximum_azimuth = 45
        self.axis_no_fly_zone_deg = [20]
        self.ranges_no_fly_zone_deg = [160]   

    def to_dataframe_from_grib(self):   

        # Open the GRIB file with cfgrib engine
        ds = xr.open_dataset(self.grib_filename, engine='cfgrib', decode_timedelta=True)

        # Assume variables are 'u100' and 'v100' or similar — adjust if different
        # Extract only 100m wind components
        wind_vars = [var for var in ds.data_vars if var in ['u100', 'v100', '100u', '100v']]

        if not wind_vars:
            raise ValueError("No 100m wind variables found in dataset")

        # Select just the wind variables
        ds_wind = ds[wind_vars]

        # Get time coordinate as pandas datetime index
        time_index = pd.to_datetime(ds_wind['time'].values)

        # Build a DataFrame from the data variables and time components
        data = {
            'year': time_index.year,
            'month': time_index.month,
            'day': time_index.day,
            'hour': time_index.hour,
        }

        for var in wind_vars:
            var_data = ds_wind[var].squeeze()
            data[var] = var_data.values

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Sort as in your original code
        self.dataframe = df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)

    def save_dataframe_to_pickle(self):
        self.dataframe.to_pickle(self.pkl_filename)         

    def load_dataframe_from_pickle(self):
        if os.path.isfile(self.pkl_filename):
            self.dataframe = pd.read_pickle(self.pkl_filename)

    def reorder_to_project_years(self):
        month_name_to_num = {month: index for index, month in enumerate(calendar.month_name) if month}
        start_month_num = month_name_to_num[self.start_month]
        end_month_num = month_name_to_num[self.end_month]

        df = self.dataframe.copy()
        year_list = sorted(df['year'].unique())
        df['project_year'] = np.nan

        # Check that all required months are present
        required_months = (
            list(range(start_month_num, 13)) + list(range(1, end_month_num + 1))
            if start_month_num > end_month_num
            else list(range(start_month_num, end_month_num + 1))
        )

        dataset_months = sorted(df['month'].unique())

        # Check that every required month exists in the dataset
        missing_months = [m for m in required_months if m not in dataset_months]
        if missing_months:
            missing_month_names = [calendar.month_name[m] for m in missing_months]
            raise ValueError(f"Dataset is missing required months for project period: {', '.join(missing_month_names)}")

        # Project year assignment logic (unchanged)
        project_year = 1
        if start_month_num > end_month_num:
            for i in range(len(year_list) - 1):
                current_year = year_list[i]
                next_year = year_list[i + 1]

                mask_start = (df['year'] == current_year) & (df['month'] >= start_month_num)
                df.loc[mask_start, 'project_year'] = project_year

                mask_end = (df['year'] == next_year) & (df['month'] <= end_month_num)
                df.loc[mask_end, 'project_year'] = project_year

                project_year += 1
        else:
            for year in year_list:
                mask = (
                    (df['year'] == year) &
                    (df['month'] >= start_month_num) &
                    (df['month'] <= end_month_num)
                )
                df.loc[mask, 'project_year'] = project_year
                if mask.any():
                    project_year += 1

        df = df.dropna(subset=['project_year']).copy()
        df['project_year'] = df['project_year'].astype(int)
        self.dataframe = df.reset_index(drop=True)
                
    def uv_to_directionmodule(self):
        # Calculate direction and module
        self.dataframe['v_w_100'] = self.dataframe.apply(lambda row: np.sqrt(row['100v']**2 + row['100u']**2), axis=1)
        self.dataframe['dir (from)'] = self.dataframe.apply(
            lambda row: (270 - np.rad2deg(np.arctan2(row['100v'], row['100u']))) % 360,
            axis=1
        )
        self.dataframe.drop(columns=["100u", "100v"], inplace=True)

    def format_date(self):
        self.dataframe["date"] = pd.to_datetime(self.dataframe[["year", "month", "day", "hour"]], yearfirst=True)
        cols = ["date"] + [col for col in self.dataframe.columns if col != "date"]
        self.dataframe = self.dataframe[cols]

    def update_height_wind_speed(self, new_z=200, z_0=0.05, z_ref=100):
        def log_wind_speed(z, v_w_ref, z_0=0.05, z_ref=100): 
            v_w = v_w_ref*np.log(z/z_0)/np.log(z_ref/z_0)
            return v_w 
               
        self.dataframe['v_w_100'] = self.dataframe.apply(lambda row: log_wind_speed(new_z, row['v_w_100']), axis=1)
        self.dataframe.rename(columns={"v_w_100": 'v_w_' + str(new_z)}, inplace=True)

    def add_power_column(self):         
        v_w_power = self.power_curve.dataframe['v_100m [m/s]'].to_numpy()
        if self.power_curve.external_battery_connected:
            print('Interpolating external battery power curve...')
            power_series = self.power_curve.dataframe['P_ext_batt [W]'].to_numpy()/1000
        else:
            print('Interpolating AC infeed power curve...')
            power_series = self.power_curve.dataframe['P_AC [W]'].to_numpy()/1000

        # Interpolate directly using numpy vectorized ops
        wind_speeds = self.dataframe['v_w_100'].to_numpy()

        power_output = np.interp(wind_speeds, v_w_power, power_series, left=0.0, right=0.0)

        self.dataframe['unfiltered_power_production'] = power_output
        
    def set_flight_windows(self, min_flight_time, max_flight_time, launch_min, landing_max):
        self.min_flight_time = min_flight_time
        self.max_flight_time = max_flight_time
        self.launch_min = launch_min
        self.landing_max = landing_max 

    def filter_flight_window(self):
        print('Filtering flight window...')
        self.dataframe['power_in_window'] = self.dataframe['unfiltered_power_production'].where(
        (self.dataframe['hour'] >= self.launch_min) & (self.dataframe['hour'] <= self.landing_max)
        )

    def set_weekdays_to_exclude(self, weekdays_to_exclude = ['Saturday', 'Sunday']):
        valid_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if any(day not in valid_weekdays for day in weekdays_to_exclude):
            raise SyntaxError('Check your input: invalid weekday(s) to exclude!')
        
        else: self.weekdays_to_exclude = weekdays_to_exclude

    def exclude_weekdays(self):
        valid_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Handle None, empty list, or 'None' string
        if not self.weekdays_to_exclude or self.weekdays_to_exclude == ['None']:
            self.dataframe['excluding_weekdays'] = self.dataframe['unfiltered_power_production']
        
        # Validate input weekday names
        elif any(day not in valid_weekdays for day in self.weekdays_to_exclude):
            raise SyntaxError('Check your input: invalid weekday(s) to exclude!')
        
        # Apply filter
        else:
            self.dataframe['excluding_weekdays'] = self.dataframe['power_in_window'].where(
                ~self.dataframe['date'].dt.day_name().isin(self.weekdays_to_exclude)
            )

    def find_min_flight_production(self): 
        self.dataframe['min_flight_production'] = self.dataframe['excluding_weekdays'].where(
            self.dataframe['excluding_weekdays'] >= 0)    
    
    def exclude_no_fly_zones(self):
        def calculate_no_fly_wind_window():
            bounds1 = []
            bounds2 = []
            for axis, range_no_fly in zip(self.axis_no_fly_zone_deg, self.ranges_no_fly_zone_deg):
                half_cone_width = range_no_fly / 2 + self.maximum_azimuth
                bound1 = (axis - half_cone_width) % 360
                bound2 = (axis + half_cone_width) % 360
                bounds1.append(bound1)
                bounds2.append(bound2)
            return bounds1, bounds2

        if not self.axis_no_fly_zone_deg:
            self.dataframe['excluding_directions'] = self.dataframe['min_flight_production']
        else:
            bounds1, bounds2 = calculate_no_fly_wind_window()

            # Start with a mask of all False
            cond = pd.Series(False, index=self.dataframe.index)

            # Combine masks for each no-fly zone
            for b1, b2 in zip(bounds1, bounds2):
                if b1 <= b2:
                    # No wraparound
                    zone_mask = (self.dataframe['dir (from)'] >= b1) & (self.dataframe['dir (from)'] <= b2)
                else:
                    # Wraparound case
                    zone_mask = (self.dataframe['dir (from)'] >= b1) | (self.dataframe['dir (from)'] <= b2)

                # Combine with existing conditions
                cond = cond | zone_mask

            # Apply mask: only keep values **outside** the no-fly zones
            self.dataframe['excluding_directions'] = self.dataframe['min_flight_production'].where(~cond)            

    def find_consecutive_flight_hours(self):
        flight_mask = ~self.dataframe['excluding_directions'].isna()
        self.dataframe['cons_flight_hrs'] = (
            flight_mask
            .astype(int)
            .groupby((~flight_mask).cumsum())
            .cumsum()
        )

    def calculate_energy_for_consecutive_hours(self):
        df = self.dataframe
        
        # Initialize a new column with NaNs
        df['flight_energy'] = np.nan
        
        # We only calculate sum if consecutive hours >= minimum threshold
        min_hours = self.min_flight_time
        max_hours = self.max_flight_time
        
        for i in range(len(df)):
            cons_hours = df.loc[i, 'cons_flight_hrs']
            if cons_hours >= min_hours:
                hours_to_sum = min(cons_hours, max_hours)
                start_idx = max(i - hours_to_sum + 1, 0)
                energy_sum = df.loc[start_idx:i, 'excluding_directions'].sum()
                df.loc[i, 'flight_energy'] = energy_sum

    def find_total_daily_energy(self):
        df = self.dataframe.copy()
        
        # Extract date only (without time)
        df['date_only'] = df['date'].dt.date
        
        # Initialize new column with NaNs
        df['max_flight_energy_at_0'] = np.nan
        
        # Group by date
        for date, group in df.groupby('date_only'):
            # Get max flight energy for the whole day
            max_energy = group['flight_energy'].max()
            
            # Find index(es) where hour == 0
            zero_hour_idx = group.index[group['hour'] == 0]
            
            # Assign max energy at hour 0
            df.loc[zero_hour_idx, 'daily_energy'] = max_energy
        
        # Assign back to original dataframe
        self.dataframe['daily_energy'] = df['daily_energy']

    def subtract_packing_losses(self):
        self.dataframe['include_packing_losses'] = (self.dataframe['daily_energy'] - self.packing_energy).where(self.dataframe['daily_energy'] - self.packing_energy >= 0)

    def calculate_packing_energy(self):        
        total_self_consumption = self.power_curve.self_cons_mot_control +\
                                    self.power_curve.self_cons_DC_AC

        if self.power_curve.external_battery_connected:
            total_self_consumption += self.power_curve.self_cons_ext_batt 

        self.packing_energy = total_self_consumption*self.average_packing_time

    def calculate_avg_speed(self):
        df = self.dataframe.copy()
        df['date_only'] = df['date'].dt.date
        df['avg_speed'] = np.nan

        for date, group in df.groupby('date_only'):
            # Get the value of daily_energy from hour 0 row
            day_start_row = group[group['hour'] == 0]
            if day_start_row.empty:
                continue

            daily_energy_val = day_start_row['daily_energy'].values[0]

            # Find the row where flight_energy == daily_energy
            match_row = group[group['flight_energy'] == daily_energy_val]
            if match_row.empty:
                continue

            match_idx = match_row.index[0]
            cons_hrs = df.loc[match_idx, 'cons_flight_hrs']

            # Determine the start and end indices of the flight block
            start_idx = match_idx - cons_hrs + 1
            end_idx = match_idx  

            # Make sure we don’t go out of bounds
            if start_idx < group.index.min():
                continue

            # Get v_w_200 values over that block
            speed_vals = df.loc[start_idx:end_idx, 'v_w_100']
            avg_speed = speed_vals.mean()

            # Assign it only at hour 0 row
            df.at[day_start_row.index[0], 'avg_speed'] = avg_speed

        # Save to the main DataFrame
        self.dataframe['avg_speed'] = df['avg_speed']

    def run_energy_pipeline(self):
        """
        Run the full energy estimation pipeline from flight filtering
        to daily energy and average wind speed calculations.
        """
        self.reorder_to_project_years()
        self.format_date()
        self.uv_to_directionmodule()
        #self.update_height_wind_speed()
        self.add_power_column()
        self.filter_flight_window()
        self.exclude_weekdays()
        self.find_min_flight_production()
        self.exclude_no_fly_zones()
        self.find_consecutive_flight_hours()
        self.calculate_energy_for_consecutive_hours()
        self.find_total_daily_energy()
        self.subtract_packing_losses()
        self.calculate_avg_speed()

    def calculate_average_weekly_production(self):
        month_name_to_num = {month: index for index, month in enumerate(calendar.month_name) if month}
        start_month_num = month_name_to_num[self.start_month]
        end_month_num = month_name_to_num[self.end_month]

        year_list = sorted(self.dataframe['year'].unique())

        n_years_in_project = year_list[-1] - year_list[0] + 1

        n_months_in_project = end_month_num - start_month_num + 1
        total_energy = self.dataframe['include_packing_losses'].sum()

        print(total_energy/(52/12*n_months_in_project*n_years_in_project))      

    def plot_figures(self, save_figs = True):
        def group_by_month(df, nmonth):
            filt_df = df.loc[df['month'] == nmonth]
            return filt_df

        def group_by_project_year(df, year):
            filt_df = df.loc[df['project_year'] == year]
            return filt_df

        def plot_weekly_production(ax, df, nmonth, year, year_label, column = 'include_packing_losses'):
            df = group_by_month(df, nmonth)
            df = group_by_project_year(df, year)

            ax.bar(df['day'], df[column], color='skyblue', width=0.8, edgecolor='black')

            # Alternate week background colors
            min_date = df['day'].min()-0.5
            max_date = df['day'].max()+0.5
            if max_date <= 29: max_date = 31 + 0.5 
            current = min_date

            week = 7
            toggle = True
            week_centers = []

            while current < max_date:
                next_week = current + week
                ax.axvspan(current, next_week, facecolor='lightgrey' if toggle else 'white', alpha=0.3)
                week_centers.append(current+3.5)
                toggle = not toggle
                current = next_week
        
            ax.set_xticks(week_centers)
            ax.set_xticklabels('')
            ax.tick_params(axis='x',length= 0)
            ax.set_xlim(min_date, max_date)
    
            # Labels and formatting
            ax.set_ylabel('Energy (kWh)')
            ax.set_title('Production for ' + calendar.month_abbr[nmonth] + ' ' + year_label, fontsize = 10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        def calculate_monthly_production(df, nmonth, year, column = 'include_packing_losses'):
            df = group_by_month(df, nmonth)
            df = group_by_project_year(df, year)
            energy_in_month = df[column].sum()
            return energy_in_month

        def plot_monthly_production_by_year(ax, df, nmonths_list, year, year_label):
            energy_in_nmonths_list = []
            for nmonth in nmonths_list:
                energy_in_month = calculate_monthly_production(df, nmonth, year)
                energy_in_nmonths_list.append(energy_in_month)

            ax.bar(range(1,len(nmonths_list)+1), energy_in_nmonths_list, color='skyblue', width=0.8, edgecolor='black')

            # Alternate month/year background colors
            minx = 1-0.5
            maxx = len(nmonths_list)+0.5
            current = minx

            interval = 1
            toggle = True
            centers = []

            while current < maxx:
                nextx = current + interval
                ax.axvspan(current, nextx, facecolor='lightgrey' if toggle else 'white', alpha=0.3)
                centers.append(current+interval/2)
                toggle = not toggle
                current = nextx

            if len(nmonths_list) <= 3: x_labels = [calendar.month_name[nmonth] for nmonth in nmonths_list]
            else: x_labels = [calendar.month_abbr[nmonth] for nmonth in nmonths_list]
            ax.set_xticks(centers)
            ax.tick_params(axis='x',length= 0)
            ax.set_xticklabels(x_labels, rotation=0)
            ax.set_xlim(minx, maxx)

            ylb, yup = ax.get_ylim()
            ax.set_yticks(np.arange(ylb, yup, 150))

            # Labels and formatting
            ax.set_ylabel('Energy (kWh)')
            ax.set_title('Production in ' + year_label, fontsize = 10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        def plot_monthly_production_by_month(ax, df, nmonth, years_list, years_labels):
            energy_in_years_list = []
            for year in years_list:
                energy_in_month = calculate_monthly_production(df, nmonth, year)
                energy_in_years_list.append(energy_in_month)


            # Normalize energy values to [0, 1]
            norm = Normalize(vmin=min(energy_in_years_list),
                                vmax=max(energy_in_years_list))

            # Get color from the 'Blues' colormap
            colors = [cm.Blues(norm(val)) for val in energy_in_years_list]

            ax.bar(range(1,len(years_list)+1), energy_in_years_list, color='skyblue', width=0.8, edgecolor='black')

            # Alternate month/year background colors
            minx = 1-0.5
            maxx = len(years_list)+0.5
            current = minx

            interval = 1
            toggle = True
            centers = []

            while current < maxx:
                nextx = current + interval
                ax.axvspan(current, nextx, facecolor='lightgrey' if toggle else 'white', alpha=0.3)
                centers.append(current+interval/2)
                toggle = not toggle
                current = nextx

            x_labels = [label for label in years_labels]
            ax.set_xticks(centers)
            ax.tick_params(axis='x',length= 0)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_xlim(minx, maxx)

            ylb, yup = ax.get_ylim()
            ax.set_yticks(np.arange(ylb, yup, 150))
            # Labels and formatting
            ax.set_ylabel('Energy (kWh)')
            ax.set_title('Production in ' + calendar.month_name[nmonth], fontsize = 10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        def calculate_nflights_in_month(df, nmonth, year, threshold, column = 'include_packing_losses'):
            df = group_by_month(df, nmonth)
            df = group_by_project_year(df, year)
            mask = df[column] >= threshold
            return sum(mask)

        def plot_flight_stats_by_year(ax, df, nmonths_list, year, year_label, threshold=1, column = 'include_packing_losses'):
            flights_in_nmonths_list = []
            for nmonth in nmonths_list:
                n_flights = calculate_nflights_in_month(df, nmonth, year, threshold, column)
                flights_in_nmonths_list.append(n_flights)

            # Normalize energy values to [0, 1]
            norm = Normalize(vmin=min(flights_in_nmonths_list),
                                vmax=max(flights_in_nmonths_list))

            # Get color from the 'Blues' colormap
            colors = [cm.Blues(norm(val)) for val in flights_in_nmonths_list]

            ax.bar(range(1,len(nmonths_list)+1), flights_in_nmonths_list, color='skyblue', width=0.8, edgecolor='black')

            # Alternate month/year background colors
            minx = 1-0.5
            maxx = len(nmonths_list)+0.5
            current = minx

            interval = 1
            toggle = True
            centers = []

            while current < maxx:
                nextx = current + interval
                ax.axvspan(current, nextx, facecolor='lightgrey' if toggle else 'white', alpha=0.3)
                centers.append(current+interval/2)
                toggle = not toggle
                current = nextx

            if len(nmonths_list) <= 3: x_labels = [calendar.month_name[nmonth] for nmonth in nmonths_list]
            else: x_labels = [calendar.month_abbr[nmonth] for nmonth in nmonths_list]
            ax.set_xticks(centers)
            ax.tick_params(axis='x',length= 0)
            ax.set_xticklabels(x_labels, rotation=0)
            ax.set_xlim(minx, maxx)

            # Labels and formatting
            ax.set_ylabel(r'N. of flights (En.$\geq$' + str(threshold) + ' (kWh))')
            ax.set_title('Flights in ' + year_label, fontsize = 10) 
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        def plot_flight_stats_by_month(ax, df, nmonth, years_list, years_labels, threshold=1, column = 'include_packing_losses'):
            flights_in_years_list = []
            for year in years_list:
                flights_in_year = calculate_monthly_production(df, nmonth, year, column)
                flights_in_years_list.append(flights_in_year)

            # Normalize energy values to [0, 1]
            norm = Normalize(vmin=min(flights_in_years_list),
                                vmax=max(flights_in_years_list))
            flights_in_years_list = []
            for year in years_list:
                n_flights = calculate_nflights_in_month(df, nmonth, year, threshold)
                flights_in_years_list.append(n_flights)

            ax.bar(range(1,len(years_list)+1), flights_in_years_list, color='skyblue', width=0.8, edgecolor='black')

            # Alternate month/year background colors
            minx = 1-0.5
            maxx = len(years_list)+0.5
            current = minx

            interval = 1
            toggle = True
            centers = []

            while current < maxx:
                nextx = current + interval
                ax.axvspan(current, nextx, facecolor='lightgrey' if toggle else 'white', alpha=0.3)
                centers.append(current+interval/2)
                toggle = not toggle
                current = nextx

            x_labels = [label for label in years_labels]
            ax.set_xticks(centers)
            ax.tick_params(axis='x',length= 0)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_xlim(minx, maxx)

            # Labels and formatting
            ax.set_ylabel(r'N. of flights (En.$\geq$' + str(threshold) + ' (kWh))')
            ax.set_title('Flights in ' + calendar.month_name[nmonth], fontsize = 10) 
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        def generate_project_year_labels(year_list, start_month, end_month, month_name_to_num):
            labels = []
            for year in year_list:
                start_num = month_name_to_num[start_month]
                end_num = month_name_to_num[end_month]
                
                if start_num > end_num:
                    # Project spans two years (e.g., Sep–Feb)
                    label = f"{year}-{year + 1}"
                else:
                    # Project is within a single calendar year
                    label = f"{year}"
                
                labels.append(label)
            if start_num > end_num: labels = labels[0:-1:1]    
            return labels

        def plot_wind_speed_distribution(ax, df, month, column='v_w_100', bin_width=0.5):
            """
            Plots the probability distribution of wind speed in specified bins.
            
            Parameters:
                df (pd.DataFrame): Input DataFrame containing wind speed data.
                column (str): Column name for wind speed.
                bin_width (float): Width of each bin in m/s.
            """
            if month != 'all':
                data = df.loc[df['month'] == month] 
            
            else: data = df.copy(deep=True)
            # Drop NaNs
            data = data[column].dropna()

            # Define bin edges
            min_speed = data.min()
            max_speed = data.max()
            bins = np.arange(np.floor(min_speed), np.ceil(max_speed) + bin_width, bin_width)

            # Plot histogram
            ax.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.5)
            ax.set_xlim(-0.2, 20.2)
            ax.set_ylim(0, 0.19)

            # Format the plot
            ax.set_title(calendar.month_name[month],  fontsize = 10)
            ax.set_xlabel('Wind speed at 200 m (m/s)')
            ax.set_ylabel('Probability density')

        def plot_wind_rose(ax, df, month):    
            def speed_labels(bins, units):   
                labels = []
                for left, right in zip(bins[:-1], bins[1:]):
                    if left == bins[0]:
                        labels.append('calm')
                    elif np.isinf(right):
                        labels.append('>{} {}'.format(left, units))
                    else:
                        labels.append('{} - {} {}'.format(left, right, units))
                return labels

            def _convert_dir(directions, N=None):
                if N is None:
                    N = directions.shape[0]
                barDir = directions * np.pi / 180. - np.pi / N
                barWidth = 2 * np.pi / N
                return barDir, barWidth

            def wind_rose(rosedata, wind_dirs, ax, palette=None):
                if palette is None:
                    n_colors = rosedata.shape[1]
                    palette = [cm.Blues(i / (n_colors - 1)) for i in range(n_colors)]

                bar_dir, bar_width = _convert_dir(wind_dirs)

                ax.set_theta_direction('clockwise')
                ax.set_theta_zero_location('N')

                for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
                    if n == 0:
                        ax.bar(bar_dir, rosedata[c1].values, 
                            width=bar_width,
                            color=palette[0],
                            edgecolor='black',
                            label=c1,
                            linewidth=0)
                    ax.bar(bar_dir, rosedata[c2].values, 
                        width=bar_width, 
                        bottom=rosedata.cumsum(axis=1)[c1].values,
                        color=palette[n+1],
                        edgecolor='black',
                        label=c2,
                        linewidth=0)

                ax.legend(ncol=10)

                # Set direction labels
                directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                angles = np.deg2rad(np.linspace(0, 360, len(directions), endpoint=False))
                ax.set_xticks(angles)
                ax.set_xticklabels(directions)

                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

                ylb, yup = ax.get_ylim()
                ax.set_ylim(ylb, 14)
                ax.set_yticks([5, 10])

            # Select data for the month
            wdf = df.loc[df['month'] == month] if month != 'all' else df.copy(deep=True)

            spd_bins = [-1, 0, 5, 10, 15, 20, np.inf]
            spd_labels = speed_labels(spd_bins, units='m/s')
            dir_bins = np.arange(-7.5, 370, 15)
            dir_labels = (dir_bins[:-1] + dir_bins[1:]) / 2

            total_count = wdf.shape[0]
            calm_count = wdf.query("v_w_100 == 0").shape[0]

            rose = (
                wdf.assign(WindSpd_bins=pd.cut(wdf['v_w_100'], bins=spd_bins, labels=spd_labels, right=True))
                .assign(WindDir_bins=pd.cut(wdf['dir (from)'], bins=dir_bins, labels=dir_labels, right=False))
                .replace({'WindDir_bins': {360: 0}})
                .groupby(by=['WindSpd_bins', 'WindDir_bins'])
                .size()
                .unstack(level='WindSpd_bins')
                .fillna(0)
                .assign(calm=calm_count / total_count)
                .sort_index(axis=1)
                .applymap(lambda x: x / total_count * 100)
            )

            directions = np.arange(0, 360, 15)
            wind_rose(rose, directions, ax=ax)

            return ax

        def plot_all_wind_speed_dist_roses(df, nmonths_list):
            fig = plt.figure(figsize=(20, 8), num='wind_stats')

            # Create a 2-row layout with unequal row heights (1:2 ratio)
            gs = GridSpec(2, len(nmonths_list), height_ratios=[1, 2], figure=fig)
            axes = [[None for _ in nmonths_list] for _ in range(2)]  # placeholder for axes references

            for i, nmonth in enumerate(nmonths_list):
                axes[0][i] = fig.add_subplot(gs[0, i])  # top row: regular axes
                plot_wind_speed_distribution(axes[0][i], df, nmonth)

            for i, nmonth in enumerate(nmonths_list):
                axes[1][i] = fig.add_subplot(gs[1, i], polar=True)  # bottom row: polar axes
                plot_wind_rose(axes[1][i], df, nmonth)
                if i != 0:
                    axes[0][i].set_ylabel('')
                if i != len(nmonths_list)-1:
                    axes[1][i].get_legend().remove()
                else:
                    leg = axes[1][i].get_legend()
                    leg.set_bbox_to_anchor((0.5, -0.08))

            plt.subplots_adjust(
                top=0.93,
                bottom=0.005,
                left=0.05,
                right=0.975,
                hspace=0.175,
                wspace=0.305
            )
        
            return fig
                

        month_name_to_num = {month: index for index, month in enumerate(calendar.month_name) if month}
        year_list = self.dataframe['year'].unique().tolist()
        project_year_list = self.dataframe['project_year'].unique().tolist()
        start_num = month_name_to_num[self.start_month]
        end_num = month_name_to_num[self.end_month]

        if start_num >= end_num:
                    # Wrap around the year (e.g., November to March)
            month_list = list(range(start_num, 13)) + list(range(1, end_num + 1)) 
        else:
                    # Simple range (e.g., March to August)
            month_list = list(range(start_num, end_num + 1))

        years_labels = generate_project_year_labels(year_list, self.start_month, self.end_month, month_name_to_num)

        fig1, axes = plt.subplots(nrows=len(project_year_list), ncols=len(month_list), figsize=(15, 8), sharey=True, num = 'daily_production')
        for i, year in enumerate(project_year_list):
            for j, month in enumerate(month_list):
                ax = axes[i, j]
                plot_weekly_production(ax, self.dataframe, month, year, years_labels[i])
                if j != 0: axes[i, j].set_ylabel('')
                if i != 0: axes[i, j].set_xlabel('')
                plt.tight_layout()

        fig2, axes = plt.subplots(nrows=1, ncols=len(project_year_list), figsize=(15, 3), sharey=True, num = 'months_by_years_production')
        for i, year in enumerate(project_year_list):
            plot_monthly_production_by_year(axes[i],self.dataframe, month_list, year, years_labels[i])
            plt.tight_layout()

        fig3, axes = plt.subplots(nrows=1, ncols=len(month_list), figsize=(15, 3), sharey=True, num = 'years_by_months_production')
        for i, month in enumerate(month_list):
            plot_monthly_production_by_month(axes[i],self.dataframe, month, project_year_list, years_labels)
            plt.tight_layout()

        fig4, axes = plt.subplots(nrows=1, ncols=len(project_year_list), figsize=(15, 3), sharey=True, num = 'flights_stats_by_year')
        for i, year in enumerate(project_year_list):
            plot_flight_stats_by_year(axes[i],self.dataframe, month_list, year, years_labels[i])
            plt.tight_layout()

        fig5, axes = plt.subplots(nrows=1, ncols=len(month_list), figsize=(15, 3), sharey=True, num = 'flights_stats_by_months')
        for i, month in enumerate(month_list):
            plot_flight_stats_by_month(axes[i],self.dataframe, month, project_year_list, years_labels)
            plt.tight_layout()

        fig6 = plot_all_wind_speed_dist_roses(self.dataframe, month_list)

        fig_list = [fig1, fig2, fig3, fig4, fig5, fig6]
        if save_figs:
            results_dir = self.project + '_' + self.start_month + '_' + self.end_month
            if not os.path.exists(results_dir): os.mkdir(results_dir)

            for fig in fig_list:
                fig.savefig(results_dir + os.sep + fig.get_label() + '.jpg', format = 'jpg')
                fig.savefig(results_dir + os.sep + fig.get_label() + '.svg', format = 'svg')


wind_data = EnergyProductionEstimator('wind_resource/data_mobilis_sepfeb.grib', 'wind_resource/data_mobilis_sepfeb.pkl')
#wind_data.to_dataframe_from_grib()
#wind_data.save_dataframe_to_pickle()
wind_data.load_dataframe_from_pickle()

wind_data.start_month = 'September'
wind_data.end_month = 'November'

pow_curve = ElectricalPowerCurve('Mobilis/baseline/power_curve_log_profile.csv', 'Mobilis/baseline/config.yaml')

pow_curve.plot_power_curves()

plt.show()
#pow_curve.dataframe.to_csv('output/alpha_power_curve.csv')

wind_data.power_curve = pow_curve

wind_data.packing_energy = 7

wind_data.set_flight_windows(2, 6, 11, 18)
wind_data.run_energy_pipeline()
wind_data.calculate_average_weekly_production()
#wind_data.dataframe.to_csv('predictions.csv')

#wind_data.plot_figures(save_figs=False)
#plt.show()
