import os
import pandas as pd
import numpy as np

from scipy import signal
from qsm import * 

def load_process_protologger_file(data_path, test_name):
    """
    Loads, filters, and processes a ProtoLogger_lidar.csv CSV file to extract flight cycle data.

    This function performs the following steps:
    1. Searches for a file ending with '_ProtoLogger_lidar.csv' within the given test directory.
    2. Loads the file into a pandas DataFrame.
    3. Filters and selects specific columns of interest.
    4. Groups data based on flight phases (1 to 4), combining them into complete flight cycles.
    
    A cycle is defined as a contiguous segment of rows where each of the four flight phases
    (index 1 through 4) occurs in sequence. Each cycle is returned as a pandas DataFrame.

    Args:
        data_path (str): Path to the base data directory.
        test_name (str): Name of the specific test folder inside the data directory.

    Returns:
        cycle_dataframe_list (List[pd.DataFrame]): A list of DataFrames, each representing a full flight cycle,
        consisting of data grouped from flight phases 1 to 4.

    Notes:
        - The function assumes the input CSV is space-delimited.
        - Rows with NaN values in 'flight_phase_index' are excluded before processing.
    """
    def load_protologger_file(data_path, test_name):
        test_path = os.path.join(data_path, test_name)

        protologger_file_path = None
        for file in os.listdir(test_path):
            if file.endswith('_ProtoLogger_lidar.csv'):
                protologger_file_path = os.path.join(test_path, file)
                break

        if protologger_file_path:
            df = pd.read_csv(protologger_file_path, delimiter=' ', low_memory=False)
            print(f"Loaded file: {protologger_file_path}")
            
        return df

    def select_columns(df, columns_to_extract):
        missing_cols = [col for col in columns_to_extract if col not in df.columns]
        if missing_cols:
            print(f"Some of the requested columns are missing: {missing_cols}")

        # Filter rows where flight_phase_index is not NaN
        df_filtered = df[df['flight_phase_index'].notna()]
        # Select only relevant columns
        df_selected = df_filtered[columns_to_extract]

        return df_selected

    def group_cycles(df):
        def group_phases(index_vector):
            group_ids = np.full(len(index_vector), -1)  # -1 = not part of any group
            current_group = 0
            in_group = False

            for i, val in enumerate(index_vector):
                if val == 1:
                    if not in_group:
                        in_group = True
                        current_group += 1
                    group_ids[i] = current_group
                else:
                    in_group = False  # End of current group
            return group_ids    
            
        def group_variable(var_vector, group_ids):
            grouped_var = []
            for gid in np.unique(group_ids):
                if gid > 0:
                    grouped_var.append(var_vector[group_ids == gid])
            return grouped_var

        results = [[], [], [], []]
        flight_idx = range(1, 5)
        for i in flight_idx:
            idx = df.flight_phase_index == i
            group_ids = group_phases(idx)

            grouped_var = group_variable(df.time.to_numpy(), group_ids)
            for n in range(0, len(grouped_var)): 
                results[i-1].append(pd.DataFrame(grouped_var[n], columns=[df.columns[0]]))    

            for var in df.columns[1:]:
                # group the variables
                grouped_var = group_variable(df[var].to_numpy(), group_ids)
                for n in range(0, len(grouped_var)):
                    # Append the grouped variable to the results
                    results[i-1][n][var] = pd.Series(grouped_var[n])

        cycle_dfs = []
        for n in range(len(results[0])): 
            cycle_dfs.append(pd.concat([results[0][n], results[1][n], results[2][n], results[3][n]], axis=0, ignore_index=True))
            cycle_dfs[n].reset_index(drop=True, inplace=True)
        
        return cycle_dfs
    
    print('Reading from .csv file...')
    protologger_file = load_protologger_file(data_path, test_name)
    # Columns you want to extract
    columns_to_extract = ['time', 'date',     
        'kite_0_pitch', 'kite_velocity_abs',
        'ground_tether_reelout_speed', 'ground_tether_length', 'ground_tether_force',
        'airspeed_angle_of_attack', 'ground_mech_power',
        'kite_actual_depower',
        'kite_pos_east', 'kite_pos_north', 'kite_height', 
        'kite_elevation', 'kite_azimuth', 'kite_distance', 'optimizer_step_index',
        'airspeed_apparent_windspeed', 'kite_estimated_va', 'kite_measured_va',
        'kite_heading', 'kite_course',
        'lift_coeff', 'drag_coeff',
        '100m Wind Speed (m/s)',
        'flight_phase', 'flight_phase_index']

    protologger_file = select_columns(protologger_file, columns_to_extract)
    cycle_dataframe_list = group_cycles(protologger_file)

    return cycle_dataframe_list

# Pattern analysis
def find_end_RO_and_max_tether_length(df=pd.DataFrame, threshold=8.):
    """
    Identifies the end of the reel-out (RO) phase based on a depower threshold,
    and returns the corresponding maximum tether length.

    This function computes the average `kite_actual_depower` during flight phase 1
    (typically representing the RO phase), then detects the first point where the
    depower exceeds this average by more than a specified threshold.
    This point is considered the end of the RO phase.

    Args:
        df (pd.DataFrame): DataFrame containing flight data, including
            'kite_actual_depower', 'flight_phase_index', and 'ground_tether_length'.
        threshold (float, optional): Depower delta used to determine end of RO.
            Defaults to 8.

    Returns:
        Tuple[float, int]: A tuple containing:
            - max_tether_length_RO (float): The tether length at the detected end of RO.
            - end_RO_idx (int): Index in the DataFrame where the end of RO occurs.
    """
    avg_ro_depwr = np.mean(df.kite_actual_depower[df.flight_phase_index==1])
    depwr_idx = (df.kite_actual_depower -  avg_ro_depwr) > threshold
    end_RO_idx = next((i for i, x in enumerate(depwr_idx) if x != 0), -1)
    max_tether_length_RO = df.ground_tether_length[end_RO_idx]
    return max_tether_length_RO, end_RO_idx

def find_start_RO(df, threshold=0.15):
    """
    Determines the start of the reel-out (RO) phase based on tether reel-out speed.

    This function identifies the first index in the DataFrame where the 
    `ground_tether_reelout_speed` exceeds or equals a specified threshold,
    indicating the beginning of the RO phase.

    Args:
        df (pd.DataFrame): DataFrame containing flight data, specifically the
            'ground_tether_reelout_speed' column.
        threshold (float, optional): Minimum speed to detect the start of reel-out.
            Defaults to 0.15.

    Returns:
        int: Index of the first row where reel-out starts (speed >= threshold),
             or -1 if no such point is found.
    """
    non_zero_speed_idx = df.ground_tether_reelout_speed >= threshold
    start_RO_idx = next((i for i, x in enumerate(non_zero_speed_idx) if x != 0), -1)
    return start_RO_idx

def find_RO_pattern_param(exp_RO_dataframe):
    """
    Extracts key pattern parameters from the reel-out (RO) phase flight data,
    specifically focusing on kite azimuth and elevation oscillation characteristics.

    The function identifies:
    - The average peak absolute azimuth value during RO (used to characterize azimuth tracking).
    - The relative elevation angle (half the peak-to-valley difference of kite elevation).
    - The average elevation angle (mean of peaks and valleys).

    A helper function `extract_complete_peaks` ensures only well-formed peak-valley cycles
    are used in elevation analysis.

    Args:
        exp_RO_dataframe (pd.DataFrame): DataFrame containing RO phase data.

    Returns:
        Tuple[float, float, float]:
            - max_az_trac (float): Mean of peak absolute azimuth values.
            - rel_el_angle (float): Half the difference between average elevation peaks and valleys.
            - avg_el_angle (float): Average of elevation peaks and valleys.
    """
    def extract_complete_peaks(sig):    
        peaks, _ = signal.find_peaks(sig, distance = 5)
        valleys, _ = signal.find_peaks(-sig, distance = 5)

        complete_peaks = []

        # Loop through valleys to find enclosed peaks
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            enclosed_peaks = [p for p in peaks if start < p < end]
            if enclosed_peaks:
                tallest = max(enclosed_peaks, key=lambda p: sig[p])
                complete_peaks.append(tallest)

        # Check for a final complete cycle after the last valley
        if len(valleys) >= 1:
            last_valley = valleys[-1]
            trailing_peaks = [p for p in peaks if p > last_valley]
            if trailing_peaks:
                tallest = max(trailing_peaks, key=lambda p: sig[p])
                complete_peaks.append(tallest)

        return complete_peaks

    peaks_idx, _ = signal.find_peaks(np.abs(exp_RO_dataframe.kite_azimuth), prominence=0.1, distance=10)    
    max_az_trac = np.mean(np.abs(exp_RO_dataframe.kite_azimuth)[peaks_idx]) 

    peaks_idx_el = extract_complete_peaks(exp_RO_dataframe.kite_elevation)
    valleys_idx_el = extract_complete_peaks(-exp_RO_dataframe.kite_elevation)

    if len(peaks_idx_el) == 0 or len(valleys_idx_el) == 0:
        peaks_idx_el, _ = signal.find_peaks(exp_RO_dataframe.kite_elevation,  distance=10)
        valleys_idx_el, _ = signal.find_peaks(exp_RO_dataframe.kite_elevation, distance=10)
        raise(ValueError)
    
    avg_el_peak = np.mean(exp_RO_dataframe.kite_elevation[peaks_idx_el])
    avg_el_valley = np.mean(exp_RO_dataframe.kite_elevation[valleys_idx_el])
    rel_el_angle = 0.5*(avg_el_peak - avg_el_valley) if not np.isnan(avg_el_peak) and not np.isnan(avg_el_valley) else np.nan
    avg_el_angle = 0.5*(avg_el_peak + avg_el_valley)

    return max_az_trac, rel_el_angle, avg_el_angle

def find_end_RI_and_min_tether_length(exp_cycle_dataframe, threshold=2):
    """
    Identifies the end of the reel-in (RI) phase in a flight cycle and returns the
    corresponding minimum tether length.

    This function analyzes the depower signal during flight phase 4 (typically RI),
    computes the average depower at the end of RI, and searches backwards in time
    to find when the depower exceeds this average by more than a specified threshold.
    This point is interpreted as the end of the RI phase.

    Args:
        exp_cycle_dataframe (pd.DataFrame): DataFrame containing a complete flight cycle.
            Must include 'kite_actual_depower', 'flight_phase_index', and 'ground_tether_length'.
        threshold (float, optional): Depower delta used to detect the end of RI.
            Defaults to 2.

    Returns:
        Tuple[float, int]:
            - min_tether_length_RI (float): Tether length at the detected end of RI.
            - end_RI_idx (int): Index in the DataFrame corresponding to the end of RI.
              Returns -1 if no such point is found.
    """
    avg_RIRO_depwr = np.mean(exp_cycle_dataframe.kite_actual_depower[exp_cycle_dataframe.flight_phase_index == 4].iloc[-5:-1])
    depwr_idx = (exp_cycle_dataframe.kite_actual_depower -  avg_RIRO_depwr) > threshold
    depwr_idx = depwr_idx.iloc[::-1]
    end_RI_idx = len(depwr_idx) - next((i for i, x in enumerate(depwr_idx) if x != 0), -1)
    min_tether_length_RI = exp_cycle_dataframe.ground_tether_length[end_RI_idx]
    return min_tether_length_RI, end_RI_idx

# Identify actual flight phases
def find_qsm_flight_phases(exp_cycle_dataframe):
    """
    Redefines flight phase labeling within a flight cycle to align with 
    quasi-steady modeling (QSM) assumptions.

    This function performs the following steps:
    1. Trims the cycle data to start at the beginning of the reel-out (RO) phase.
    2. Reassigns flight phase indices:
        - Sets all data from start of RO to its end as `1` (pure RO).
        - Sets data from the start of reel-in-roll-out (RIRO) to end of reel-in (RI) as `3` (pure RI),
          effectively overriding RIRO portions labeled as `4`.

    Args:
        exp_cycle_dataframe (pd.DataFrame): DataFrame representing a single kite flight cycle.
            Must include the columns:
            - 'flight_phase_index'
            - 'kite_actual_depower'
            - 'ground_tether_length'
            - 'ground_tether_reelout_speed'

    Returns:
        pd.DataFrame: Modified DataFrame with updated `flight_phase_index` values to represent
        clean RO (1) and RI (3) phases for quasi-steady analysis.
    """    
    start_RO_idx = find_start_RO(exp_cycle_dataframe)
    exp_cycle_dataframe.drop(exp_cycle_dataframe.index[:start_RO_idx], inplace=True)
    exp_cycle_dataframe.reset_index(drop=True, inplace=True)

    # Change the flight_phase_index of elements of RORI to RO
    _, end_RO_idx = find_end_RO_and_max_tether_length(exp_cycle_dataframe, threshold=8.0)
    exp_cycle_dataframe.loc[0:end_RO_idx, 'flight_phase_index'] = 1

    # Change the flight_phase_index of elements of RIRO to RI
    start_RIRO_idx = next((i for i, x in enumerate(exp_cycle_dataframe.flight_phase_index) if x == 4), -1)
    _, end_RI_idx = find_end_RI_and_min_tether_length(exp_cycle_dataframe, threshold=2)
    exp_cycle_dataframe.loc[start_RIRO_idx:end_RI_idx, 'flight_phase_index'] = 3
    return exp_cycle_dataframe

def pack_operational_parameters_and_results(exp_cycle_dataframe): 
    """
    Extracts and computes key operational parameters and performance metrics from 
    a full flight cycle of a tethered kite system.

    This function:
    - Refines flight phase labels using `find_qsm_flight_phases`.
    - Segregates the data into distinct flight phases: Reel-Out (RO), Reel-In (RI), and RIRO.
    - Calculates important metrics including:
        - Azimuth and elevation characteristics during RO
        - Tether length extremes
        - Average forces, reeling speeds, and mechanical power across phases
        - Wind speed and optimizer step index averages

    Args:
        exp_cycle_dataframe (pd.DataFrame): DataFrame representing a full flight cycle.
            Must include the following columns:
            - 'flight_phase_index'
            - 'kite_actual_depower'
            - 'kite_azimuth', 'kite_elevation'
            - 'ground_tether_length', 'ground_tether_force', 'ground_tether_reelout_speed'
            - 'ground_mech_power'
            - 'optimizer_step_index'
            - '100m Wind Speed (m/s)'

    Returns:
        dict: Dictionary of computed metrics and parameters for the flight cycle, including:
            - Azimuth and elevation parameters in radians and degrees
            - Min/max tether lengths (RO/RI)
            - Average tether force, reeling speed, and mechanical power for each phase
            - Mean wind speed and optimizer step during RO
    """  
    exp_cycle_dataframe = find_qsm_flight_phases(exp_cycle_dataframe)
    RO_dataframe = exp_cycle_dataframe[exp_cycle_dataframe.flight_phase_index == 1]
    RI_dataframe = exp_cycle_dataframe[exp_cycle_dataframe.flight_phase_index == 3]
    RIRO_dataframe = exp_cycle_dataframe[exp_cycle_dataframe.flight_phase_index == 4]    

    max_az_trac, rel_el_angle, avg_el_angle = find_RO_pattern_param(RO_dataframe)
    max_tether_length_RO, _ = find_end_RO_and_max_tether_length(exp_cycle_dataframe)
    min_tether_length_RI, _ = find_end_RI_and_min_tether_length(exp_cycle_dataframe)
    
    exp_cycle_res_dict = {'RO_max_azimuth_rad': max_az_trac,
                 'RO_max_azimuth_deg': np.rad2deg(max_az_trac),
                 'RO_rel_elevation_rad': rel_el_angle,
                 'RO_rel_elevation_deg': np.rad2deg(rel_el_angle),
                 'RO_avg_elevation_rad': avg_el_angle,
                 'RO_avg_elevation_deg': np.rad2deg(avg_el_angle),

                 'RO_max_tether_length_m' : max_tether_length_RO,
                 'RO_min_tether_length_m' : exp_cycle_dataframe.ground_tether_length[0],
                 'RI_min_tether_length_m' : min_tether_length_RI,

                 'RO_tether_force_avg_N' : np.mean(RO_dataframe.ground_tether_force)*9.806,
                 'RI_tether_force_avg_N' : np.mean(RI_dataframe.ground_tether_force)*9.806, 

                 'RO_reelout_speed_avg_mps' : np.mean(RO_dataframe.ground_tether_reelout_speed),
                 'RI_reelout_speed_avg_mps' : np.mean(RI_dataframe.ground_tether_reelout_speed),
                 'RIRO_reelout_speed_avg_mps' : np.mean(RIRO_dataframe.ground_tether_reelout_speed),
        
                 'RI_mech_power_avg_kW': np.mean(RI_dataframe.ground_mech_power)/1000,
                 'RO_mech_power_avg_kW': np.mean(RO_dataframe.ground_mech_power)/1000,
                 'RIRO_mech_power_avg_kW': np.mean(RIRO_dataframe.ground_mech_power)/1000,
                 'cycle_mech_power_avg_kW': np.mean(exp_cycle_dataframe.ground_mech_power)/1000,
                 'RO_autotuner_step': np.mean(RO_dataframe.optimizer_step_index),
                 'wind_speed_100m_mps': np.mean(exp_cycle_dataframe['100m Wind Speed (m/s)'])    
                 }
    return exp_cycle_res_dict

# Running simulations
def run_simulation_from_exp_dataframe(exp_cycle_dataframe, sys_props, timestep = 0.25, control = 'speed'):
    """
    Runs a simulated power cycle of a tethered kite system using parameters extracted from experimental flight data.

    This function:
    - Identifies flight phases from the input experimental data.
    - Computes operational parameters like reeling speed, tether force, and flight pattern geometry.
    - Constructs environment wind profiles based on average wind speed at 100m during each flight phase.
    - Initializes and runs a simulation cycle using the extracted parameters and specified control strategy.
    - Returns simulation outputs including time-series data and aggregate performance metrics.

    Args:
        exp_cycle_dataframe (pd.DataFrame): Experimental data from a full flight cycle.
            Must contain the following columns:
            - 'flight_phase_index', 'kite_actual_depower', 'kite_azimuth', 'kite_elevation',
            - 'ground_tether_length', 'ground_tether_force', 'ground_tether_reelout_speed',
            - 'ground_mech_power', '100m Wind Speed (m/s)'

        sys_props (SystemProperties): System configuration used to run the simulation.

        timestep (float, optional): Simulation timestep in seconds. Defaults to 0.25.

        control (str, optional): Control strategy to use for the simulation.
            Options:
                - 'speed': Uses average reeling speeds for each phase (default).
                - 'force': Uses average ground tether force as control input.
                - 'hybrid': Uses speed control for reel-out (RO) and force control for reel-in (RI).

    Returns:
        Tuple[pd.DataFrame, dict]: 
            - sim_cycle_dataframe: Simulated time-series data for the full cycle.
            - sim_cycle_res_dict: Dictionary of summary performance metrics from the simulation.
    """
    exp_cycle_dataframe = find_qsm_flight_phases(exp_cycle_dataframe)

    exp_RO_dataframe = exp_cycle_dataframe[exp_cycle_dataframe.flight_phase_index == 1]
    exp_RI_dataframe = exp_cycle_dataframe[exp_cycle_dataframe.flight_phase_index == 3]
    exp_RIRO_dataframe = exp_cycle_dataframe[exp_cycle_dataframe.flight_phase_index == 4]    

    max_az_trac, rel_el_angle, avg_el_angle = find_RO_pattern_param(exp_RO_dataframe)

    avg_tether_force_RO = np.mean(exp_RO_dataframe.ground_tether_force)*9.806
    avg_tether_force_RI = np.mean(exp_RI_dataframe.ground_tether_force)*9.806
    
    avg_reeling_speed_RO = np.mean(exp_RO_dataframe.ground_tether_reelout_speed)
    avg_reeling_speed_RI = np.mean(exp_RI_dataframe.ground_tether_reelout_speed)
    avg_reeling_speed_RIRO = np.mean(exp_RIRO_dataframe.ground_tether_reelout_speed)
   
    max_tether_length_RO, _ = find_end_RO_and_max_tether_length(exp_cycle_dataframe)
    min_tether_length_RI, _ = find_end_RI_and_min_tether_length(exp_cycle_dataframe)

    env_avg = LogProfile()
    env_avg.set_reference_wind_speed(np.mean(exp_cycle_dataframe['100m Wind Speed (m/s)']))
    env_avg.set_reference_height(100)

    env_trac = LogProfile()
    env_trac.set_reference_wind_speed(np.mean(exp_RO_dataframe['100m Wind Speed (m/s)']))
    env_trac.set_reference_height(100)
    env_retr = LogProfile()
    env_retr.set_reference_wind_speed(np.mean(exp_RI_dataframe['100m Wind Speed (m/s)']))
    env_retr.set_reference_height(100)
    env_trans = LogProfile()
    env_trans.set_reference_wind_speed(np.mean(exp_RIRO_dataframe['100m Wind Speed (m/s)']))
    env_trans.set_reference_height(100)
   
    RO_settings = {'control': ('reeling_speed', avg_reeling_speed_RO),
                   'pattern': {'azimuth_angle': max_az_trac, 'rel_elevation_angle': rel_el_angle}, 'time_step': timestep}
        
    # Default cycle settings for speed control
    cycle_settings = {'cycle': {
                            'traction_phase': TractionPhasePattern,
                            'include_transition_energy': True
                        },
                        'retraction': {
                            'control': ('reeling_speed', avg_reeling_speed_RI),
                            'time_step': timestep

                        },
                        'transition': {
                            'control': ('reeling_speed', avg_reeling_speed_RIRO),
                            'time_step': timestep
                        },
                        'traction': RO_settings
                    }
    
    if control == 'force': 
        print('Setting to force control...')
        cycle_settings['traction']['control'] = ('tether_force_ground', avg_tether_force_RO)
        cycle_settings['retraction']['control'] = ('tether_force_ground', avg_tether_force_RI)

    if control == 'hybrid':
        print('Setting to hybrid control (RO-speed control, RI-force control)...')  
        cycle_settings['retraction']['control'] = ('tether_force_ground', avg_tether_force_RI)

    cycle = Cycle(cycle_settings)
    cycle.elevation_angle_traction = avg_el_angle
    cycle.tether_length_start_retraction = max_tether_length_RO
    cycle.tether_length_start_traction = exp_cycle_dataframe.ground_tether_length[0]
    cycle.tether_length_end_retraction = min_tether_length_RI
    
    
    cycle.run_simulation(sys_props, [env_retr, env_trans, env_trac], print_summary=False) 

    sim_cycle_dataframe, sim_cycle_res_dict = pack_results_sim(cycle, env_avg)

    
    return sim_cycle_dataframe, sim_cycle_res_dict

# Packing results 
def pack_results_sim(cycle, env_state):
    """
    Extracts time-series data and summary metrics from a completed kite power cycle simulation.

    This function:
    - Compiles time-series data (reel-out speed, tether force, power, position, tether length, etc.)
      for each flight phase: retraction (RI), transition (RIRO), and traction (RO).
    - Aligns time indices sequentially across all phases.
    - Calculates average performance metrics for each phase and for the full cycle.
    - Packages both the time-series data and summary results into structured outputs.

    Args:
        cycle (Cycle): Simulated kite power cycle object containing phase data, results,
                       and metadata for traction, retraction, and transition phases.

        env_state (LogProfile): Atmospheric log wind profile used to simulate the environment.
                                Used to extract reference wind speed (e.g., at 100 m altitude).

    Returns:
        Tuple[pd.DataFrame, dict]:
            - sim_cycle_dataframe (pd.DataFrame): Time-series simulation data, including:
                - 'time': Time [s]
                - 'ground_tether_reelout_speed': Reel-out speed [m/s]
                - 'ground_tether_force': Ground tether force [N]
                - 'ground_tether_length': Tether length [m]
                - 'ground_mech_power': Ground mechanical power [W]
                - 'x_pos', 'y_pos', 'z_pos': Kite position coordinates [m]
                - 'flight_phase_index': Encoded phase: 1 (RO), 3 (RI), 4 (RIRO)

            - sim_cycle_res_dict (dict): Summary metrics, including:
                - Avg. tether forces (N and kgf) and reeling speeds for each phase (RO, RI, RIRO)
                - Avg. mechanical power per phase and total
                - Duration [s] of each phase and total
                - Avg. elevation angle during RO [rad]
                - Tether lengths at start/end of phases [m]
                - Wind speed at 100m [m/s]
    
    Raises:
        ValueError: If any of the phase attributes (e.g., `steady_states`) are empty or malformed.
    """
    # --- Traction Phase ---
    time_trac = cycle.traction_phase.time
    time_trac = [x - cycle.traction_phase.time[0] for x in time_trac]
    reel_speeds = [s.reeling_speed for s in cycle.traction_phase.steady_states]
    RO_reel_speed = np.mean(reel_speeds)
    tether_forces = [s.tether_force_ground for s in cycle.traction_phase.steady_states]
    RO_force = np.mean(tether_forces)
    app_wind_speed = [s.apparent_wind_speed for s in cycle.traction_phase.steady_states]
    power_ground = [s.power_ground for s in cycle.traction_phase.steady_states]
    RO_power = np.mean(power_ground)
    x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in cycle.traction_phase.kinematics])
    x_traj = list(x_traj)
    y_traj = list(y_traj)
    z_traj = list(z_traj)
    tether_length = [kin.straight_tether_length for kin in cycle.traction_phase.kinematics]
    flight_phase_index = 1*np.ones_like(tether_length)
    # Assuming each list is a column
    data = [time_trac, reel_speeds, tether_forces, tether_length, power_ground, x_traj, y_traj, z_traj, flight_phase_index]
    RO_sim_df = pd.DataFrame(list(zip(*data)),\
                              columns=['time', 'ground_tether_reelout_speed', 'ground_tether_force',\
                                        'ground_tether_length', 'ground_mech_power', 'x_pos', 'y_pos', 'z_pos', 'flight_phase_index'])

    # --- Retraction Phase ---
    time_retr = cycle.retraction_phase.time
    time_retr = [x - (cycle.retraction_phase.time[0] - time_trac[-1]) for x in time_retr]
    reel_speeds = [s.reeling_speed for s in cycle.retraction_phase.steady_states]
    RI_reel_speed = np.mean(reel_speeds)
    tether_forces = [s.tether_force_ground for s in cycle.retraction_phase.steady_states]
    RI_force = np.mean(tether_forces)    
    power_ground = [s.power_ground for s in cycle.retraction_phase.steady_states]
    x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in cycle.retraction_phase.kinematics])
    x_traj = list(x_traj)
    y_traj = list(y_traj)
    z_traj = list(z_traj)
    tether_length = [s.straight_tether_length for s in cycle.retraction_phase.kinematics]
    flight_phase_index = 3*np.ones_like(tether_length)
    # Assuming each list is a column
    data = [time_retr, reel_speeds, tether_forces, tether_length, power_ground, x_traj, y_traj, z_traj, flight_phase_index]
    RI_sim_df = pd.DataFrame(list(zip(*data)),\
                              columns=['time', 'ground_tether_reelout_speed', 'ground_tether_force',\
                                        'ground_tether_length', 'ground_mech_power','x_pos', 'y_pos', 'z_pos', 'flight_phase_index'])


    # --- Transition Phase ---
    time_RIRO = cycle.transition_phase.time
    time_RIRO = [x - (cycle.transition_phase.time[0] - time_retr[-1]) for x in time_RIRO]
    reel_speeds = [s.reeling_speed for s in cycle.transition_phase.steady_states]
    RIRO_reel_speed = np.mean(reel_speeds)
    tether_forces = [s.tether_force_ground for s in cycle.transition_phase.steady_states]
    RIRO_force = np.mean(tether_forces)
    power_ground = [s.power_ground for s in cycle.transition_phase.steady_states]
    x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in cycle.transition_phase.kinematics])
    x_traj = list(x_traj)
    y_traj = list(y_traj)
    z_traj = list(z_traj)
    tether_length = [s.straight_tether_length for s in cycle.transition_phase.kinematics]
    flight_phase_index = 4*np.ones_like(tether_length)
    # Assuming each list is a column
    data = [time_RIRO, reel_speeds, tether_forces, tether_length, power_ground, x_traj, y_traj, z_traj, flight_phase_index]
    RIRO_sim_df = pd.DataFrame(list(zip(*data)),\
                                columns=['time', 'ground_tether_reelout_speed', 'ground_tether_force',\
                                          'ground_tether_length', 'ground_mech_power', 'x_pos', 'y_pos', 'z_pos', 'flight_phase_index'])

    sim_cycle_dataframe = pd.concat([RO_sim_df, RI_sim_df, RIRO_sim_df], axis=0, ignore_index=True)
    sim_cycle_dataframe.reset_index(drop=True, inplace=True)

    sim_cycle_res_dict = ({'RI_reelout_speed_avg_mps': RI_reel_speed,
                      'RI_tether_force_avg_kgf': RI_force/9.8066,
                      'RI_tether_force_avg_N': RI_force,

                      'RO_reelout_speed_avg_mps': RO_reel_speed,
                      'RO_tether_force_avg_kgf': RO_force/9.8066, 
                      'RO_tether_force_avg_N': RO_force,  

                      'RIRO_reelout_speed_avg_mps': RIRO_reel_speed,
                      'RIRO_tether_force_avg_kgf': RIRO_force/9.8066,
                      'RIRO_tether_force_avg_N': RIRO_force,

                       # Trajectory data and cycle settings
                      'RO_elevation_kite_avg_rad': cycle.traction_phase.elevation_angle,
                      'tether_length_reelout_min_m': cycle.tether_length_end_retraction,
                      'tether_length_reelout_max_m': cycle.tether_length_start_retraction,  

                      # Power
                      'RI_mech_power_avg_kW': cycle.retraction_phase.average_power/1000,
                      'RO_mech_power_avg_kW': cycle.traction_phase.average_power/1000,
                      'RIRO_mech_power_avg_kW': cycle.transition_phase.average_power/1000,
                      'cycle_mech_power_avg_kW': cycle.average_power/1000,

                      # Duration
                      'RO_duration_s': cycle.traction_phase.duration,
                      'RIRO_duration_s': cycle.transition_phase.duration,
                      'RI_duration_s': cycle.retraction_phase.duration,
                      'duration_cycle_s': cycle.duration,
                      'wind_speed_100m_mps': env_state.calculate_wind(100)})
    
    return sim_cycle_dataframe, sim_cycle_res_dict

def run_simulations_from_list_of_exp_dataframes(cycle_dataframe_list, sys_props, control='speed'):
    all_sim_dataframes = [] 
    all_exp_dataframes = []
    all_cycle_res_sim = pd.DataFrame()  
    all_cycle_res_exp = pd.DataFrame() 

    for i, exp_cycle_dataframe in enumerate(cycle_dataframe_list):
        try:
            sim_cycle_dataframe, cycle_res_sim_dict = run_simulation_from_exp_dataframe(exp_cycle_dataframe, sys_props, control = control)
            # Append experimental results
            exp_cycle_dataframe = find_qsm_flight_phases(exp_cycle_dataframe)
            all_exp_dataframes.append(exp_cycle_dataframe)         
            cycle_res_exp_dict = pack_operational_parameters_and_results(exp_cycle_dataframe)
            all_cycle_res_exp = pd.concat([all_cycle_res_exp, pd.DataFrame([cycle_res_exp_dict])])    

            # Append simulation results
            all_sim_dataframes.append(sim_cycle_dataframe)        
            all_cycle_res_sim = pd.concat([all_cycle_res_sim, pd.DataFrame([cycle_res_sim_dict])])   

        except Exception as e:
            print(f'Sim failed for cycle {i}: {e}')
            continue
    print("Simulation complete.")
    
    return all_sim_dataframes, all_cycle_res_sim, all_exp_dataframes, all_cycle_res_exp
    
def cycle_to_cycle_plot(df_sim, df_exp,cycle_sim, cycle_exp):
    colors = ["#F85033",  # Tomato
          "#4682B4",  # SteelBlue
          "#32CD32",  # LimeGreen
          "#FFD700"]  # Gold
    
    ec = '#00395d'
    sc = '#00aeef'

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[0.8, 1], wspace=0.4)

    fig.suptitle('Cycle comparison')
    # 3D Plot on the Left
    ax3d = fig.add_subplot(gs[0:3, 0], projection='3d')
    ax3d.view_init(elev=35, azim=40)
    for i in range(1,5):
        ax3d.plot(df_sim[df_sim.flight_phase_index==i].x_pos, 
                df_sim[df_sim.flight_phase_index==i].y_pos,
                df_sim[df_sim.flight_phase_index==i].z_pos,
                c = colors[i-1], linewidth = 2)

    x_pos = df_exp.ground_tether_length * np.cos(df_exp.kite_azimuth) * np.cos(df_exp.kite_elevation)
    y_pos = df_exp.ground_tether_length * np.sin(df_exp.kite_azimuth) * np.cos(df_exp.kite_elevation)
    z_pos = df_exp.ground_tether_length * np.sin(df_exp.kite_elevation)
        
    for i in range(1,5):
        ax3d.plot(x_pos[df_exp.flight_phase_index==i], 
                y_pos[df_exp.flight_phase_index==i],
                z_pos[df_exp.flight_phase_index==i],
                c = colors[i-1], linewidth = 2, linestyle = '--')
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_title("3D trajectories")
    ax3d.legend(['Reel-out', 'RORI', 'Reel-in', 'RIRO'])


    ax0 = fig.add_subplot(gs[3, 0])
    # Data extraction
    labels = ["RO", "RI", "RIRO", "Cycle"]
    sim_values = [
        cycle_sim["RO_mech_power_avg_kW"],
        cycle_sim["RI_mech_power_avg_kW"],
        cycle_sim["RIRO_mech_power_avg_kW"],
        cycle_sim["cycle_mech_power_avg_kW"]
        
    ]
    exp_values = [
        cycle_exp["RO_mech_power_avg_kW"],
        cycle_exp["RI_mech_power_avg_kW"],
        cycle_exp["RIRO_mech_power_avg_kW"],
        cycle_exp["cycle_mech_power_avg_kW"]
    ]

    # Plot settings
    x = np.arange(len(labels))
    width = 0.35  # width of the bars

    bar1 = ax0.bar(x - width / 2, sim_values, width, label="Simulation", color=sc)
    bar2 = ax0.bar(x + width / 2, exp_values, width, label="Experiment", color=ec)
    xlb, xub = plt.xlim()
    ax0.hlines(0, xlb, xub, colors=['black'], linewidth = 1)
    # Adding labels, title, and formatting
    ax0.set_xlabel("Power Metrics")
    ax0.set_ylabel("Average power (kW)")
    ax0.set_title("Comparison of average mechanical power")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)


    # Subplot 1 (Length)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(df_exp.time - df_exp.time[0], df_exp.ground_tether_length, label="Experiment", c=ec)
    ax1.plot(df_sim.time, df_sim.ground_tether_length, label="Simulation", c=sc)
    ax1.set_title("Tether length")
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_ylabel('Length [m]')


    # Subplot 2 (Reelout Speed)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(df_exp.time - df_exp.time[0], df_exp.ground_tether_reelout_speed, label="Experiment", c=ec)
    ax2.plot(df_sim.time, df_sim.ground_tether_reelout_speed, label="Simulation", c=sc)
    ax2.set_title("Reelout speed")
    ax2.set_xticks([])
    ax2.set_ylabel('Speed [m/s]')


    # Subplot 3 (Force)
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(df_exp.time - df_exp.time[0], df_exp.ground_tether_force * 9.806 / 1000, label="Experiment", c=ec)
    ax3.plot(df_sim.time, df_sim.ground_tether_force / 1000, label="Simulation", c=sc)
    ax3.set_title("Ground tether force")
    ax3.set_ylabel('Force [kN]')
    ax3.set_xticks([])

    # Subplot 4 (Power)
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.plot(df_exp.time - df_exp.time[0], df_exp.ground_mech_power / 1000, label="Experiment", c=ec)
    ax4.plot(df_sim.time, df_sim.ground_mech_power / 1000, label="Simulation", c=sc)
    ax4.set_title("Mechanical power")
    ax4.set_ylabel('Power [kW]')
    ax4.set_xlabel('Time [s]')