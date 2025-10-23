import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
sys.path.append(os.path.abspath('./'))
from exp_validation_utils import *
from qsm import *
from awe_pe.utils import *

"""
This file provides a collection of `example_*` functions that load an experimental kite-power system cycle
from a pickle file (prepared via Protologger), analyze flight phases, and benchmark against corresponding simulations.
Visualizations cover reel-in/out transitions, depower control, 3D-to-2D trajectory mapping, and aggregated performance plots.

Functions
---------
example_1()
    Plot the start and end of the **Reeling Out (RO)** phase:
    - Ground tether length with vertical lines marking start/end of RO (using QSM thresholds)
    - Kite depower signal overlay with threshold-based markers

example_2()
    Plot the end of the **Reeling In (RI)** phase:
    - Tether length traces showing the point of minimum length
    - Kite depower trace with threshold marker at the start of re-powering

example_3()
    Run and compare a **speed controlled** simulation against experimental data:
    - Packs operational parameters
    - Runs simulation for the same cycle
    - Overlays time-series via `cycle_to_cycle_plot()`

example_4()
    Same as `example_3()`, but using **force controlled** simulation.

example_5()
    Same as `example_3()`, but using **hybrid control** (speed + force).

example_6()
    Visualize the **RO pattern in 2D**:
    - Computes kite XYZ positions based on tether, azimuth, elevation
    - Extracts average/relative elevation and max azimuth
    - Plots elevation profile and plan-view trajectory arcs

example_7()
    Compare simulated vs experimental **RO pattern trajectories**:
    - Uses force controlled simulation
    - Overplots experimental and simulated RO trajectories in both elevation and plan views

example_8()
    Batch process multiple cycles:
    - Loads a list of exp cycles via Protologger
    - Runs speed controlled simulations for each
    - Scatter plots cycle-averaged metrics (power, reel-out speed, tether force) vs wind speed

example_9()
    Aggregated **wind speed binned comparison** between experiment and simulation:
    - Bins cycles by 100 m wind speed
    - Computes mean ± std of cycle mechanical power per bin
    - Plots error-bar curves overlaying experimental vs simulated results

Usage
-----
- Ensure `examples_data/experimental_cycle_example.pkl` and, if using `example_8/9`, 
  `examples_data/experimental_cycles_list_example.pkl` are present.
- Configure the kite system via `config.yaml`.
- Run any `example_*()` to generate the corresponding plot(s) and visual analysis
"""

colors = ["#F85033",  # Tomato
          "#4682B4",  # SteelBlue
          "#32CD32",  # LimeGreen
          "#FFD700"]  # Gold

# Create the colormap
my_cmap = mcolors.ListedColormap(colors)

def example_1():
    # Uncomment the following section to read the Protologger file once you downloaded some. For the example we are reading the 
    # pickle file from 'examples_data/experimental_cycle_example.pkl'.
    """
    # Your path to experimental data
    data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/2024/'
    test_name = 'Test-2024-02-15_GS3/'
    cycle_dataframe_list = load_process_protologger_file(data_path, test_name)

    example_exp_cycle_idx = 14

    example_exp_cycle = cycle_dataframe_list[example_exp_cycle_idx]
    
    with open('examples_data/experimental_cycle_example.pkl', "wb") as file: pickle.dump(example_exp_cycle, file)
    """
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)

    # Find start and end of RO
    end_RO_threshold = 8.
    start_RO_idx = find_start_RO(example_exp_cycle)
    max_tether_length_RO, end_RO_idx = find_end_RO_and_max_tether_length(example_exp_cycle, end_RO_threshold)
    start_time = example_exp_cycle.time[0]
    end_time = example_exp_cycle.time.to_numpy()[-1]
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex = True)

    for i in range(1,5):
        phase_mask = example_exp_cycle.flight_phase_index == i   
        phase = example_exp_cycle[phase_mask]    
        ax[0].plot(phase.time - start_time, phase.ground_tether_length,
                    c=colors[i-1], linewidth=2,  zorder=1)
        ax[1].plot(phase.time - start_time, phase.kite_actual_depower,
                    c=colors[i-1], linewidth=2,  zorder=1)
    
    y_lb, y_ub = ax[0].get_ylim()
    ax[0].vlines(example_exp_cycle.time[start_RO_idx] - start_time, y_lb, y_ub,
                  color='#a9a29c', linestyle='--', label='Start of RO for QSM')
    ax[0].vlines(example_exp_cycle.time[end_RO_idx] - start_time, y_lb, y_ub,
                  color='#a9a29c', linestyle='--', label='End of RO for QSM')
    ax[0].scatter(example_exp_cycle.time[end_RO_idx] - start_time, max_tether_length_RO,
                   c='#d62828', label='Max tether length',zorder=2)
    
    ax[0].set_ylabel('Tether length [m]')
    ax[0].set_ylim([y_lb, y_ub])
    ax[0].legend()

    y_lb, y_ub = ax[1].get_ylim()
    ax[1].vlines(example_exp_cycle.time[start_RO_idx] - start_time, y_lb, y_ub,
                  color='#a9a29c', linestyle='--', label='Start of RO for QSM')
    ax[1].vlines(example_exp_cycle.time[end_RO_idx] - start_time, y_lb, y_ub,
                  color='#a9a29c', linestyle='--', label='End of RO for QSM')
    ax[1].hlines(example_exp_cycle.kite_actual_depower[end_RO_idx], 0, (end_time-start_time),
                  color='#a9a29c', linestyle=':', label=str(end_RO_threshold) + '% threshold')
    ax[1].scatter(example_exp_cycle.time[end_RO_idx] - start_time,
                  example_exp_cycle.kite_actual_depower[end_RO_idx], c='#d62828',
                    label='Start depowering', zorder=3)    
    
    ax[1].set_xlabel('Time [s]')    
    ax[1].set_ylabel('Kite depower [%]')
    ax[1].set_ylim([y_lb, y_ub])
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def example_2():
    # See example 1 on how to get the example_exp_cycle
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)
    start_time = example_exp_cycle.time.iloc[0]
    end_time = example_exp_cycle.time.to_numpy()[-1]
    end_RI_threshold = 2.
    min_tether_length_RI, end_RI_idx = find_end_RI_and_min_tether_length(example_exp_cycle, end_RI_threshold)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    for i in range(1, 5):
        phase_mask = example_exp_cycle.flight_phase_index == i
        ax[0].plot(example_exp_cycle[phase_mask].time - start_time,
                   example_exp_cycle[phase_mask].ground_tether_length,
                   c=colors[i-1], linewidth=2, zorder=1)
        ax[1].plot(example_exp_cycle[phase_mask].time - start_time,
                   example_exp_cycle[phase_mask].kite_actual_depower,
                   c=colors[i-1], linewidth=2, zorder=1)

    y_lb, y_ub = ax[0].get_ylim()
    ax[0].vlines(example_exp_cycle.time[end_RI_idx] - start_time, y_lb, y_ub,
                 color='#a9a29c', linestyle='--', label='End of RI for QSM')
    ax[0].scatter(example_exp_cycle.time[end_RI_idx] - start_time, min_tether_length_RI,
                  c='#d62828', label='Min tether length', zorder=2)
    ax[0].set_ylabel('Tether length [m]')
    ax[0].set_ylim([y_lb, y_ub])
    ax[0].legend()       

    ax[1].scatter(example_exp_cycle.time[end_RI_idx] - start_time,
                  example_exp_cycle.kite_actual_depower[end_RI_idx],
                  c='#d62828', label='Finish powering', zorder=3)

    y_lb, y_ub = ax[1].get_ylim()
    ax[1].vlines(example_exp_cycle.time[end_RI_idx] - start_time, y_lb, y_ub,
                 color='#a9a29c', linestyle='--', label='End of RI for QSM')
    ax[1].hlines(example_exp_cycle.kite_actual_depower[end_RI_idx], 0, end_time - start_time,
                 color='#a9a29c', linestyle=':', label=str(end_RI_threshold) + '% threshold')

    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Kite depower [%]')
    ax[1].set_ylim([y_lb, y_ub])
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def example_3():
    # See example 1 on how to get the example_exp_cycle
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)
    exp_cycle_res_dict = pack_operational_parameters_and_results(example_exp_cycle)
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f) 
    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)
    sim_cycle_dataframe, sim_cycle_res_dict = run_simulation_from_exp_dataframe(example_exp_cycle, sys_props, control='speed')

    cycle_to_cycle_plot(sim_cycle_dataframe, example_exp_cycle, sim_cycle_res_dict, exp_cycle_res_dict)

    plt.show()

def example_4():
    # See example 1 on how to get the example_exp_cycle
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)
    exp_cycle_res_dict = pack_operational_parameters_and_results(example_exp_cycle)
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f) 
    sys_props = parse_system_properties_and_bounds(config)
        
    sys_props = SystemProperties(sys_props)
    
    # Change drag coefficient
    sys_props.kite_drag_coefficient_powered = 0.181
    
    sim_cycle_dataframe, sim_cycle_res_dict = run_simulation_from_exp_dataframe(example_exp_cycle, sys_props, control='force')

    cycle_to_cycle_plot(sim_cycle_dataframe, example_exp_cycle, sim_cycle_res_dict, exp_cycle_res_dict)

    plt.show()

def example_5():
    # See example 1 on how to get the example_exp_cycle
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)
    exp_cycle_res_dict = pack_operational_parameters_and_results(example_exp_cycle)
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f) 
    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)
    sim_cycle_dataframe, sim_cycle_res_dict = run_simulation_from_exp_dataframe(example_exp_cycle, sys_props, control='hybrid')

    cycle_to_cycle_plot(sim_cycle_dataframe, example_exp_cycle, sim_cycle_res_dict, exp_cycle_res_dict)

    plt.show()

def example_6():
    def plot_arc(angle0, angle1, radius):
        a_range = np.linspace(angle0, angle1, 30)
        x_cor = radius*np.cos(a_range)
        y_cor = radius*np.sin(a_range)
        plt.plot(x_cor, y_cor, linewidth=1, color='black')
    # See example 1 on how to get the example_exp_cycle
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)

    example_exp_cycle['x_pos'] = example_exp_cycle.ground_tether_length * np.cos(example_exp_cycle.kite_azimuth) * np.cos(example_exp_cycle.kite_elevation)
    example_exp_cycle['y_pos'] = example_exp_cycle.ground_tether_length * np.sin(example_exp_cycle.kite_azimuth) * np.cos(example_exp_cycle.kite_elevation)
    example_exp_cycle['z_pos'] = example_exp_cycle.ground_tether_length * np.sin(example_exp_cycle.kite_elevation)
    
    RO_phase_dataframe = example_exp_cycle[example_exp_cycle.flight_phase_index == 1]
    max_azimuth, rel_elevation, avg_elevation = find_RO_pattern_param(RO_phase_dataframe)

    L_min, _ = find_end_RI_and_min_tether_length(example_exp_cycle)
    L_max, _ = find_end_RO_and_max_tether_length(example_exp_cycle)
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_elevation)*x_plt
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.sqrt(RO_phase_dataframe.x_pos**2 + RO_phase_dataframe.y_pos**2),                 
                    RO_phase_dataframe.z_pos, c = colors[0], linewidth = 2)

    plot_arc(0, avg_elevation+rel_elevation, L_min)
    plot_arc(0, avg_elevation+rel_elevation, L_max)
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, avg_elevation, 50)
    plt.text(55, 10, r'$\theta_{avg}$', {'fontsize': 12})
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_elevation+rel_elevation)*x_plt
    plot_arc(avg_elevation, avg_elevation+rel_elevation, 70)
    plot_arc(avg_elevation, avg_elevation+rel_elevation, 75)
    plt.text(12, 50, r'$\theta_{rel}$', {'fontsize': 12})
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_elevation-rel_elevation)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.axis('equal')
    plt.xlim([0, L_max + 20])
    plt.hlines(0, 0, L_max + 20, colors=['black'], linestyles=['-'], linewidth=1)
    plt.xticks([0, L_min, L_max], ['GS', r'L$_{min}$', 'L$_{max}$'])
    plt.yticks([], [])
    plt.subplot(1,2,2)
    plt.plot(RO_phase_dataframe.x_pos, RO_phase_dataframe.y_pos, c = colors[0], linewidth = 2)
    plt.hlines(0, 0, 250, colors=['black'], linestyles=['-'], linewidth=1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(max_azimuth)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.plot(x_plt, -y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, max_azimuth, 50)
    plt.text(55, 10, r'$\phi_{max}$', {'fontsize': 12})
    plt.axis('equal')
    plt.xlim([0, 250])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

def example_7():
    def plot_arc(angle0, angle1, radius):
        a_range = np.linspace(angle0, angle1, 30)
        x_cor = radius*np.cos(a_range)
        y_cor = radius*np.sin(a_range)
        plt.plot(x_cor, y_cor, linewidth=1, color='black')
    # See example 1 on how to get the example_exp_cycle
    with open('examples_data/experimental_cycle_example.pkl', "rb") as file: example_exp_cycle = pickle.load(file)
    
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f) 
    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props)
    sim_cycle_dataframe, _ = run_simulation_from_exp_dataframe(example_exp_cycle, sys_props, control='force')


    exp_RO_phase_dataframe = example_exp_cycle[example_exp_cycle.flight_phase_index == 1]
    max_azimuth, rel_elevation, avg_elevation = find_RO_pattern_param(exp_RO_phase_dataframe)

    L_min, _ = find_end_RI_and_min_tether_length(example_exp_cycle)
    L_max, _ = find_end_RO_and_max_tether_length(example_exp_cycle)
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_elevation)*x_plt
    
    sim_RO_phase_dataframe = sim_cycle_dataframe[sim_cycle_dataframe.flight_phase_index == 1]
    

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.sqrt(sim_RO_phase_dataframe.x_pos**2 + sim_RO_phase_dataframe.y_pos**2),                 
                    sim_RO_phase_dataframe.z_pos, c = colors[0], linewidth = 2)

    plot_arc(0, avg_elevation+rel_elevation, L_min)
    plot_arc(0, avg_elevation+rel_elevation, L_max)
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, avg_elevation, 50)
    plt.text(55, 10, r'$\theta_{avg}$', {'fontsize': 12})
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_elevation+rel_elevation)*x_plt
    plot_arc(avg_elevation, avg_elevation+rel_elevation, 70)
    plot_arc(avg_elevation, avg_elevation+rel_elevation, 75)
    plt.text(12, 50, r'$\theta_{rel}$', {'fontsize': 12})
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_elevation-rel_elevation)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.axis('equal')
    plt.xlim([0, L_max + 20])
    plt.hlines(0, 0, L_max + 20, colors=['black'], linestyles=['-'], linewidth=1)
    plt.xticks([0, L_min, L_max], ['GS', r'L$_{min}$', 'L$_{max}$'])
    plt.yticks([], [])
    plt.subplot(1,2,2)
    plt.plot(sim_RO_phase_dataframe.x_pos, sim_RO_phase_dataframe.y_pos, c = colors[0], linewidth = 2)
    plt.hlines(0, 0, 250, colors=['black'], linestyles=['-'], linewidth=1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(max_azimuth)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.plot(x_plt, -y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, max_azimuth, 50)
    plt.text(55, 10, r'$\phi_{max}$', {'fontsize': 12})
    plt.axis('equal')
    plt.xlim([0, 250])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()

def example_8():
    with open('examples_data/experimental_cycles_list_example.pkl', "rb") as file: cycle_dataframe_list = pickle.load(file)

    # --- Load kite system properties --- 
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f) 
    sys_props = parse_system_properties_and_bounds(config)
    sys_props = SystemProperties(sys_props) 

    _, all_cycle_res_sim, _, all_cycle_res_exp = \
        run_simulations_from_list_of_exp_dataframes(cycle_dataframe_list, sys_props, control='speed')
    
    data_to_plot = ['cycle_mech_power_avg_kW', 'RO_reelout_speed_avg_mps', 'RO_tether_force_avg_N']
    labels = ['Cycle mech. power [kW]', 'Reel-out speed [m/s]', 'Reel-out tether force [N]']

    fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True, figsize=(14, 6))
    
    for i, data in enumerate(data_to_plot):    
        ax[i].scatter(all_cycle_res_exp.wind_speed_100m_mps, all_cycle_res_exp[data])
        ax[i].scatter(all_cycle_res_sim.wind_speed_100m_mps, all_cycle_res_sim[data])
        ax[i].set_xlabel('Wind speed at 100 m altitude [m/s]')
        ax[i].set_ylabel(labels[i])
    plt.tight_layout()   
    ax[-1].legend(['Experimental data', 'Simulation data']) 
    plt.show()

def example_9():
    ec = '#00395d'
    sc = '#00aeef'
    def bin_wind_speeds(dataframe, low=4, up=19, int=1):
        # Define bin edges and labels
        bins = np.arange(low, up, int)
        labels = bins[:-1] + int/2 # Labels are lower bin edges

        # Create the bins
        dataframe['wind_bin_100m_mps'] = pd.cut(dataframe['wind_speed_100m_mps'],
                                                 bins=bins, labels=labels, right=False)

        # Group by the bins and calculate mean power
        binned_dataframe = dataframe.groupby('wind_bin_100m_mps')['cycle_mech_power_avg_kW'].agg(
            cycle_mech_power_avg_kW_mean='mean',
            cycle_mech_power_avg_kW_std='std'
            ).reset_index()

        return binned_dataframe
    
    with open('examples_data/experimental_cycles_list_example.pkl', "rb") as file: cycle_dataframe_list = pickle.load(file)

    # --- Load kite system properties --- 
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f) 
    sys_props = parse_system_properties_and_bounds(config)      
    sys_props = SystemProperties(sys_props)   
    
    _, all_cycle_res_sim, _, all_cycle_res_exp = \
        run_simulations_from_list_of_exp_dataframes(cycle_dataframe_list, sys_props, control='speed')
    
    binned_dataframe_exp = bin_wind_speeds(all_cycle_res_exp)
    binned_dataframe_sim = bin_wind_speeds(all_cycle_res_sim)
    
    plt.errorbar(binned_dataframe_exp.wind_bin_100m_mps, binned_dataframe_exp.cycle_mech_power_avg_kW_mean,
                yerr = binned_dataframe_exp.cycle_mech_power_avg_kW_std, elinewidth = 1.5, capsize = 5, c=ec, ms = 6,
                    marker='o', linewidth = 0)

    plt.errorbar(binned_dataframe_sim.wind_bin_100m_mps, binned_dataframe_sim.cycle_mech_power_avg_kW_mean,
                yerr = binned_dataframe_sim.cycle_mech_power_avg_kW_std, elinewidth = 1.5, capsize = 5, c=sc, ms = 6,
                    marker='o', linewidth = 0)


    plt.xlabel('Windspeed [m/s]')
    plt.ylabel('Cycle power [kW]')
    plt.legend(['Experiment', 'Simulation'])
    plt.show()

def main():
    import sys

    # Mapping from option number to example function
    example_funcs = {
        "1": example_1,
        "2": example_2,
        "3": example_3,
        "4": example_4,
        "5": example_5,
        "6": example_6,
        "7": example_7,
        "8": example_8,
        "9": example_9,
    }

    print("Examples about experimental validation and comparison of the QSM model\n")
    print("Select an example to run:")
    for i in range(1, 10):
        print(f"  {i}. example_{i}()")

    choice = input("\nEnter the number of the example to run (1-9), or 'q' to quit: ").strip()

    if choice.lower() == 'q':
        print("Exiting.")
        sys.exit(0)

    if choice in example_funcs:
        print(f"\nRunning example_{choice}()...\n")
        try:
            example_funcs[choice]()
        except Exception as e:
            print(f"An error occurred while running example_{choice}: {e}")
    else:
        print("Invalid choice. Please enter a number between 1 and 9.")

# Optional: run main if script is called directly
if __name__ == "__main__":
    main()