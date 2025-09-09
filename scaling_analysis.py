import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from power_curve_single_profile import load_power_curve_single_profile_results_and_plot_trajectories

def output_angles_to_deg(output_dataframe:pd.DataFrame):
    cols_rad = ['theta_avg_RO [rad]', 'theta_rel_RO [rad]', 'phi_max_RO [rad]']
    cols_deg = ['theta_avg_RO [deg]', 'theta_rel_RO [deg]', 'phi_max_RO [deg]']

    for col_deg, col_rad in zip(cols_deg, cols_rad):
        output_dataframe[col_deg] = output_dataframe[col_rad]*180/np.pi 

    return output_dataframe


fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True)

to_plot = ['P_cycle [W]', 'F_RO [N]',\
       'F_RI [N]', 'theta_avg_RO [deg]', 'theta_rel_RO [deg]',\
       'phi_max_RO [deg]', 'stroke_tether [m]', 'min_length_tether [m]',\
       'n_crosswind_patterns [-]']



case_names = ['baseline', '100', '200', '300', '400']

for case_name in case_names:
    config_filename = 'output/config_' + case_name + '.yaml'
    output_filename = 'output/' + case_name + '/power_curve_log_profile.csv'

    output_dataframe = pd.read_csv(output_filename, sep = ';')
    output_dataframe = output_angles_to_deg(output_dataframe)

    mask = output_dataframe['success [-]'] == True
    for ax, plot in zip(axs.flatten(), to_plot):
        ax.plot(output_dataframe['v_100m [m/s]'].loc[mask], output_dataframe[plot].loc[mask])
        ax.set_ylabel(plot)

axs[0,0].legend(case_names)

for ax in axs.flatten()[-3:]:
    ax.set_xlabel('wind speed at 100 m')
plt.show()