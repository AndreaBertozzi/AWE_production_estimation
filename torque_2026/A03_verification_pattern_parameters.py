import sys
import os
sys.path.append(os.path.abspath('./'))

from awe_pe.exp_validation_utils import *
from awe_pe.utils import *

import numpy as np
import pickle

# cycle_dfs = load_process_protologger_file('', 'experimental_data')

# # Open a file in write-binary mode
# with open('experimental_data/v3_data.pkl', 'wb') as file:
#     pickle.dump(cycle_dfs, file)

# Open a file in write-binary mode
with open('experimental_data/v3_data.pkl', 'rb') as file:
    cycle_dfs = pickle.load(file)

cycle_dfs = load_process_protologger_file('', 'experimental_data')

for df in cycle_dfs:
    df.time = df.time - df.time[0]


df = cycle_dfs[0]


vars_to_plot = ['ground_tether_length', 'ground_tether_reelout_speed', 'ground_tether_force',\
                 'kite_actual_depower', 'kite_elevation', 'kite_azimuth']
ylabs = ['L', r'$v_r$', 'F', r'dep $\%$', r'$\theta$', r'$\phi$']


for df in cycle_dfs[72:73]:
    fig, axs = plt.subplots(ncols=1, nrows=6, sharex=True)
    start_RO_idx = find_start_RO(df)
    max_tether_length_RO, end_RO_idx = find_end_RO_and_max_tether_length(df)
    min_tether_length_RI, end_RI_idx = find_end_RI_and_min_tether_length(df, threshold=0.5)
    df = find_qsm_flight_phases(df)
    max_az_trac, rel_el_angle, avg_el_angle = find_RO_pattern_param(df)

    for ii, ax in enumerate(axs.flatten()):
        ax.plot(df.time[df.flight_phase_index == 1], df[vars_to_plot[ii]][df.flight_phase_index == 1], color = 'blue')
        ax.plot(df.time[df.flight_phase_index == 2], df[vars_to_plot[ii]][df.flight_phase_index == 2], color = 'orange')
        ax.plot(df.time[df.flight_phase_index == 3], df[vars_to_plot[ii]][df.flight_phase_index == 3], color = 'green')
        ax.plot(df.time[df.flight_phase_index == 4], df[vars_to_plot[ii]][df.flight_phase_index == 4], color = 'purple')

        ax.set_ylabel(ylabs[ii])

        ax.axvline(df.time[start_RO_idx], color = 'black', linestyle = ':')
        ax.axvline(df.time[end_RO_idx], color = 'black', linestyle = ':')
        ax.scatter(df.time[end_RO_idx], df[vars_to_plot[ii]][end_RO_idx], color = 'red')
        ax.axvline(df.time[end_RI_idx], color = 'black', linestyle = ':')
        ax.scatter(df.time[end_RI_idx], df[vars_to_plot[ii]][end_RI_idx], color = 'green')

        if vars_to_plot[ii] == 'kite_elevation':
            ax.axhline(avg_el_angle, color = 'black', linestyle = '--')
            ax.axhline(avg_el_angle + rel_el_angle, color = 'black', linestyle = ':')
            ax.axhline(avg_el_angle - rel_el_angle, color = 'black', linestyle = ':')

        if vars_to_plot[ii] == 'kite_azimuth':
            ax.axhline(max_az_trac, color = 'black', linestyle = '--')
            ax.axhline(-max_az_trac, color = 'black', linestyle = '--')



    plt.show()