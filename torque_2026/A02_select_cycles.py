import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('./'))

import pandas as pd
import numpy as np


from scipy.optimize import curve_fit, least_squares

from utils import read_h5_file, convert_dict_to_dataframe, log_wind_profile

ekf_data = read_h5_file('experimental_data/v3_2019-10-08.h5')
ekf_data = convert_dict_to_dataframe(ekf_data)

for col in ekf_data.columns:
    print(col)


cycle_number = []
z_0 = []
v_ref = []
h_ref = []

opt_cost_fun = []
opt_status = []
yaw_rate = []
df_straight = ekf_data[ekf_data['flight_data/turn_straight'] == 'straight']
df_turn = ekf_data[ekf_data['flight_data/turn_straight'] == 'turn']

for cycle_num in range(1, 88):
    flight_data_cycle = pd.read_csv(f'experimental_data/20191008_{cycle_num:04d}.csv')
    mask_cycle = (ekf_data['flight_data/unix_time'] >= flight_data_cycle['time'].iloc[0]) &\
                 (ekf_data['flight_data/unix_time'] < flight_data_cycle['time'].iloc[-1])
    
    ekf_data_cycle = ekf_data[mask_cycle]
    df_straight = ekf_data_cycle[ekf_data_cycle['flight_data/turn_straight'] == 'straight']
    df_turn = ekf_data_cycle[ekf_data_cycle['flight_data/turn_straight'] == 'turn']

    mean_ground_v_w = np.mean(ekf_data_cycle['flight_data/ground_wind_speed'])

    v_w_straight = df_straight['ekf_output/wind_speed_horizontal']
    z_straight = df_straight['ekf_output/kite_position_z']

    v_w_turn = df_turn['ekf_output/wind_speed_horizontal']
    z_turn = df_turn['ekf_output/kite_position_z']

    # Fit
    # Initial guesses: [v_w_ref, z_0]
    initial_guess = [5.0, 0.1]
    # Set bounds to ensure z_0 > 0
    eps = 1e-9
    bounds = ([0, 0.03], [50, 0.5])  # Adjust upper limits if needed

    # Curve fitting
    # popt, pcov = curve_fit(lambda z, v_w_ref, z_0: log_wind_profile(z, z_0, v_w_ref, z_ref),
    #                     z_straight, v_w_straight, p0=initial_guess, bounds=bounds)
    # robust loss functions: 'soft_l1', 'huber', 'cauchy'
    z_ref = 6.
    
    res = least_squares(lambda params: log_wind_profile(z_straight, params[1], params[0], z_ref) - v_w_straight,
                        x0=initial_guess,
                        bounds=bounds,
                        loss='huber')

    #v_w_ref_fit, z_0_fit = popt
    v_w_ref_fit, z_0_fit = res.x


    z_prof = np.linspace(1, np.max(ekf_data_cycle['ekf_output/kite_position_z'])+15)
    v_w_prof = np.array([log_wind_profile(z, z_0_fit, v_w_ref_fit, z_ref) for z in z_prof])

        
    # plt.plot(v_w_prof, z_prof, linestyle=':')
    # plt.scatter(v_w_straight, z_straight, s=2, color='gold')
    # plt.scatter(v_w_turn, z_turn, s=2, color='purple')
    # plt.xlim([0, 15])
    # plt.ylim([0, 300])
    # plt.grid(True)
    # plt.show()

    opt_cost_fun.append(res.cost)
    opt_status.append(res.status)
    cycle_number.append(cycle_num)
    z_0.append(z_0_fit)
    v_ref.append(v_w_ref_fit)
    h_ref.append(z_ref)
    yaw_rate.append(max(df_turn['ekf_output/yaw_rate']))

print(np.mean(yaw_rate))
plt.figure()
plt.scatter(range(len(yaw_rate)), yaw_rate)
plt.show()
wind_prof_data = pd.DataFrame({
    'cycle_number': cycle_number,
    'z_0': z_0,
    'v_ref': v_ref,
    'h_ref': h_ref,
    'opt_cost_fun': opt_cost_fun,
    'opt_status': opt_status
})
wind_prof_data.to_csv('experimental_data/wind_profile_data.csv', index=False)
wind_prof_data = pd.read_csv('experimental_data/wind_profile_data.csv', index_col=None)

# best_profile_data = wind_prof_data.iloc[np.argmin(wind_prof_data.opt_cost_fun)]


# ekf_data_cycle = ekf_data[ekf_data['flight_data/cycle']==best_profile_data.cycle_number]

# df_straight = ekf_data_cycle[ekf_data_cycle['flight_data/turn_straight'] == 'straight']
# df_turn = ekf_data_cycle[ekf_data_cycle['flight_data/turn_straight'] == 'turn']

# mean_ground_v_w = np.mean(ekf_data_cycle['flight_data/ground_wind_speed'])

# v_w_straight = df_straight['ekf_output/wind_speed_horizontal']
# z_straight = df_straight['ekf_output/kite_position_z']

# v_w_turn = df_turn['ekf_output/wind_speed_horizontal']
# z_turn = df_turn['ekf_output/kite_position_z']

# z_prof = np.linspace(4, np.max(ekf_data_cycle['ekf_output/kite_position_z'])+15)
# v_w_prof = log_wind_profile(z_prof, best_profile_data.z_0, best_profile_data.v_ref, best_profile_data.h_ref)

# plt.plot(v_w_prof, z_prof, linestyle=':')
# plt.scatter(v_w_straight, z_straight, s=2, color='gold')
# plt.scatter(v_w_turn, z_turn, s=2, color='purple')
# plt.xlim([0, 15])
# plt.ylim([0, 300])
# plt.grid(True)
# plt.show()


# ----------------------------------------------------------
# 1. Define numeric bin edges
# ----------------------------------------------------------
bin_edges = [3, 3.5, 4, 4.5, 5.5]

# ----------------------------------------------------------
# 2. Categorize v_ref into these bins
# ----------------------------------------------------------
wind_prof_data['v_bin'] = pd.cut(wind_prof_data['v_ref'], bins=bin_edges)

# ----------------------------------------------------------
# 3. Count cycles per bin
# ----------------------------------------------------------
grouped = wind_prof_data.groupby('v_bin').agg({
    'cycle_number': 'count'
})

# ----------------------------------------------------------
# 4. Plot count per bin
# ----------------------------------------------------------
plt.bar(grouped.index.astype(str), grouped['cycle_number'])
plt.xticks(rotation=45)
plt.xlabel("Wind speed bin (m/s)")
plt.ylabel("Number of cycles")
plt.title("Number of cycles per wind speed bin")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. Select best (min opt_cost_fun) profile per bin
# ----------------------------------------------------------
best_profiles = []

for lower_edge, upper_edge in zip(bin_edges[:-1], bin_edges[1:]):
    interval = pd.Interval(left=lower_edge, right=upper_edge, closed='right')
    
    # select the rows in this bin
    bin_df = wind_prof_data[wind_prof_data['v_bin'] == interval]
    
    if bin_df.empty:
        continue
    
    # find row with lowest cost function
    best_idx = bin_df['opt_cost_fun'].idxmin()
    
    best_profiles.append(wind_prof_data.loc[best_idx])

# Convert to DataFrame
best_profiles = pd.DataFrame(best_profiles)


best_profiles_list = []

for lower_edge, upper_edge in zip(bin_edges[:-1], bin_edges[1:]):
    interval = pd.Interval(left=lower_edge, right=upper_edge, closed='right')

    bin_df = wind_prof_data[wind_prof_data['v_bin'] == interval]
    if bin_df.empty:
        continue

    best_idx = bin_df['opt_cost_fun'].idxmin()
    best_profiles_list.append(wind_prof_data.loc[best_idx])

best_profiles = pd.DataFrame(best_profiles_list)

best_profiles.to_csv('experimental_data/best_profiles.csv', index=False)

# -------------------------------------------------------------
# Loop through the best profiles
# -------------------------------------------------------------
for row in best_profiles.itertuples():

    ekf_data_cycle = ekf_data[ekf_data['flight_data/cycle'] == row.cycle_number]

    df_straight = ekf_data_cycle[ekf_data_cycle['flight_data/turn_straight'] == 'straight']
    df_turn     = ekf_data_cycle[ekf_data_cycle['flight_data/turn_straight'] == 'turn']

    mean_ground_v_w = np.mean(ekf_data_cycle['flight_data/ground_wind_speed'])

    v_w_straight = df_straight['ekf_output/wind_speed_horizontal']
    z_straight   = df_straight['ekf_output/kite_position_z']

    v_w_turn = df_turn['ekf_output/wind_speed_horizontal']
    z_turn   = df_turn['ekf_output/kite_position_z']

    z_prof = np.linspace(4, ekf_data_cycle['ekf_output/kite_position_z'].max() + 15)
    v_w_prof = log_wind_profile(z_prof, row.z_0, row.v_ref, row.h_ref)

    plt.plot(v_w_prof, z_prof, linestyle=':')
    plt.scatter(v_w_straight, z_straight, s=2, color='gold')
    plt.scatter(v_w_turn, z_turn, s=2, color='purple')

    plt.xlim([0, 15])
    plt.ylim([0, 300])
    plt.grid(True)
    plt.show()

