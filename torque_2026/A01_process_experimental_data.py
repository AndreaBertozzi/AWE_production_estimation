import sys
import os
import datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.append(os.path.abspath('./'))

from awe_pe.exp_validation_utils import *
from awe_pe.utils import *
from utils import read_h5_file, convert_dict_to_dataframe, log_wind_profile
from scipy.optimize import curve_fit, least_squares

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

# # ------------------------------ Load Protologger ------------------------------------
# protologger = pd.read_csv('experimental_data/ProtoLogger.csv', sep = ' ',
#                            low_memory=False)


# # --------------------------------- Load EKF -----------------------------------------
# ekf_data = read_h5_file('experimental_data/v3_2019-10-08.h5')
# ekf_data = convert_dict_to_dataframe(ekf_data)

# # ---------- Assign cycle value to Protologger based on segmentation of EKF ----------
# for cycle_num in np.unique(ekf_data['flight_data/cycle']):
#     ekf_data_cycle = ekf_data[ekf_data['flight_data/cycle'] == cycle_num]
    
#     time_start = ekf_data_cycle['flight_data/unix_time'].iloc[0]
#     time_end = ekf_data_cycle['flight_data/unix_time'].iloc[-1]
    
#     mask = (protologger['time'] >= time_start) & (protologger['time'] < time_end)

#     if mask is None:
#         print('Something wrong!')
#     protologger.loc[mask, 'cycle'] = cycle_num
    
# protologger.dropna(subset=['cycle'], inplace=True)
# protologger.to_csv('experimental_data/EKF_corrected.csv', index=False)


def wrap_to_pi(angle_rad):
    """
    Wrap angle in radians to [-π, π].
    
    Parameters:
        angle_rad (float or np.ndarray): Angle(s) in radians.
        
    Returns:
        float or np.ndarray: Wrapped angle(s) in radians.
    """
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

cycle_dfs = load_process_protologger_file('C:/Users/andre/repos/AWE_production_estimation/experimental_data/', '')
#cycle_dfs = cycle_dfs[72:73]

n = 72

cycle_dfs = cycle_dfs[n:(n+1)]


with open("config/config_V3_speed.yaml") as f:
    config = yaml.safe_load(f)

sys_props = parse_system_properties_and_bounds(config)
sys_props = SystemProperties(sys_props)

wind_profile_data = pd.read_csv('experimental_data/wind_profile_data.csv')
wind_profile_data = wind_profile_data[64:65]

ekf_data = read_h5_file('experimental_data/v3_2019-10-08.h5')
ekf_data = convert_dict_to_dataframe(ekf_data)
df_straight = ekf_data[ekf_data['flight_data/turn_straight'] == 'straight']
df_turn = ekf_data[ekf_data['flight_data/turn_straight'] == 'turn']

flight_data_cycle = cycle_dfs[0]
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
bounds = ([0, 0.03], [50, 20])  # Adjust upper limits if needed

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

wind_prof_data = pd.DataFrame({
    'z_0': z_0_fit,
    'v_ref': v_w_ref_fit,
    'h_ref': 6}, index=[0]
)

all_sim_dataframes, all_cycle_res_sim, all_exp_dataframes, all_cycle_res_exp =\
      run_simulations_from_list_of_exp_dataframes(cycle_dfs, sys_props, wind_profile_data, control='hybrid')



font_size = 22
# for df_sim, df_exp, ii in zip(all_sim_dataframes, all_exp_dataframes, range(0, len(all_sim_dataframes))):

#     ec = '#00395d'
#     sc = '#00aeef'
#     lim = [np.min(df_sim.time), 74.2]
#     plt.subplots(3,1,sharex=True)
    
#     plt.subplot(3,1,1)
#     plt.plot(df_exp.time-df_exp.time[0], 180/np.pi*df_exp.kite_elevation, color = ec)
#     plt.plot(df_sim.time, 180/np.pi*df_sim.elevation_angle, color = sc, linestyle = '--')
#     plt.xlim(lim)
#     plt.ylim([180/np.pi*(-0.15+np.min(df_sim.elevation_angle)), 180/np.pi*(np.max(df_sim.elevation_angle)+0.15)])
#     plt.ylabel(r'$\theta$ [deg]', fontsize=fs)
#     plt.tick_params(axis='both', labelsize=fs)
#     # Move legend outside to the right
#     plt.legend(['Experiment', 'Simulation'], loc='upper left', bbox_to_anchor=(1, 1), fontsize=fs)
#     plt.grid(visible=True, axis='x')
    
    
#     plt.subplot(3,1,2)
#     plt.plot(df_exp.time-df_exp.time[0], 180/np.pi*df_exp.kite_azimuth, color = ec)
#     plt.plot(df_sim.time, 180/np.pi*df_sim.azimuth_angle, color = sc, linestyle = '--')
#     plt.xlim(lim)
#     plt.ylabel(r'$\phi$ [deg]', fontsize=fs)
#     plt.grid(visible=True, axis='x')
#     plt.tick_params(axis='both', labelsize=fs)

#     plt.subplot(3,1,3)
#     plt.plot(df_exp.time-df_exp.time[0], 180/np.pi*df_exp.kite_heading-180, color = ec)
#     plt.plot(df_sim.time, 180/np.pi*df_sim.course_angle, color = sc, linestyle = '--')
#     plt.xlim(lim)
#     plt.ylabel(r'$\chi$ [deg]', fontsize=fs)
#     plt.xlabel(r'Time [s]', fontsize=fs)
#     plt.tick_params(axis='both', labelsize=fs)
#     plt.grid(visible=True, axis='x') 

#     cycle_to_cycle_plot(df_sim, df_exp, all_cycle_res_sim.iloc[ii], all_cycle_res_exp.iloc[ii])
#     #plt.tight_layout()
#     plt.show()

# for df_sim, df_exp, ii in zip(all_sim_dataframes, all_exp_dataframes, range(0, len(all_sim_dataframes))):
#     cycle_to_cycle_plot(df_sim, df_exp, all_cycle_res_sim.iloc[ii], all_cycle_res_exp.iloc[ii])
#     plt.show()


df_sim = all_sim_dataframes[0]
df_exp = all_exp_dataframes[0]
time_exp = df_exp.time-df_exp.time[0]
time_sim = df_sim.time  

# df_sim.to_csv('baseline_reference_sim.csv')
# df_exp.to_csv('baseline_reference_exp.csv')

# plt.figure()
# plt.plot(df_exp.time, df_exp.ground_tether_force*9.806*df_exp.ground_tether_reelout_speed)
# plt.plot(df_exp.time, df_exp.ground_mech_power)
# plt.show()


df_exp = find_qsm_flight_phases(df_exp)

sim_ph_power = np.mean(df_sim.ground_tether_force*df_sim.ground_tether_reelout_speed)
exp_ph_power = np.mean(df_exp.ground_tether_force*9.806*df_exp.ground_tether_reelout_speed)

df_sim_phase = df_sim[df_sim.flight_phase_index == 1]
df_exp_phase = df_exp[df_exp.flight_phase_index == 1]
sim_ph_power_RO = np.mean(df_sim_phase.ground_tether_force*df_sim_phase.ground_tether_reelout_speed)
exp_ph_power_RO = np.mean(df_exp_phase.ground_tether_force*9.806*df_exp_phase.ground_tether_reelout_speed)

df_sim_phase = df_sim[df_sim.flight_phase_index == 3]
df_exp_phase = df_exp[df_exp.flight_phase_index == 3]
sim_ph_power_RI = np.mean(df_sim_phase.ground_tether_force*df_sim_phase.ground_tether_reelout_speed)
exp_ph_power_RI = np.mean(df_exp_phase.ground_tether_force*9.806*df_exp_phase.ground_tether_reelout_speed)

df_sim_phase = df_sim[df_sim.flight_phase_index == 4]
df_exp_phase = df_exp[df_exp.flight_phase_index == 4]
sim_ph_power_RIRO = np.mean(df_sim_phase.ground_tether_force*df_sim_phase.ground_tether_reelout_speed)
exp_ph_power_RIRO = np.mean(df_exp_phase.ground_tether_force*9.806*df_exp_phase.ground_tether_reelout_speed)



fig = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1], width_ratios=[1.2,1.2,1])
plt.rcParams.update({'font.size': font_size, 'axes.linewidth': 1.5})

ax1_top = fig.add_subplot(gs[0, 0])     # top
ax1_middle = fig.add_subplot(gs[1, 0])  # middle
ax1_bottom = fig.add_subplot(gs[2, 0])  # bottom

tether_length_exp = df_exp.ground_tether_length
tether_length_sim = df_sim.ground_tether_length

v_r_exp = df_exp.ground_tether_reelout_speed
v_r_sim = df_sim.ground_tether_reelout_speed

F_t_exp = df_exp.ground_tether_force
F_t_sim = df_sim.ground_tether_force

ax1_top.plot(time_exp, tether_length_exp, label='Experiment', linewidth=2)
ax1_top.plot(time_sim, tether_length_sim, label='Simulation', linewidth=2)
ax1_top.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax1_top.set_xticklabels([])
ax1_top.set_ylabel('$L$ (m)')
ax1_top.text(-4, 290, s='a)', fontdict={'fontsize': font_size})
leg = ax1_top.legend(edgecolor='black', draggable=True, fancybox=False, ncols=2)
frame = leg.get_frame()
frame.set_linewidth(1.5)
frame.set_facecolor('white')   # white background
frame.set_alpha(1.0)           # fully opaque

ax1_middle.plot(time_exp, v_r_exp, linewidth=2)
ax1_middle.plot(time_sim, v_r_sim, linewidth=2)
ax1_middle.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax1_middle.set_xticklabels([])
ax1_middle.set_ylabel(r'$v_\mathrm{r}$ (m/s)')

ax1_middle.text(-4, -0.2, s='b)', fontdict={'fontsize': font_size})

ax1_bottom.plot(time_exp, F_t_exp*9.806/1000, linewidth=2)
ax1_bottom.plot(time_sim, F_t_sim/1000, linewidth=2)
ax1_bottom.set_xlabel('Time (s)')
ax1_bottom.set_ylabel(r'$F_\mathrm{t}$ (kN)')
ax1_bottom.text(-4, 5.1, s='c)', fontdict={'fontsize': font_size})
ax1_bottom.set_yticks([1,3,5])
ax1_bottom.set_xticks([0, 20, 40, 60, 80, 100, 120])


ax2_top = fig.add_subplot(gs[0, 1])     # top
ax2_middle = fig.add_subplot(gs[1, 1])  # middle
ax2_bottom = fig.add_subplot(gs[2, 1])  # bottom


elev_exp = df_exp.kite_elevation
elev_sim = df_sim.elevation_angle

az_exp = df_exp.kite_azimuth
az_sim = df_sim.azimuth_angle

course_exp = np.unwrap(df_exp.kite_course) - np.pi
course_sim = df_sim.course_angle
course_sim[course_sim==np.pi] = -np.pi

ax2_top.plot(time_exp, np.rad2deg(elev_exp), linewidth=2)
ax2_top.plot(time_sim, np.rad2deg(elev_sim), linewidth=2)
ax2_top.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax2_top.set_xticklabels([])
ax2_top.set_ylabel(r'$\theta$ (deg)')
ax2_top.text(-4, 71, s='d)', fontdict={'fontsize': font_size})

ax2_middle.plot(time_exp, np.rad2deg(az_exp), linewidth=2)
ax2_middle.plot(time_sim, np.rad2deg(az_sim), linewidth=2)
ax2_middle.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax2_middle.set_xticklabels([])
ax2_middle.set_ylabel(r'$\phi$ (deg)')
ax2_middle.text(-4, 13, s='e)', fontdict={'fontsize': font_size})

ax2_bottom.plot(time_exp, np.rad2deg(course_exp), linewidth=2)
ax2_bottom.plot(time_sim, np.rad2deg(course_sim), linewidth=2)
ax2_bottom.set_xlabel('Time (s)')
ax2_bottom.set_ylabel(r'$\chi$ (deg)')
ax2_bottom.text(-4, 71, s='f)', fontdict={'fontsize': font_size})
ax2_bottom.set_xticks([0, 20, 40, 60, 80, 100, 120])


gs_col3 = gs[:, 2].subgridspec(
    2, 1,
    height_ratios=[1, 1],   # equal → each spans 1.5 original rows
    hspace=0.25
)

ax3_top = fig.add_subplot(gs_col3[0, 0])
ax3_bottom = fig.add_subplot(gs_col3[1, 0])

x_pos_exp = df_exp.ground_tether_length*np.cos(df_exp.kite_elevation)*np.cos(df_exp.kite_azimuth)
y_pos_exp = df_exp.ground_tether_length*np.cos(df_exp.kite_elevation)*np.sin(df_exp.kite_azimuth)
z_pos_exp = df_exp.ground_tether_length*np.sin(df_exp.kite_elevation)
x_pos_sim = df_sim.ground_tether_length*np.cos(df_sim.elevation_angle)*np.cos(df_sim.azimuth_angle)
y_pos_sim = df_sim.ground_tether_length*np.cos(df_sim.elevation_angle)*np.sin(df_sim.azimuth_angle)
z_pos_sim = df_sim.ground_tether_length*np.sin(df_sim.elevation_angle)

ax3_top.plot(x_pos_exp, z_pos_exp, linewidth=2)
ax3_top.plot(x_pos_sim, z_pos_sim, linewidth=2)
ax3_top.set_xticks([0, 100, 200, 300])
ax3_top.set_xticklabels([])
x_lims = [-10, 310]
ax3_top.set_xlim(x_lims)
ax3_top.set_ylim([-10, 260])
ax3_top.set_aspect('equal', 'box')
ax3_top.set_ylabel('Altitude (m)')
ax3_top.text(-4, 230, s='g)', fontdict={'fontsize': font_size})



ax3_bottom.plot(x_pos_exp, y_pos_exp, linewidth=2)
ax3_bottom.plot(x_pos_sim, y_pos_sim, linewidth=2)
ax3_bottom.set_aspect('equal', 'box')
ax3_bottom.set_xlim(x_lims)
ax3_bottom.set_ylim([-135, 135])
ax3_bottom.set_ylabel('Y pos. (m)')
ax3_bottom.set_xlabel('X pos. (m)')
ax3_bottom.set_xticks([0, 100, 200, 300])
ax3_bottom.text(-4, 100, s='h)', fontdict={'fontsize': font_size})



plt.tight_layout()
plt.show()

cycle_sim, cycle_exp = all_cycle_res_sim.iloc[0], all_cycle_res_exp.iloc[0]


plt.rcParams.update({'font.size': font_size, 'axes.linewidth': 1.5, })
fig, (ax0, ax1) = plt.subplots(1, 2, width_ratios=[1.5, 1], figsize=(18,3))


n = 10
window = np.ones(n) / n
smooth = np.convolve(df_exp.ground_mech_power/1000, window, mode='same')


ax0.plot(time_exp, df_exp.ground_tether_force*9.806*df_exp.ground_tether_reelout_speed/1000,
          color = 'C0', linewidth = 2, label = r'$P_\mathrm{m}$ (Experiment)')
ax0.plot(time_sim, df_sim.ground_mech_power/1000, color='C1', linewidth = 2, 
         label = r'$P_\mathrm{m}$ (Simulation)')
ax0.plot(time_exp, smooth, color='C3', linewidth = 2, linestyle = '--',
         label = r'$P^{*}_\mathrm{m}$ (Experiment)')
ax0.set_ylabel("Power (kW)")
ax0.set_xlabel("Time (s)")
ax0.text(-5, 30, s='a)', fontdict={'fontsize': font_size})

leg = ax0.legend(
    edgecolor='black',
    framealpha=1.0,     # fully opaque
    fancybox=False,
    facecolor='white',
    draggable=True
)
leg.get_frame().set_linewidth(1.5)


labels = ["RO", "RI", "RIRO", "Cycle"]

sim_values = [
    sim_ph_power_RO/1000,
    sim_ph_power_RI/1000,
    sim_ph_power_RIRO/1000,
    sim_ph_power/1000,
]

exp_values = [
    exp_ph_power_RO/1000,
    exp_ph_power_RI/1000,
    exp_ph_power_RIRO/1000,
    exp_ph_power/1000,
]

errors = (np.array(sim_values)-np.array(exp_values))/np.array(exp_values)

x = np.arange(len(labels))
width = 0.35
# Send grid to the background
ax1.set_axisbelow(True)
ax1.grid(True ,axis='y')  # Show grid, but send it to the background
ax1.bar(x + width / 2, exp_values, width, label="Experiment", linewidth = 1.5, edgecolor='black')
ax1.bar(x - width / 2, sim_values, width, label="Simulation", linewidth = 1.5, edgecolor='black')

for ii, x_pos in enumerate(x):
    if exp_values[ii] >= 0:
        y_pos = np.max([exp_values[ii], sim_values[ii]])
        if errors[ii] >=0:
            ax1.text(x_pos- width -0.1, y_pos+0.4, rf"$\Delta\approx+{errors[ii]*100:.0f}\%$", fontsize = 18)
        else:
            ax1.text(x_pos- width -0.1, y_pos+0.4, rf"$\Delta\approx{errors[ii]*100:.0f}\%$", fontsize = 18)
    else:
        y_pos = np.min([exp_values[ii], sim_values[ii]])
        if errors[ii] >=0:
            ax1.text(x_pos-width -0.1, y_pos-1., rf"$\Delta\approx+{errors[ii]*100:.0f}\%$", fontsize = 18)
        else:
            ax1.text(x_pos-width -0.1, y_pos-1., rf"$\Delta\approx{errors[ii]*100:.0f}\%$", fontsize = 18)

    

xlb, xub = ax1.get_xlim()
ax1.hlines(0, xlb, xub, colors=['black'], linewidth = 1.5)
# Adding labels, title, and formatting
# ax1.set_xlabel("Power Metrics")
ax1.set_ylabel("Power (kW)")
#ax1.set_title("Comparison of average mechanical power")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
leg = ax1.legend(
    edgecolor='black',
    framealpha=1.0,     # fully opaque
    fancybox=False,
    facecolor='white',
    draggable = True
)
leg.get_frame().set_linewidth(1.5)
ax1.text(-0.7, 5., s='b)', fontdict={'fontsize': font_size})
ax1.set_ylim([-5, 6])

plt.tight_layout()
plt.show()