import sys
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Arc
import pandas as pd
import numpy as np

from scipy.optimize import curve_fit, least_squares
from utils import read_h5_file, convert_dict_to_dataframe, log_wind_profile


sys.path.append(os.path.abspath('./'))
from awe_pe.exp_validation_utils import *
from awe_pe.utils import *


n = 72
font_size = 18

cycle_dfs = load_process_protologger_file('C:/Users/andre/repos/AWE_production_estimation/experimental_data/', '')
cycle_dfs = cycle_dfs[n:(n+1)]

ekf_data = read_h5_file('experimental_data/v3_2019-10-08.h5')
ekf_data = convert_dict_to_dataframe(ekf_data)

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
bounds = ([0, 0.1-eps], [50, 0.1])  # Adjust upper limits if needed

# Curve fitting
# robust loss functions: 'soft_l1', 'huber', 'cauchy'
z_ref = 6.

res = least_squares(lambda params: log_wind_profile(z_straight, params[1], params[0], z_ref) - v_w_straight,
                    x0=initial_guess,
                    bounds=bounds,
                    loss='huber')

#v_w_ref_fit, z_0_fit = popt
v_w_ref_fit, z_0_fit = res.x

print(v_w_ref_fit, z_0_fit)


z_prof = np.linspace(1, np.max(ekf_data_cycle['ekf_output/kite_position_z'])+15)
v_w_prof = np.array([log_wind_profile(z, z_0_fit, v_w_ref_fit, z_ref) for z in z_prof])


time_vec = ekf_data_cycle['flight_data/unix_time'].values - ekf_data_cycle['flight_data/unix_time'].values[0]
lift_coeff = ekf_data_cycle['ekf_output/wing_lift_coefficient'].values
drag_coeff = ekf_data_cycle['ekf_output/wing_drag_coefficient'].values
wind_speed_at_kite = ekf_data_cycle['ekf_output/wind_speed_horizontal'].values
wind_speed_at_ground = ekf_data_cycle['flight_data/ground_wind_speed'].values
kite_z_pos = ekf_data_cycle['ekf_output/kite_position_z'].values

idx_RO  = find_start_RO(flight_data_cycle)
time_start_RO = flight_data_cycle['time'].values[idx_RO] - flight_data_cycle['time'].values[0] 
_, idx_RO  = find_end_RO_and_max_tether_length(flight_data_cycle)
time_end_RO = flight_data_cycle['time'].values[idx_RO] - flight_data_cycle['time'].values[0] 
L_min, idx_RI = find_end_RI_and_min_tether_length(flight_data_cycle)
time_end_RI = flight_data_cycle['time'].values[idx_RI] - flight_data_cycle['time'].values[0] 
time_end_RIRO = ekf_data_cycle['flight_data/unix_time'].values[-1] - ekf_data_cycle['flight_data/unix_time'].values[0]


start_RO = find_start_RO(flight_data_cycle)
L_max, end_RO = find_end_RO_and_max_tether_length(flight_data_cycle)
RO_flight_data = flight_data_cycle[start_RO:end_RO]
x_pos = RO_flight_data.ground_tether_length*np.cos(RO_flight_data.kite_elevation)*np.cos(RO_flight_data.kite_azimuth)
y_pos = RO_flight_data.ground_tether_length*np.cos(RO_flight_data.kite_elevation)*np.sin(RO_flight_data.kite_azimuth)
z_pos = RO_flight_data.ground_tether_length*np.sin(RO_flight_data.kite_elevation)

fig = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(2, 3, height_ratios=[1,1], width_ratios=[1,2.5,2])

# ---------------------------
# First subplot (1,3,1)
# ---------------------------
plt.rcParams.update({'font.size': font_size, 'axes.linewidth': 1.5})

ax1 = fig.add_subplot(gs[:, 0])   # spans both rows, first column
ax1.scatter(v_w_straight, z_straight, s=2, color='gold', label = 'Straight')
ax1.scatter(v_w_turn, z_turn, s=2, color='gray', label='Turns')
ax1.plot(v_w_prof, z_prof, linewidth=2, label='Log fit')


ax1.boxplot(
    x=ekf_data_cycle['flight_data/ground_wind_speed'],
    orientation='horizontal',
    positions=[6],
    widths=[12],
    boxprops=dict(linewidth=1.5, color='black'),
    medianprops=dict(linewidth=2, color='red'),
    capprops=dict(linewidth=1.5, color='black'),
    whiskerprops=dict(linewidth=1.5, color='black', linestyle='-')
)

ax1.text(0.5, 285, s='a)', fontdict={'fontsize': font_size})

#label = r'$v_{w, 6m}=$' + f"{v_w_ref_fit:.1f} m/s" + '\n' +  r'$z_{0, 6m}=$' + f"{z_0_fit:.1f} m"
#ax1.text(2, 52, s=label)
ax1.set_yticks(range(0, 350, 50), [f"{tick}" for tick in range(0, 350, 50)])
ax1.set_xlabel(r'$v_w$ (m/s)')
ax1.set_ylabel('Altitude (m)')
ax1.set_xlim([0,15])
ax1.set_ylim([0,300])
leg = ax1.legend(edgecolor='black', draggable=True, fancybox=False, markerscale=6)
frame = leg.get_frame()
frame.set_linewidth(1.5)
frame.set_facecolor('white')   # white background
frame.set_alpha(1.0)           # fully opaque


ax1.grid(True, axis='y')

# ---------------------------
# Second subplot (1,3,2)
# ---------------------------
ax2_top = fig.add_subplot(gs[0, 1])     # top
ax2_bottom = fig.add_subplot(gs[1, 1])  # bottom

lns1 = ax2_top.plot(time_vec, wind_speed_at_kite, linewidth=2, label=r'$v_w$ at kite')
ax2_top.set_ylabel(r'$v_w$ (m/s)')
y_min, y_max = ax2_top.get_ylim()
ax2_top.vlines([time_start_RO, time_end_RO, time_end_RI],
           ymin=[y_min,y_min,y_min], ymax=[y_max,y_max,y_max],
           colors=['gray','gray','gray'], linestyles=[':',':',':'])
ax2_top.set_ylim(y_min, y_max)
#ax2_top.set_xlabel('Time (s)')

ax2_top_bis = ax2_top.twinx()
lns2 = ax2_top_bis.plot(time_vec, kite_z_pos, linewidth=2, color='green', label='Kite altitude')
ax2_top_bis.set_ylabel('Altitude (m)')
ax2_top.text(-4, 12, s='b)', fontdict={'fontsize': font_size})
ax2_top.set_xticklabels('')

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]
leg0 = ax2_top_bis.legend(
    lns, labs,
    ncols=2,
    fancybox=False,
    draggable=True,
    framealpha=1.0,
    edgecolor='black'
)

leg0.set_zorder(1000)

frame0 = leg0.get_frame()
frame0.set_facecolor((1, 1, 1, 1))
frame0.set_linewidth(1.5)

ax2_top.set_facecolor('white')
ax2_top.patch.set_alpha(1.0)
ax2_top_bis.set_facecolor('white')
ax2_top_bis.patch.set_alpha(1.0)

fig = ax2_top.figure
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)


ax2_bottom.plot(time_vec, lift_coeff, linewidth=2, label=r'$C_{Lw}$')
ax2_bottom.plot(time_vec, drag_coeff, linewidth=2, label=r'$C_{Dw}$')
y_min, y_max = 0, 1
ax2_bottom.vlines([time_start_RO, time_end_RO, time_end_RI],
           ymin=[y_min,y_min,y_min], ymax=[y_max,y_max,y_max],
           colors=['gray','gray','gray'], linestyles=[':',':',':'])
ax2_bottom.set_ylim(y_min, y_max)

ax2_bottom.text(-4, 0.88, s='c)', fontdict={'fontsize': font_size})

ax2_bottom.text(30, 0.88, 'Reel-out')
ax2_bottom.text(95, 0.7, 'Reel-in')
ax2_bottom.set_ylabel('(-)')
ax2_bottom.set_xlabel('Time (s)')

leg1 = ax2_bottom.legend(edgecolor='black', draggable=True, fancybox=False, framealpha=1.0, ncols=2)
frame1 = leg1.get_frame()
frame1.set_linewidth(1.5)
frame1.set_facecolor('white')   # white background



# ---------------------------
# Third subplot (split into two)
# ---------------------------
ax3_top = fig.add_subplot(gs[0, 2])     # top-right
ax3_bottom = fig.add_subplot(gs[1, 2])  # bottom-right

# Now you can add your two different plots here:
# Example:

phi, theta_rel, theta_avg = find_RO_pattern_param(flight_data_cycle[flight_data_cycle.flight_phase_index == 1])
x_ = np.array([0, 320])
x_range = x_*np.cos(theta_avg)
y_range = x_*np.sin(theta_avg)    
ax3_top.plot(x_pos, z_pos, linewidth = 2)
ax3_top.plot(x_range, y_range, linestyle='--', color='gray', linewidth = 1.5)
x_range = x_*np.cos(theta_avg+theta_rel)
y_range = x_*np.sin(theta_avg+theta_rel)    
ax3_top.plot(x_range, y_range, linestyle='--', color='gray', linewidth = 1.5)
x_range = x_*np.cos(theta_avg-theta_rel)
y_range = x_*np.sin(theta_avg-theta_rel)    
ax3_top.plot(x_range, y_range, linestyle='--', color='gray', linewidth = 1.5)
arc = Arc(xy=(0, 0), width=2*L_max, height=2*L_max, angle=0, theta1=np.rad2deg(theta_avg-theta_rel) - 5, theta2=np.rad2deg(theta_avg+theta_rel) + 5,\
           linestyle='--', color='gray', linewidth = 1.5)
ax3_top.add_patch(arc)
ax3_top.text(240, 200, s=r"$L_{max}$")#=$" + f"{L_max:.0f} m")

arc = Arc(xy=(0, 0), width=2*L_min, height=2*L_min, angle=0, theta1=np.rad2deg(theta_avg-theta_rel) - 5, theta2=np.rad2deg(theta_avg+theta_rel) + 5,\
           linestyle='--', color='gray', linewidth = 1.5) 
ax3_top.add_patch(arc)
ax3_top.text(185, 90, s=r"$L_{min}$")#=$" + f"{L_min:.0f} m")

arc = Arc(xy=(0, 0), width=160, height=160, angle=0, theta1=0, theta2=np.rad2deg(theta_avg),\
           linestyle='-', color='black', linewidth = 1.5) 
ax3_top.add_patch(arc)
ax3_top.text(85, 10, s=r"$\theta_{avg}$")#=$" + f"{np.rad2deg(theta_avg):.1f} deg")

arc = Arc(xy=(0, 0), width=240, height=240, angle=0, theta1=np.rad2deg(theta_avg-theta_rel), theta2=np.rad2deg(theta_avg),\
           linestyle='-', color='black', linewidth = 1.5) 
ax3_top.add_patch(arc)

arc = Arc(xy=(0, 0), width=250, height=250, angle=0, theta1=np.rad2deg(theta_avg-theta_rel), theta2=np.rad2deg(theta_avg),\
           linestyle='-', color='black', linewidth = 1.5) 
ax3_top.add_patch(arc)

ax3_top.text(110, 50, s=r"$\theta_{rel}$")#=$" + f"{np.rad2deg(theta_rel):.1f} deg")

ax3_top.set_ylim(0)
ax3_top.set_xlim([-15, 315])

ax3_top.set_aspect('equal', 'box')
ax3_top.set_ylabel('Altitude (m)')
ax3_top.set_xticklabels('')
#ax3_top.set_xlabel('X pos. (m)')


ax3_top.text(-8, 195, s='d)', fontdict={'fontsize': font_size})



x_range = x_*np.cos(phi)
y_range = x_*np.sin(phi)    
ax3_bottom.plot(x_pos, y_pos, linewidth = 2)
ax3_bottom.plot(x_range, y_range, linestyle='--', color='gray', linewidth = 1.5)
ax3_bottom.plot(x_range, -y_range, linestyle='--', color='gray', linewidth = 1.5)
ax3_bottom.plot([0, max(x_range)], [0, 0], linestyle='--', color='gray', linewidth = 1.5 )
ax3_bottom.set_ylabel('Y pos. (m)')
ax3_bottom.set_xlabel('X pos. (m)')
ax3_bottom.set_xlim([-15, 315])
ax3_bottom.set_aspect('equal', 'box')

arc = Arc(xy=(0, 0), width=90, height=90, angle=0, theta1=0, theta2=np.rad2deg(phi),\
           linestyle='-', color='black', linewidth = 1.5) 
ax3_bottom.add_patch(arc)
ax3_bottom.text(50, 5, s=r"$|\phi_{max}|$")#=$" + f"{np.rad2deg(phi):.0f} deg")


ax3_bottom.text(-8, 85, s='e)', fontdict={'fontsize': font_size})




plt.tight_layout()
plt.show()


print(L_min)
print(L_max)
print(np.rad2deg(theta_avg))
print(np.rad2deg(theta_rel))
print(np.rad2deg(phi))


powered_mask = ekf_data_cycle['flight_data/powered']=='powered'

ekf_data_cycle.dropna(subset=['ekf_output/wing_lift_coefficient', 'ekf_output/wing_drag_coefficient'], inplace=True)
print('Powered state:')

print(np.mean(ekf_data_cycle[powered_mask]['ekf_output/wing_lift_coefficient']))
print(np.mean(ekf_data_cycle[powered_mask]['ekf_output/wing_drag_coefficient']))
print(np.mean(ekf_data_cycle[powered_mask]['ekf_output/wing_drag_coefficient'])+0.06)

print('Depowered state:')
powered_mask = ekf_data_cycle['flight_data/powered']!='powered'

print(np.mean(ekf_data_cycle[powered_mask]['ekf_output/wing_lift_coefficient']))
print(np.mean(ekf_data_cycle[powered_mask]['ekf_output/wing_drag_coefficient']))
print(np.mean(ekf_data_cycle[powered_mask]['ekf_output/wing_drag_coefficient'])+0.06)

